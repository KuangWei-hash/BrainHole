"""Phase-aware multi-frame fusion network.

Design follows the v11 measurement results:

* The 16 sampling phases tile the high-resolution mosaic exactly, so each frame
  measures a disjoint residue class of the output pixels. The phase offset is
  therefore known exactly and costs nothing to apply.
* An optional local motion offset is estimated on the low-resolution grid.
* Historical support determines where the corresponding inputs can contribute.

Spatial re-sampling is per CFA plane or within feature channels. One step on the
2x80 grid is four high-resolution CFA plane steps, or eight RAW mosaic pixels.
"""
import torch
from torch import nn
from torch.nn import functional as F

from baseline import upscale

PLANE_H, PLANE_W = 2, 80
HR_PLANE_H, HR_PLANE_W = 8, 320
WIDTH, HEIGHT = 640, 16
PHASES = 16


def phase_offsets(phases):
    """Plane-unit sampling offsets (px, py) of 0-based phases, both within 0..3."""
    if phases.dtype != torch.long:
        raise ValueError("phases must be int64")
    return phases % 4, phases // 4


def phase_shift(reference, other):
    """Source sampling offset in high-resolution CFA plane steps."""
    px_ref, py_ref = phase_offsets(reference)
    px_other, py_other = phase_offsets(other)
    return px_ref - px_other, py_ref - py_other


def make_bilinear(decoded, phases):
    """Phase-aligned bilinear base image, mosaic domain: ``[B, T, 1, 16, 640]``."""
    batch = decoded.shape[0]
    return upscale(decoded.reshape(-1, 1, 4, 160), phases.reshape(-1)).reshape(batch, -1, 1, HEIGHT, WIDTH)


def _base_grid(height, width, device, dtype):
    ys = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y


def warp(features, dx, dy, *, return_validity=False):
    """Warp ``features`` ``[B, C, H, W]`` by pixel offsets ``dx``/``dy`` ``[B, H, W]``."""
    _, _, height, width = features.shape
    grid_x, grid_y = _base_grid(height, width, features.device, features.dtype)
    step_x = 2.0 / max(width - 1, 1)
    step_y = 2.0 / max(height - 1, 1)
    grid = torch.stack((grid_x[None] + dx * step_x, grid_y[None] + dy * step_y), dim=-1)
    aligned = F.grid_sample(features, grid, align_corners=True, padding_mode="zeros")
    if return_validity:
        return aligned, grid.abs().le(1).all(dim=-1).unsqueeze(1)
    return aligned


class OffsetNet(nn.Module):
    """Predicts a small continuous motion offset for one frame relative to the current one."""

    def __init__(self, channels, max_offset=1.0):
        super().__init__()
        self.max_offset = max_offset
        self.net = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1, padding_mode="replicate"), nn.SiLU(),
            nn.Conv2d(channels, 2, 3, padding=1, padding_mode="replicate"))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, current, other):
        offset = torch.tanh(self.net(torch.cat((other, current), dim=1)))
        return offset[:, 0] * self.max_offset, offset[:, 1] * self.max_offset


class WeightNet(nn.Module):
    """Per-pixel fusion weights over the frames of a window."""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1, padding_mode="replicate"), nn.SiLU(),
            nn.Conv2d(channels, 1, 3, padding=1, padding_mode="replicate"))

    def forward(self, current, aligned):
        return self.net(torch.cat((aligned, current), dim=1))


class FusionNet(nn.Module):
    """Multi-frame fusion with phase-aware alignment and an exact-measurement branch.

    Args:
        frames: number of frames in the window (the current frame first).
        channels: feature width in the low resolution plane grid.
        blocks: residual blocks in the reconstruction head.
        max_offset: largest learned motion offset in low-resolution plane steps.
    """

    def __init__(self, frames=4, channels=32, blocks=2, max_offset=1.0, alignment="motion", anchor="current"):
        super().__init__()
        if frames < 1 or channels < 8 or channels % 8 or blocks < 1:
            raise ValueError("invalid frames, channels or blocks")
        if alignment not in ("phase", "motion"):
            raise ValueError("unknown alignment")
        if anchor not in ("current", "phase-mosaic") or (anchor == "phase-mosaic" and frames != PHASES):
            raise ValueError("phase-mosaic anchor requires sixteen phases")
        self.frames, self.channels = frames, channels
        self.alignment = alignment
        self.anchor = anchor
        self.encoder = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1, padding_mode="replicate"), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate"), nn.SiLU())
        self.offset = OffsetNet(channels, max_offset)
        self.weight = WeightNet(channels)
        self.exact_encoder = nn.Sequential(
            nn.Conv2d(5, channels, 3, padding=1, padding_mode="replicate"), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate"), nn.SiLU())
        self.exact_weight = WeightNet(channels)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate"), nn.SiLU(),
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate"))
        head_in = 2 * channels + 1 + 4
        self.head = nn.Sequential(
            nn.Conv2d(head_in, channels, 3, padding=1, padding_mode="replicate"), nn.SiLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate"),
                          nn.SiLU(),
                          nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate"))
            for _ in range(blocks)])
        self.tail = nn.Conv2d(channels, 4, 3, padding=1, padding_mode="replicate")
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def exact_map(self, decoded, phases):
        """Sparse high-resolution plane map of the exactly measured samples.

        Returns ``(values, mask)`` with shape ``[B, T, 4, 8, 320]``.
        """
        batch, frames = decoded.shape[0], decoded.shape[1]
        planes = F.pixel_unshuffle(decoded.reshape(batch * frames, 1, 4, 160), 2)
        planes = planes.reshape(batch, frames, 4, PLANE_H, PLANE_W)
        values = planes.new_zeros(batch, frames, 4, HR_PLANE_H * HR_PLANE_W)
        mask = torch.zeros_like(values, dtype=torch.bool)
        px, py = phase_offsets(phases)
        rows_base = torch.arange(PLANE_H, device=decoded.device)
        cols_base = torch.arange(PLANE_W, device=decoded.device)
        rows = 4 * rows_base + py[..., None]
        cols = 4 * cols_base + px[..., None]
        destination = (rows[..., :, None] * HR_PLANE_W + cols[..., None, :]).flatten(-2)
        destination = destination.unsqueeze(2).expand(-1, -1, 4, -1)
        values.scatter_(-1, destination, planes.flatten(-2))
        mask.scatter_(-1, destination, True)
        return values.unflatten(-1, (HR_PLANE_H, HR_PLANE_W)), mask.unflatten(-1, (HR_PLANE_H, HR_PLANE_W))

    def forward(self, decoded, phases, *, return_support=False):
        """``decoded`` ``[B, T, 1, 4, 160]``, ``phases`` ``[B, T]`` -> ``[B, 1, 16, 640]``."""
        if decoded.dim() != 5 or decoded.shape[2] != 1 or decoded.shape[3] != 4 or decoded.shape[4] != 160:
            raise ValueError("expected decoded [B,T,1,4,160]")
        batch, frames = decoded.shape[0], decoded.shape[1]
        if phases.shape != (batch, frames) or phases.dtype != torch.long:
            raise ValueError("expected phases [B,T] as int64")
        if frames != self.frames:
            raise ValueError(f"model expects {self.frames} frames")
        if phases.min() < 0 or phases.max() > PHASES - 1:
            raise ValueError("phase must be within 0..15")

        planes = F.pixel_unshuffle(decoded.reshape(batch * frames, 1, 4, 160), 2)
        planes = planes.reshape(batch, frames, 4, PLANE_H, PLANE_W)
        features = self.encoder(planes.reshape(batch * frames, 4, PLANE_H, PLANE_W))
        features = features.reshape(batch, frames, self.channels, PLANE_H, PLANE_W)
        current = features[:, 0]

        exact_values, exact_mask = self.exact_map(decoded, phases)
        # All four CFA planes share the same sampling lattice, so one mask channel is enough.
        exact_input = torch.cat((exact_values, exact_mask[:, :, :1].float()), dim=2)
        exact_features = self.exact_encoder(exact_input.reshape(
            batch * frames, 5, HR_PLANE_H, HR_PLANE_W))
        exact_features = exact_features.reshape(batch, frames, self.channels, HR_PLANE_H, HR_PLANE_W)
        exact_current = exact_features[:, 0]

        fused, fused_exact = current, exact_current
        support = exact_mask[:, :, :1].to(decoded.dtype)
        if frames > 1:
            shifted, aligned, low_validity, support = self.align_history(features, exact_features, support, phases)
            fused = self.fuse(current, shifted, low_validity, self.weight)
            fused_exact = self.fuse(exact_current, aligned, support[:, 1:] > 0, self.exact_weight)
        coverage = support.amax(dim=1)
        base = make_bilinear(decoded[:, :1], phases[:, :1])
        base_planes = F.pixel_unshuffle(base.reshape(batch, 1, HEIGHT, WIDTH), 2)
        anchor = base_planes
        if self.anchor == "phase-mosaic":
            if not bool(exact_mask.any(dim=1).all()):
                raise ValueError("phase-mosaic anchor requires a complete set of distinct phases")
            anchor = exact_values.sum(dim=1)

        upsampled = self.upsample(fused)
        head = torch.cat((upsampled, fused_exact, coverage, anchor), dim=1)
        head = self.head(head)
        for block in self.blocks:
            head = head + block(head)
        residual = self.tail(head)
        output = F.pixel_shuffle(anchor + residual, 2).clamp(0, 1)
        if return_support:
            return output, exact_mask.any(dim=1)[:, :1], coverage.gt(0)
        return output

    def align_history(self, features, exact_features, support, phases):
        batch, frames = phases.shape
        current = features[:, 0]
        other = features[:, 1:].flatten(0, 1)
        dx, dy = phase_shift(phases[:, :1], phases[:, 1:])
        dx = dx.reshape(-1, 1, 1).to(current.dtype) * 0.25
        dy = dy.reshape(-1, 1, 1).to(current.dtype) * 0.25
        if self.alignment == "motion":
            current_pairs = current.repeat_interleave(frames - 1, dim=0)
            motion_x, motion_y = self.offset(current_pairs, other)
            motion_x_hr = F.interpolate(motion_x[:, None], scale_factor=4, mode="nearest")[:, 0] * 4
            motion_y_hr = F.interpolate(motion_y[:, None], scale_factor=4, mode="nearest")[:, 0] * 4
            # Exact samples are already on a common spatial grid. Warp only for motion.
            joined = torch.cat((exact_features[:, 1:], support[:, 1:]), dim=2).flatten(0, 1)
            aligned, valid = warp(joined, motion_x_hr, motion_y_hr, return_validity=True)
            exact_history = aligned[:, :-1].unflatten(0, (batch, frames - 1))
            sampled = aligned[:, -1:]
            valid_support = (sampled * (valid & (sampled > 1e-4))).unflatten(0, (batch, frames - 1))
            support = torch.cat((support[:, :1], valid_support), dim=1)
            dx, dy = dx + motion_x, dy + motion_y
        else:
            exact_history = exact_features[:, 1:]
        aligned_low, valid_low = warp(other, dx, dy, return_validity=True)
        return (aligned_low.unflatten(0, (batch, frames - 1)), exact_history,
                valid_low.unflatten(0, (batch, frames - 1)), support)

    @staticmethod
    def fuse(current, history, valid, weight):
        batch, frames = history.shape[:2]
        candidates = torch.cat((current[:, None], history), dim=1)
        reference = current.repeat_interleave(frames + 1, dim=0)
        logits = weight(reference, candidates.flatten(0, 1)).unflatten(0, (batch, frames + 1))
        valid = torch.cat((torch.ones_like(valid[:, :1]), valid), dim=1)
        return (logits.masked_fill(~valid, -torch.inf).softmax(dim=1) * candidates).sum(dim=1)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
