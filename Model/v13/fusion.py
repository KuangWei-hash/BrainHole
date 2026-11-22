"""Repack a complete phase mosaic into newest-first same-color frame features."""
import torch
from torch import nn
from torch.nn import functional as F
from v13.native import NativeRestorer


GUIDED_VARIANT = "R2-frames-guide"
CLEAN_VARIANT = "R2-frames-clean"
CONSISTENCY_VARIANT = "R2-frames-consistency"
EXACT_VARIANT = "R2-frames-exact"
TREND_VARIANT = "R2-frames-trend"
DEAGE_VARIANT = "R2-frames-deage"
NATIVE_VARIANTS = (CLEAN_VARIANT, EXACT_VARIANT, TREND_VARIANT, DEAGE_VARIANT)
FUSION_VARIANTS = ("R2-frames-mean", "R2-frames", "R2-frames-phase", "R2-frames-align",
                   GUIDED_VARIANT, CONSISTENCY_VARIANT, *NATIVE_VARIANTS)
PHASE_VARIANTS = ("R2-frames-phase", "R2-frames-align", GUIDED_VARIANT, CONSISTENCY_VARIANT, *NATIVE_VARIANTS)
GUIDE_TEMPERATURE = 0.08
GUIDE_WEIGHT = 0.02


def nearest_indices():
    """Map each destination CFA cell to a measured cell of each phase."""
    phase = torch.arange(16)[:, None, None]
    py, px = phase // 4, phase % 4
    y, x = torch.arange(8)[None, :, None], torch.arange(320)[None, None, :]
    source_y = ((y - py + 2) // 4).clamp(0, 1) * 4 + py
    source_x = ((x - px + 2) // 4).clamp(0, 79) * 4 + px
    return (source_y * 320 + source_x).flatten(1)


def phases_in_age_order(age):
    return age[:, 0, :4, :4].reshape(len(age), 16).argsort(dim=1)


def lift_frames(measured, phase_by_age, indices):
    n, channels, height, width = measured.shape
    by_age = indices[phase_by_age].unsqueeze(2).expand(-1, -1, channels, -1)
    flat = measured.flatten(2).unsqueeze(1).expand(-1, 16, -1, -1)
    return flat.gather(3, by_age).reshape(n, 16, channels, height, width)


def newest_first(measured, age, indices):
    """Recover sixteen phase planes, lifting each with v12's nearest-cell rule."""
    return lift_frames(measured, phases_in_age_order(age), indices)


def consistency_logits(frames, age):
    """Lower the weight of old frames that disagree with the current local color."""
    coarse = F.avg_pool2d(frames.flatten(0, 1), 5, stride=1, padding=2, count_include_pad=False)
    coarse = coarse.unflatten(0, frames.shape[:2])
    mismatch = (coarse - coarse[:, :1]).abs().mean(dim=2, keepdim=True)
    return -mismatch * (1 + age) / 0.05


def temporal_trend(frames, age):
    """Measure the low-frequency fitted change from mid-window to the current frame."""
    coarse = F.avg_pool2d(frames.flatten(0, 1), (5, 9), stride=1, padding=(2, 4), count_include_pad=False)
    coarse = coarse.unflatten(0, frames.shape[:2])
    centered = 0.5 - age
    trend = 0.5 * (coarse * centered).sum(dim=1) / centered.square().sum(dim=1)
    return trend.clamp(-0.25, 0.25)


def extrapolate_low_frequency(frames, age, trend):
    change = (2 * age * trend.unsqueeze(1)).clamp(-0.25, 0.25)
    return (frames + change).clamp(0, 1)


class LocalAlignment(nn.Module):
    def __init__(self):
        super().__init__()
        y, x = torch.meshgrid(torch.arange(8), torch.arange(320), indexing="ij")
        self.register_buffer("grid", torch.stack(((2 * x + 1) / 320 - 1, (2 * y + 1) / 8 - 1), dim=-1),
                             persistent=False)
        self.register_buffer("scale", torch.tensor([4 / 320, 4 / 8]), persistent=False)
        self.estimate = nn.Sequential(nn.Conv2d(15, 12, 3, padding=1, padding_mode="replicate"),
                                      nn.SiLU(), nn.Conv2d(12, 2, 3, padding=1, padding_mode="replicate"))
        nn.init.zeros_(self.estimate[-1].weight)
        nn.init.zeros_(self.estimate[-1].bias)

    def forward(self, frames, foot, age):
        current = frames[:, :1].expand_as(frames)
        current_foot = foot[:, :1].expand_as(foot)
        features = torch.cat((current, frames, frames - current, foot, current_foot, age), dim=2)
        offset = self.estimate(features.flatten(0, 1).contiguous(memory_format=torch.channels_last))
        shift = offset.permute(0, 2, 3, 1).tanh() * self.scale
        # Keep age zero at its exact coordinates.
        shift = shift * (age.flatten(0, 1).permute(0, 2, 3, 1) > 0)
        combined = torch.cat((frames, foot), dim=2).flatten(0, 1)
        aligned = F.grid_sample(combined, self.grid + shift, padding_mode="border", align_corners=False)
        aligned = aligned.unflatten(0, frames.shape[:2])
        return aligned[:, :, :4], aligned[:, :, 4:]


class FrameFusion(nn.Module):
    def __init__(self, variant, width):
        super().__init__()
        self.register_buffer("indices", nearest_indices(), persistent=False)
        self.register_buffer("ages", torch.arange(16) / 15, persistent=False)
        self.use_phase = variant in PHASE_VARIANTS
        self.use_consistency = variant == CONSISTENCY_VARIANT
        positions = torch.arange(8 * 320)
        self.register_buffer("footprint", self.indices == positions, persistent=False)
        self.encoder = nn.Conv2d(14 if self.use_phase else 13, 8, 3, padding=1, padding_mode="replicate")
        self.projection = nn.Conv2d(8, width, 1)
        self.score = nn.Conv2d(8, 1, 1) if variant not in ("R2-frames-mean", CONSISTENCY_VARIANT) else None
        if self.score is not None:
            nn.init.zeros_(self.score.weight)
            nn.init.zeros_(self.score.bias)
        self.align = LocalAlignment() if variant == "R2-frames-align" else None
        self.native = NativeRestorer() if variant in NATIVE_VARIANTS else None
        self.use_trend = variant in (TREND_VARIANT, DEAGE_VARIANT)
        self.deage = variant == DEAGE_VARIANT
        self.direct = (nn.Conv2d(9 if self.use_trend else 5, width, 3, padding=1, padding_mode="replicate")
                       if variant in (EXACT_VARIANT, TREND_VARIANT, DEAGE_VARIANT) else None)
        if self.native is not None:
            self.native.requires_grad_(False)
            self.register_buffer("samples", self.footprint.nonzero()[:, 1].reshape(16, 160), persistent=False)
            native_indices = ((self.indices // 320) // 4) * 80 + ((self.indices % 320) // 4)
            self.register_buffer("native_indices", native_indices, persistent=False)

    def clean_frames(self, measured, phases):
        n, channels, height, width = measured.shape
        samples = self.samples[phases].unsqueeze(2).expand(-1, -1, channels, -1)
        raw = measured.flatten(2).unsqueeze(1).expand(-1, 16, -1, -1).gather(3, samples)
        decoded = F.pixel_shuffle(raw.reshape(-1, channels, 2, 80) * 255, 2)
        corrected = F.pixel_unshuffle(self.native(decoded).clamp(0, 1), 2).reshape(n, 16, channels, -1)
        upsample = self.native_indices[phases].unsqueeze(2).expand(-1, -1, channels, -1)
        return corrected.gather(3, upsample).reshape(n, 16, channels, height, width)

    def forward(self, base, measured, age, target=None):
        phases = phases_in_age_order(age)
        frames = (lift_frames(measured, phases, self.indices) if self.native is None
                  else self.clean_frames(measured, phases))
        reference = base.unsqueeze(1).expand_as(frames)
        n, history, _, height, width = frames.shape
        age_planes = self.ages.view(1, 16, 1, 1, 1).expand(n, -1, -1, height, width)
        if self.use_trend:
            trend = temporal_trend(frames, age_planes)
            if self.deage:
                frames = extrapolate_low_frequency(frames, age_planes, trend)
        if self.use_phase:
            foot = self.footprint[phases].reshape(n, history, 1, height, width).to(frames.dtype)
            if self.align is not None:
                frames, foot = self.align(frames, foot, age_planes)
            inputs = torch.cat((reference, frames, frames - reference, age_planes, foot), dim=2)
        else:
            inputs = torch.cat((reference, frames, frames - reference, age_planes), dim=2)
        encoded = F.silu(self.encoder(inputs.flatten(0, 1).contiguous(memory_format=torch.channels_last)))
        by_age = encoded.reshape(n, history, -1, height, width)
        if self.use_consistency:
            logits = consistency_logits(frames, age_planes)
            merged = (by_age * logits.softmax(dim=1)).sum(dim=1)
        elif self.score is None:
            merged = by_age.mean(dim=1)
        else:
            logits = self.score(encoded).reshape(n, history, 1, height, width)
            merged = (by_age * logits.softmax(dim=1)).sum(dim=1)
        fused = self.projection(merged)
        if self.direct is not None:
            sampled = (frames * foot).sum(dim=1)
            sampled_age = (age_planes * foot).sum(dim=1)
            evidence = [sampled - base, sampled_age]
            if self.use_trend:
                evidence.append(trend)
            fused = fused + self.direct(torch.cat(evidence, dim=1))
        if target is None:
            return fused
        with torch.no_grad():
            error = (frames - F.pixel_unshuffle(target, 2).unsqueeze(1)).abs().mean(dim=2)
            local_error = F.avg_pool2d(F.pad(error, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
            target_weights = (-local_error.unsqueeze(2) / GUIDE_TEMPERATURE).softmax(dim=1)
        divergence = F.kl_div(logits.log_softmax(dim=1), target_weights, reduction="none")
        return fused, divergence.sum(dim=1).mean()
