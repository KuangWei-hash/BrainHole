"""Phase-exact reassembly of decoded strips into the 640x16 high-resolution mosaic.

The sensor mosaic is sampled with a rotating phase: ``Phase(frame) = frame % 16 + 1``.
Every frame therefore observes a different 2x2 sub-block inside every 8x8 RGGB tile,
and the 16 phases tile each tile exactly once. This module places the decoded
samples of each frame at their exact high-resolution positions without any
interpolation, and reports which pixels were actually measured.

No learnable parameters and no interpolation: the placement is a pure function of
the phase, so it can be verified value by value against the decoded input.
"""
import torch

WIDTH = 640
HEIGHT = 16
LR_WIDTH = 160
LR_HEIGHT = 4
TILE = 8
PHASES = 16


def phase_offsets(phases):
    """Mosaic-unit sampling offsets of 0-based phases: px, py in {0,2,4,6}.

    Mirrors ``Downsample`` in training-data-preparation/internal/dataset/image.go:
    ``px, py = ((phase-1)%4)*2, ((phase-1)/4)*2``.
    """
    if phases.dtype != torch.long:
        raise ValueError("phases must be int64")
    px = (phases % 4) * 2
    py = (phases // 4) * 2
    return px, py


def scatter_indices(phase, device=None):
    """Rows and columns of the high-resolution mosaic receiving each decoded pixel.

    Returns ``(rows, cols)`` with shapes ``[4]`` and ``[160]`` for a single phase.
    """
    px, py = phase_offsets(torch.as_tensor(phase, dtype=torch.long))
    px, py = int(px), int(py)
    device = device if device is not None else torch.device("cpu")
    rows = torch.arange(LR_HEIGHT, device=device)
    cols = torch.arange(LR_WIDTH, device=device)
    hr_rows = TILE * (rows // 2) + py + (rows % 2)
    hr_cols = TILE * (cols // 2) + px + (cols % 2)
    return hr_rows, hr_cols


def reassemble(decoded, phases):
    """Place every frame's decoded samples into the high-resolution mosaic.

    Args:
        decoded: ``[B, T, 1, 4, 160]`` float tensor of decoded strips (normalized).
        phases: ``[B, T]`` int64 tensor of 0-based phases in ``[0, 15]``.

    Returns:
        values: ``[B, T, 1, 16, 640]`` float tensor; unmeasured pixels are zero.
        mask:   ``[B, T, 1, 16, 640]`` bool tensor; True where the pixel was measured.
    """
    if decoded.dim() != 5 or decoded.shape[2] != 1 or decoded.shape[3] != LR_HEIGHT or decoded.shape[4] != LR_WIDTH:
        raise ValueError("expected decoded [B,T,1,4,160]")
    batch, frames = decoded.shape[0], decoded.shape[1]
    if phases.shape != (batch, frames) or phases.dtype != torch.long:
        raise ValueError("expected phases [B,T] as int64")
    if phases.min() < 0 or phases.max() > PHASES - 1:
        raise ValueError("phase must be within 0..15")
    values = decoded.new_zeros(batch, frames, 1, HEIGHT, WIDTH)
    mask = torch.zeros(batch, frames, 1, HEIGHT, WIDTH, dtype=torch.bool, device=decoded.device)
    for frame in range(frames):
        for b in range(batch):
            rows, cols = scatter_indices(int(phases[b, frame]), decoded.device)
            flat = (rows[:, None] * WIDTH + cols[None, :]).reshape(-1)
            values[b, frame, 0].reshape(-1)[flat] = decoded[b, frame, 0].reshape(-1)
            mask[b, frame, 0].reshape(-1)[flat] = True
    return values, mask


def coverage(mask):
    """Fraction of measured pixels per frame: ``[B, T]`` float tensor.

    A window of ``T`` frames with pairwise distinct phases covers exactly ``T/16``.
    """
    if mask.dim() != 5:
        raise ValueError("expected mask [B,T,1,16,640]")
    return mask.float().mean(dim=(2, 3, 4))


def exact_measurements(values, mask):
    """Collapse the frame axis into one measurement plus a confidence count.

    Returns:
        fused: ``[B, 1, 1, 16, 640]`` mean of the measured values (zero if unmeasured).
        count: ``[B, 1, 1, 16, 640]`` int tensor of how many frames measured each pixel.
    """
    if values.shape != mask.shape:
        raise ValueError("values and mask must match")
    counted = mask.sum(1, keepdim=True)
    total = (values * mask).sum(1, keepdim=True)
    fused = total / counted.clamp_min(1)
    return fused, counted
