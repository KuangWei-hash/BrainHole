"""v14: patch-split, CFA-aware DCT-residual compact codebook.

Generic, dataset-agnostic math shared by every v14 script (BBB's
train_v14_compact_codebook.py and the demo.mp4 scripts). Each 640x16 RAW8
strip is split into `tiles` tiles of `width` pixels; each tile is represented
by a 16-dim mean/gradient descriptor (the lookup key) plus a 4*RANK-coefficient
low-frequency DCT projection of its residual against a phase-aware bilinear
base (the stored value, int8-quantized by `pack`).

`phase_data` requires the actual per-tile input sample (`low`, shape
[N,30,4,160]) to be supplied explicitly -- it must be the real decoded signal a
receiver would have (the `decoded` SQLite column), not a value derived from
the ground-truth RAW. BBB's historical results were produced by passing
`sample(raw, phase)` (an idealized, noise-free stand-in) as `low`; that
inflates the reported improvement (confirmed on demo.mp4: RAW MAE 49.8%
better than bilinear using the idealized input vs 46.6% using the real
`decoded` column -- see train_v14_demo_compact_codebook_real.py and
實驗紀錄-v14-demo影片驗證與壓縮碼本.md). New work should always pass the real
decoded samples.
"""
import math

import torch
from torch.nn import functional as F

from baseline import upscale
from v11.phase import scatter_indices


DEFAULT_WIDTH = 40
RANK = 8
CAPACITIES = (1, 4)
ROUNDS = 5
FORMAT = "v14-patch-cfa-dct-v1"
GATES = [(0, 0.)] + [(limit, strength) for limit in (4, 8, 12, 16, 24) for strength in (0.5, 1.)]


def architecture(width):
    return FORMAT if width == DEFAULT_WIDTH else f"{FORMAT}-w{width}"


def tiles_for(width):
    return 640 // width


def check_width(width):
    if width % 8 or width < 16 or 640 % width:
        raise ValueError("patch width must divide 640, be a multiple of 8 and at least 16")
    return width


def split_strips(tensor, width, tiles):
    return tensor.reshape(*tensor.shape[:-2], tensor.shape[-2], tiles, width).transpose(-3, -2)


def describe(samples, width, tiles):
    patches = split_strips(samples, width // 4, tiles)
    planes = F.pixel_unshuffle(patches.reshape(-1, 1, 4, width // 4).float(), 2)
    dx = planes.diff(dim=-1)
    dy = planes.diff(dim=-2)
    features = torch.cat((planes.mean(dim=(-2, -1)) - 128, dx.mean(dim=(-2, -1)) * 0.5,
                          dy.mean(dim=(-2, -1)) * 0.5, dx.abs().mean(dim=(-2, -1)) * 0.5), dim=-1)
    return features.reshape(*patches.shape[:-2], 16)


def dct_axis(length):
    freq = torch.arange(length, dtype=torch.float32)[:, None]
    coordinate = torch.arange(length, dtype=torch.float32) + 0.5
    basis = (freq * coordinate * (math.pi / length)).cos() * math.sqrt(2 / length)
    basis[0] /= math.sqrt(2)
    return basis


def dct_basis(width, rank=None):
    """Low-frequency DCT modes for one patch, ordered by (y+x) then (y, x).

    `rank` defaults to the module-level RANK. Callers that work with a
    codebook whose own dct_rank differs must pass it explicitly (or set the
    module RANK first, which is what the training scripts do)."""
    rows, cols = dct_axis(8), dct_axis(width // 2)
    modes = sorted(((y + x, y, x) for y in range(len(rows)) for x in range(len(cols))))[:rank or RANK]
    return torch.stack([(rows[y, :, None] * cols[x, None, :]).flatten() for _, y, x in modes])


def bilinear(samples, phases):
    return upscale(samples[:, None].float(), phases).round().byte()


def phase_data(raw, low, phase, basis, width, tiles):
    """Descriptor keys and DCT residual values for one phase group.

    raw: [N,30,16,640] ground-truth RAW8 (residual target).
    low: [N,30,4,160] the actual per-tile input sample for this phase -- must
        be the real decoded signal, not derived from `raw` (see module docstring).
    """
    flat_low = low.reshape(-1, 4, 160)
    base = bilinear(flat_low, torch.full((len(flat_low),), phase)).reshape_as(raw)
    residual = split_strips(raw.float() - base.float(), width, tiles)
    planes = F.pixel_unshuffle(residual.reshape(-1, 1, 16, width), 2).flatten(-2)
    features = describe(low, width, tiles).reshape(len(raw), 30 * tiles, 16).transpose(0, 1)
    projected = (planes @ basis.T).reshape(len(raw), 30 * tiles, 4 * RANK).transpose(0, 1)
    return features, projected


def fit_round(features, residuals, keys, codes):
    nearest = torch.cdist(features, keys, compute_mode="donot_use_mm_for_euclid_dist").argmin(dim=-1)
    counts = torch.zeros(keys.shape[:2], dtype=torch.int32)
    counts.scatter_add_(1, nearest, torch.ones_like(nearest, dtype=torch.int32))
    return (average_groups(features, nearest, counts[..., None], keys),
            average_groups(residuals, nearest, counts[..., None], codes), counts)


def average_groups(data, groups, counts, fallback):
    index = groups[..., None].expand(-1, -1, data.shape[-1])
    totals = torch.zeros_like(fallback).scatter_add_(1, index, data)
    return torch.where(counts > 0, totals / counts.clamp_min(1), fallback)


def pack(phases, config):
    scale = torch.stack([codes.abs().amax(dim=(0, 1)) for _, codes in phases]).amax(dim=0).clamp_min(1e-6) / 127
    tiles = phases[0][0].shape[0] // 30
    keys = torch.cat([keys for keys, _ in phases]).reshape(16, 30, tiles, -1, 16)
    codes = torch.cat([codes for _, codes in phases]).reshape(16, 30, tiles, -1, 4 * RANK)
    return {"descriptor_keys": keys.round().clamp(-128, 127).char(),
            "residual_codes": (codes / scale).round().clamp(-127, 127).char(),
            "scales": scale, "config": config}


def retrieve(book, samples, positions, phases, basis):
    width = book["config"].get("patch_width", DEFAULT_WIDTH)
    tiles = tiles_for(width)
    query = describe(samples, width, tiles)
    keys = book["descriptor_keys"][phases, positions].float()
    values = book["residual_codes"][phases, positions]
    distance, index = (keys - query[:, :, None]).square().mean(dim=-1).sqrt().min(dim=-1)
    selected = values.gather(2, index[..., None, None].expand(-1, -1, 1, 4 * RANK)).squeeze(2)
    decoded = (selected.float() * book["scales"]).reshape(-1, 4, RANK) @ basis
    patches = F.pixel_shuffle(decoded.reshape(-1, 4, 8, width // 2), 2).reshape(-1, tiles, 16, width)
    return patches, distance


def reconstruct(base, patches, distance, phases, limit, strength):
    if strength == 0:
        return base
    weight = (1 - distance / limit).clamp_min(0) * strength
    delta = (patches * weight[..., None, None]).transpose(1, 2).reshape_as(base)
    for phase in phases.unique().tolist():
        rows, cols = scatter_indices(phase)
        indices = (phases == phase).nonzero().flatten()
        delta[indices[:, None, None], 0, rows[:, None], cols[None, :]] = 0
    return (base + delta).round().clamp(0, 255).byte()


def load_book(path):
    """Shape/dtype sanity check only -- does not enforce a specific `architecture`
    string, so both BBB's "v14-patch-cfa-dct-v1[-wN]" and the demo scripts'
    differently-named codebooks load through the same check.

    The coefficient count is taken from the codebook's own `dct_rank`, not
    from the module-level RANK, so codebooks built with a different rank
    (the demo scripts use 64/160) load without the caller having to mutate
    the global first."""
    book = torch.load(path, map_location="cpu", weights_only=True)
    config = book["config"]
    width = config.get("patch_width", DEFAULT_WIDTH)
    tiles = tiles_for(width)
    rank = config.get("dct_rank", RANK)
    shape = (16, 30, tiles, config["entries_per_tile"])
    if (book["descriptor_keys"].shape != (*shape, 16) or
            book["residual_codes"].shape != (*shape, 4 * rank) or book["scales"].shape != (4 * rank,) or
            book["descriptor_keys"].dtype != torch.int8 or book["residual_codes"].dtype != torch.int8):
        raise ValueError("invalid compact v14 codebook")
    return book
