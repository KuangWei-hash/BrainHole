"""v15: a simple 50/50 ensemble of v12 and v14's finished frames.

Motivation (visual inspection, 2026-11-23): v12's 16-frame real-pixel temporal
blend converges to genuinely accurate texture in truly static regions (it
accumulates real measured pixels, not a learned/compressed approximation),
but smears moving objects due to blending across time without motion
compensation. v14 (compact codebook + current-phase protection + 30/70
recursive blend with the previous output) keeps shapes intact under motion
and scores a lower RAW MAE, but its DCT-residual-compressed texture is softer
than v12's in static regions -- confirmed by eye, not just by the metric (see
實驗紀錄-v14-demo影片驗證與壓縮碼本.md). v12 and v14 are computed completely
independently; this module only averages their two finished 640x16 frames.
"""
import torch


def blend(a, b, weight_a=0.5):
    """Average two same-shape uint8 RAW8 tensors. weight_a is a's share."""
    return (a.float() * weight_a + b.float() * (1 - weight_a)).round().clamp(0, 255).byte()
