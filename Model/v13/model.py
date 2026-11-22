"""Single-pass RAW8 residual CNN."""
import torch
from torch import nn
from torch.nn import functional as F
from v13.fusion import FUSION_VARIANTS, FrameFusion
from v13.native import conv


class ResidualCNN(nn.Module):
    def __init__(self, variant, width=32, context="compact"):
        super().__init__()
        self.variant = variant
        self.stem = conv(4 if variant == "R0" else 10, width)
        spacings = (1, 2, 3, 1) if context == "extended" else (1, 1, 1, 1)
        self.blocks = nn.ModuleList(nn.Sequential(conv(width, width, step), nn.SiLU(), conv(width, width, step))
                                    for step in spacings)
        self.head = conv(width, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.fusion = FrameFusion(variant, width) if variant in FUSION_VARIANTS else None

    def forward(self, features, alpha=1, fusion_target=None):
        base = features[:, :1].float() / 255
        packed = F.pixel_unshuffle(base, 2)
        inputs = packed
        if self.variant != "R0":
            measured = F.pixel_unshuffle(features[:, 1:2].float() / 255, 2)
            age = features[:, 2:3, ::2, ::2]
            inputs = torch.cat((packed, measured, age.float() / 15,
                                features[:, 3:4, ::2, ::2].float()), dim=1)
        result = self.stem(inputs)
        if self.fusion is not None:
            fused = self.fusion(packed, measured, age, fusion_target)
            if fusion_target is not None:
                fused, guidance = fused
            result = result + fused
        result = F.silu(result)
        for block in self.blocks:
            result = result + 0.1 * block(result)
        raw = base + alpha * F.pixel_shuffle(self.head(result), 2)
        return raw if fusion_target is None else (raw, guidance)

    def predict(self, features, alpha=1):
        return quantize(self(features, alpha)) if alpha else features[:, :1]


def quantize(raw):
    return (raw.clamp(0, 1) * 255).round().to(torch.uint8)


def mix_residual_bands(base, full, low_alpha, high_alpha):
    delta = full - base
    if low_alpha == high_alpha:
        return base + low_alpha * delta
    packed = F.pixel_unshuffle(delta, 2)
    low = F.avg_pool2d(F.pad(packed, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
    return base + F.pixel_shuffle(low_alpha * low + high_alpha * (packed - low), 2)


def loss_for(variant, prediction, target, gradient_weight=0.1, band_weight=0.):
    error = prediction - target
    loss = error.abs().mean()
    if variant == "R2" or variant in FUSION_VARIANTS:
        packed = F.pixel_unshuffle(error, 2)
        loss = loss + gradient_weight * (packed.diff(dim=-1).abs().mean() + packed.diff(dim=-2).abs().mean())
        if band_weight:
            low = F.avg_pool2d(F.pad(packed, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
            loss = loss + band_weight * (packed - low).abs().mean()
    return loss
