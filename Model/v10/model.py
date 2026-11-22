"""Two-frame, dual-input CFA residual denoiser. No V9 dependency."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from baseline import upscale


def make_bilinear(decoded, phases):
    batch = decoded.shape[0]
    return upscale(decoded.reshape(-1, 1, 4, 160), phases.reshape(-1)).reshape(batch, 2, 1, 16, 640)


def align_features(features, phases):
    """Same CFA-plane coordinates as baseline.upscale, including border handling."""
    x = torch.arange(320, device=features.device, dtype=features.dtype)[None, None, :]
    y = torch.arange(8, device=features.device, dtype=features.dtype)[None, :, None]
    px = (phases % 4).to(features.dtype)[:, None, None]
    py = (phases // 4).to(features.dtype)[:, None, None]
    gx = ((x-px)/4/79*2-1).expand(-1, 8, -1)
    gy = ((y-py)/4*2-1).expand(-1, -1, 320)
    return F.grid_sample(features, torch.stack((gx, gy), -1), align_corners=True, padding_mode="border")


def validate_inputs(decoded, bilinear, phases, check_pair=False):
    batch = len(decoded)
    if (decoded.shape != (batch, 2, 1, 4, 160) or bilinear.shape != (batch, 2, 1, 16, 640)
            or phases.shape != (batch, 2)):
        raise ValueError("expected decoded [B,2,1,4,160], bilinear [B,2,1,16,640], phases [B,2]")
    if not decoded.is_floating_point() or bilinear.dtype != decoded.dtype or phases.dtype != torch.long:
        raise ValueError("normalized floating images and int64 phases required")
    if (decoded.device != bilinear.device or decoded.device != phases.device or
            not torch.isfinite(decoded).all() or not torch.isfinite(bilinear).all() or
            decoded.min() < 0 or decoded.max() > 1 or bilinear.min() < 0 or bilinear.max() > 1 or
            phases.min() < 0 or phases.max() > 15):
        raise ValueError("invalid range, phase, device or nonfinite input")
    if check_pair and not torch.allclose(bilinear, make_bilinear(decoded, phases), atol=1e-6, rtol=1e-5):
        raise ValueError("bilinear does not match decoded and phase")


class TimeBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=(1, dilation),
                               dilation=(1, dilation), padding_mode="replicate")
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=(1, dilation),
                               dilation=(1, dilation), padding_mode="replicate")
        self.film = nn.Linear(128, channels*2)

    def forward(self, x, embedding):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(embedding).chunk(2, dim=1)
        h = self.norm2(h)*(1+scale[:, :, None, None])+shift[:, :, None, None]
        return (x+self.conv2(F.silu(h)))/math.sqrt(2)


class ResidualDenoiser(nn.Module):
    def __init__(self, channels=48, condition="both", prediction_type="epsilon"):
        super().__init__()
        if channels < 8 or channels % 8 or condition not in ("both", "decoded", "bilinear"):
            raise ValueError("channels must be divisible by 8; unknown condition")
        self.channels, self.condition_mode = channels, condition
        if prediction_type not in ("epsilon", "v"):
            raise ValueError("unknown prediction type")
        self.prediction_type = prediction_type
        self.low_encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1, padding_mode="replicate"), nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, padding_mode="replicate"), nn.SiLU())
        self.phase = nn.Embedding(16, 8)
        self.time_mlp = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.stem = nn.Conv2d(60, channels, 3, padding=1, padding_mode="replicate")
        self.blocks = nn.ModuleList([TimeBlock(channels, d) for d in (1, 2, 4, 8)])
        self.tail = nn.Conv2d(channels, 4, 3, padding=1, padding_mode="replicate")
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)
        self.register_buffer("residual_scale", torch.ones(1, 4, 1, 1))

    def prepare_condition(self, decoded, bilinear, phases):
        validate_inputs(decoded, bilinear, phases)
        batch = len(decoded)
        low = F.pixel_unshuffle(decoded.reshape(-1, 1, 4, 160), 2)
        low = align_features(self.low_encoder(low), phases.reshape(-1)).reshape(batch, 32, 8, 320)
        high = F.pixel_unshuffle(bilinear.reshape(-1, 1, 16, 640), 2).reshape(batch, 8, 8, 320)
        if self.condition_mode == "bilinear":
            low = low*0
        if self.condition_mode == "decoded":
            high = high*0
        phase = self.phase(phases).reshape(batch, 16, 1, 1).expand(-1, -1, 8, 320)
        return torch.cat((low, high, phase), dim=1)

    def forward(self, noisy_residual, timesteps, condition):
        if noisy_residual.shape != (len(condition), 4, 8, 320) or timesteps.shape != (len(condition),):
            raise ValueError("invalid residual or time shape")
        frequencies = torch.exp(-math.log(10000)*torch.arange(64, device=condition.device)/64)
        angles = timesteps.float()[:, None]*frequencies[None]
        embedding = self.time_mlp(torch.cat((angles.cos(), angles.sin()), dim=1))
        x = self.stem(torch.cat((noisy_residual, condition), dim=1))
        for block in self.blocks:
            x = block(x, embedding)
        return self.tail(x)


def build_v10(config):
    if config.get("architecture") != "v10" or config.get("history") != 2:
        raise ValueError("requires v10 history=2 checkpoint")
    return ResidualDenoiser(config.get("channels", 48), config.get("condition", "both"),
                            config.get("prediction_type", "epsilon"))


def load_v10(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = build_v10(saved["config"]).to(device)
    model.load_state_dict(saved["model"])
    return model.eval(), saved


def load_detail_anchor(saved, device):
    state = saved.get("detail_anchor")
    if state is None:
        return None
    anchor = build_v10(state["config"]).to(device)
    anchor.load_state_dict(state["model"])
    return anchor.eval().requires_grad_(False)
