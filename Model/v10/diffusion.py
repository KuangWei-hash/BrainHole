"""Cosine residual diffusion and stateless, batch-independent DDIM sampling."""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


class Schedule(nn.Module):
    def __init__(self, timesteps=1000):
        super().__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be >=2")
        self.timesteps = timesteps
        x = torch.linspace(0, timesteps, timesteps+1, dtype=torch.float64)
        cumulative = torch.cos(((x/timesteps+0.008)/1.008)*math.pi/2).square()
        cumulative = cumulative/cumulative[0]
        betas = (1-cumulative[1:]/cumulative[:-1]).clamp(1e-8, 0.999)
        self.register_buffer("alpha_bar", (1-betas).cumprod(0).float())

    def alpha(self, t):
        return self.alpha_bar[t].view(-1, 1, 1, 1)

    def add_noise(self, clean, t, noise):
        a = self.alpha(t)
        return a.sqrt()*clean+(1-a).sqrt()*noise

    def clean_from_noise(self, noisy, t, noise):
        a = self.alpha(t)
        return (noisy-(1-a).sqrt()*noise)/a.sqrt()

    def training_target(self, clean, t, noise, prediction_type="epsilon"):
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "v":
            a = self.alpha(t)
            return a.sqrt()*noise-(1-a).sqrt()*clean
        raise ValueError("unknown prediction type")

    def decode_prediction(self, noisy, t, prediction, prediction_type="epsilon"):
        a = self.alpha(t)
        if prediction_type == "epsilon":
            return self.clean_from_noise(noisy, t, prediction), prediction
        if prediction_type == "v":
            return a.sqrt()*noisy-(1-a).sqrt()*prediction, (1-a).sqrt()*noisy+a.sqrt()*prediction
        raise ValueError("unknown prediction type")


def keyed_noise(keys, seed, video_id, device, policy="per_frame"):
    if policy not in ("per_frame", "fixed_screen"):
        raise ValueError("unknown noise policy")
    samples = []
    for frame, block in keys:
        frame_key = int(frame) if policy == "per_frame" else "screen"
        label = f"{video_id}:{seed}:{frame_key}:{int(block)}".encode()
        derived = int.from_bytes(hashlib.sha256(label).digest()[:8], "little") % (2**63-1)
        rng = torch.Generator(device="cpu").manual_seed(derived)
        samples.append(torch.randn((4, 8, 320), generator=rng))
    return torch.stack(samples).to(device)


def compose(bilinear, residual, scale, gamma=1., *, base=None):
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    base = bilinear[:, 0] if base is None else base
    if gamma == 0:
        return base.clone()
    planes = F.pixel_unshuffle(base, 2)
    return F.pixel_shuffle(planes+gamma*scale*residual, 2).clamp(0, 1)


@torch.inference_mode()
def generate_residual(model, schedule, decoded, bilinear, phases, keys, *, seed=0,
                      video_id="", steps=32, policy="per_frame", mode="diffusion", initial_noise=None, base=None):
    if model.training:
        raise ValueError("sampling requires eval()")
    if steps < 1 or steps > schedule.timesteps:
        raise ValueError("sampling steps out of range")
    condition = model.prepare_condition(decoded, bilinear, phases)
    noisy = keyed_noise(keys, seed, video_id, decoded.device, policy) if initial_noise is None else initial_noise.clone()
    if noisy.shape != (len(decoded), 4, 8, 320) or not torch.isfinite(noisy).all():
        raise ValueError("invalid initial noise")
    if mode in ("deterministic", "single_noise"):
        times = torch.full((len(decoded),), 0 if mode == "deterministic" else schedule.timesteps-1,
                           device=decoded.device, dtype=torch.long)
        return model(torch.zeros_like(noisy) if mode == "deterministic" else noisy, times, condition)
    if mode != "diffusion":
        raise ValueError("unknown generation mode")
    grid = torch.linspace(schedule.timesteps-1, 0, steps, device=decoded.device).round().long()
    base = F.pixel_unshuffle(bilinear[:, 0] if base is None else base, 2)
    lower, upper = -base/model.residual_scale, (1-base)/model.residual_scale
    for i, t in enumerate(grid):
        times = t.expand(len(decoded))
        prediction = model(noisy, times, condition)
        a = schedule.alpha_bar[t]
        clean, epsilon = schedule.decode_prediction(noisy, times, prediction, getattr(model, "prediction_type", "epsilon"))
        clean = clean.clamp(lower, upper)
        if i == len(grid)-1:
            return clean
        # Recompute epsilon after clipping x0 to physical RAW [0,1] bounds.
        epsilon = (noisy-a.sqrt()*clean)/(1-a).sqrt()
        next_a = schedule.alpha_bar[grid[i+1]]
        noisy = next_a.sqrt()*clean+(1-next_a).sqrt()*epsilon


def low_pass(planes):
    axis = torch.arange(-2, 3, device=planes.device, dtype=planes.dtype)
    kernel = torch.exp(-axis.square()/2)
    kernel = kernel/kernel.sum()
    kernel = (kernel[:, None]*kernel[None, :]).expand(4, 1, 5, 5)
    return F.conv2d(F.pad(planes, (2, 2, 2, 2), mode="replicate"), kernel, groups=4)


def high_pass(planes):
    return planes-low_pass(planes)


@torch.no_grad()
def anchor_image(anchor, decoded, bilinear, phases):
    if anchor is None:
        return bilinear[:, 0]
    condition = anchor.prepare_condition(decoded, bilinear, phases)
    zeros = decoded.new_zeros((len(decoded), 4, 8, 320))
    times = torch.zeros(len(decoded), dtype=torch.long, device=decoded.device)
    return compose(bilinear, anchor(zeros, times, condition), anchor.residual_scale)


def target_residual(target, base, *, detail_only=False):
    planes = F.pixel_unshuffle(target-base, 2)
    return high_pass(planes) if detail_only else planes
