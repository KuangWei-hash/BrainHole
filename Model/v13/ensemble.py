"""Learn when to trust either of two fixed residual reconstructions."""
import torch
from torch import nn
from torch.nn import functional as F

from v13.native import conv


class EnsembleGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(conv(13, 16), nn.SiLU(), conv(16, 16), nn.SiLU())
        self.weight = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.weight.weight)
        nn.init.zeros_(self.weight.bias)

    def forward(self, x, trend, spatial):
        base = F.pixel_unshuffle(x[:, :1].float() / 255, 2)
        time = F.pixel_unshuffle(trend, 2)
        space = F.pixel_unshuffle(spatial, 2)
        age = x[:, 2:3, ::2, ::2].float() / 15
        features = torch.cat((base, time - base, space - base, age), dim=1)
        mix = self.weight(self.features(features)).sigmoid()
        result = torch.lerp(time, space, mix)
        return F.pixel_shuffle(result, 2)
