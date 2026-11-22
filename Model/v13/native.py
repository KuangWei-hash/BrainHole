"""Predict clean sample values at the native decoded CFA coordinates."""
from torch import nn
from torch.nn import functional as F

def conv(inputs, outputs, horizontal_dilation=1):
    spacing = (1, horizontal_dilation)
    return nn.Conv2d(inputs, outputs, 3, padding=spacing, dilation=spacing, padding_mode="replicate")


class NativeRestorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = conv(4, 24)
        self.blocks = nn.ModuleList(nn.Sequential(conv(24, 24), nn.SiLU(), conv(24, 24)) for _ in range(3))
        self.head = conv(24, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, decoded):
        native = F.pixel_unshuffle(decoded.float() / 255, 2)
        result = F.silu(self.stem(native))
        for block in self.blocks:
            result = result + 0.1 * block(result)
        return F.pixel_shuffle(native + self.head(result), 2)
