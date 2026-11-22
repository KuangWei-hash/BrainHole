"""Learn an encoded-byte representation, then generate a RAW Bayer block."""
import torch
from torch import nn
from torch.nn import functional as F


class ByteRawModel(nn.Module):
    def __init__(self, history=1, decay=0.2):
        super().__init__()
        if history not in (1, 2, 16) or decay <= 0:
            raise ValueError("history must be 1, 2 or 16; decay must be positive")
        self.history = history
        self.embedding = nn.Embedding(257, 16, padding_idx=256)
        self.encoder = nn.Sequential(
            nn.Conv1d(16, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.ReLU(),
        )
        self.phase = nn.Embedding(16, 8)
        self.project = nn.Sequential(nn.Linear(32 * 40 + 8 + 1, 32 * 2 * 80), nn.ReLU())
        self.reconstruct = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 4, 3, padding=1), nn.Sigmoid(),
        )
        weights = torch.exp(-decay * torch.arange(history, dtype=torch.float32))
        self.register_buffer("time_weights", weights / weights.sum())

    def forward(self, tokens, phases, lengths):
        batch, history, width = tokens.shape
        if history != self.history or width != 640:
            raise ValueError("unexpected history or byte width")
        features = self.embedding(tokens.reshape(-1, width)).transpose(1, 2)
        features = F.adaptive_avg_pool1d(self.encoder(features), 40).flatten(1)
        features = torch.cat((features, self.phase(phases.reshape(-1)),
                              lengths.reshape(-1, 1).float() / 638), dim=1)
        spatial = self.project(features).reshape(batch, history, 32, 2, 80)
        fused = (spatial * self.time_weights.view(1, history, 1, 1, 1)).sum(1)
        # Four planes at H=8, W=320; pixel shuffle interleaves R,G1,G2,B.
        return F.pixel_shuffle(self.reconstruct(fused), 2)


class ByteRawModelV2(nn.Module):
    """Encoded-only inference; decoded low RAW is auxiliary supervision in training."""
    def __init__(self, history=1, decay=0.2, residual_scale=0.1):
        super().__init__()
        if history not in (1, 2, 16) or decay <= 0:
            raise ValueError("history must be 1, 2 or 16; decay must be positive")
        self.history = history
        if not 0 < residual_scale <= 1:
            raise ValueError("residual_scale must be in (0,1]")
        self.residual_scale = residual_scale
        self.embedding = nn.Embedding(257, 16, padding_idx=256)
        self.encoder = nn.Sequential(
            nn.Conv1d(16, 32, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(32, 16, 5, stride=2, padding=2), nn.GELU(),
        )
        self.low_head = nn.Sequential(nn.Flatten(), nn.Linear(16*160, 640), nn.Sigmoid())
        self.phase = nn.Embedding(16, 8)
        self.refine = nn.Sequential(
            nn.Conv2d(4+4+8, 24, 3, padding=1), nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(),
            nn.Conv2d(24, 4, 3, padding=1), nn.Tanh(),
        )
        nn.init.zeros_(self.refine[-2].weight)
        nn.init.zeros_(self.refine[-2].bias)
        weights = torch.exp(-decay*torch.arange(history, dtype=torch.float32))
        self.register_buffer("time_weights", weights/weights.sum())

    def forward(self, tokens, phases, lengths, return_low=False):
        from baseline import upscale
        batch, history, width = tokens.shape
        if history != self.history or width != 640:
            raise ValueError("unexpected history or byte width")
        encoded = self.embedding(tokens.reshape(-1, 640)).transpose(1, 2)
        low = self.low_head(self.encoder(encoded)).reshape(batch*history, 1, 4, 160)
        high = upscale(low, phases.reshape(-1)).reshape(batch, history, 1, 16, 640)
        current = high[:, 0]
        fused = (high*self.time_weights.view(1, history, 1, 1, 1)).sum(1)
        phase = self.phase(phases[:, 0])[:, :, None, None].expand(-1, -1, 8, 320)
        features = torch.cat((F.pixel_unshuffle(current, 2), F.pixel_unshuffle(fused, 2), phase), 1)
        residual = F.pixel_shuffle(self.refine(features), 2)*self.residual_scale
        prediction = (current+residual).clamp(0, 1)
        if return_low:
            return prediction, low.reshape(batch, history, 1, 4, 160)
        return prediction


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions; optional hidden expansion, identity output width."""
    def __init__(self, channels=24, expansion=1, layers=2):
        super().__init__()
        if expansion not in (1, 2, 4):
            raise ValueError("supported residual expansion factors: 1, 2 or 4")
        hidden = channels*expansion
        if layers not in (2, 4) or (layers == 4 and expansion != 2):
            raise ValueError("four-layer blocks require expansion=2")
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )
        if layers == 4:
            self.body = nn.Sequential(
                nn.Conv2d(channels, hidden, 3, padding=1), nn.ReLU(),
                nn.Conv2d(hidden, hidden*2, 3, padding=1), nn.ReLU(),
                nn.Conv2d(hidden*2, hidden, 3, padding=1), nn.ReLU(),
                nn.Conv2d(hidden, channels, 3, padding=1),
            )

    def forward(self, features):
        return features+self.body(features)


class HybridRawModel(nn.Module):
    """v3-v8: real decoded RAW is an input at training and inference."""
    requires_decoded = True
    def __init__(self, history=1, decay=0.2, residual_scale=0.5, residual_blocks=0, residual_expansion=1, residual_layers=2):
        super().__init__()
        if history not in (1, 2, 16) or decay <= 0 or not 0 < residual_scale <= 1:
            raise ValueError("invalid history, decay or residual scale")
        self.history, self.residual_scale = history, residual_scale
        self.embedding = nn.Embedding(257, 8, padding_idx=256)
        self.encoder = nn.Sequential(
            nn.Conv1d(8, 16, 7, stride=4, padding=3), nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(128, 8),
        )
        self.phase = nn.Embedding(16, 4)
        if residual_blocks not in (0, 4, 5, 6):
            raise ValueError("supported backbones: original v3, four, five or six residual blocks")
        self.residual_blocks = residual_blocks
        if residual_expansion not in (1, 2, 4):
            raise ValueError("invalid residual expansion")
        self.residual_expansion = residual_expansion
        self.residual_layers = residual_layers
        stem = [nn.Conv2d(4+4+8+4+1, 24, 3, padding=1), nn.ReLU()]
        body = ([ResidualBlock(24, residual_expansion, residual_layers) for _ in range(residual_blocks)] if residual_blocks else
                [nn.Conv2d(24, 24, 3, padding=1), nn.ReLU()])
        self.refine = nn.Sequential(
            *stem,
            *body,
            nn.Conv2d(24, 4, 3, padding=1), nn.Tanh(),
        )
        nn.init.zeros_(self.refine[-2].weight)
        nn.init.zeros_(self.refine[-2].bias)
        weights = torch.exp(-decay*torch.arange(history, dtype=torch.float32))
        self.register_buffer("time_weights", weights/weights.sum())

    def forward(self, tokens, phases, lengths, decoded_raw):
        from baseline import upscale
        batch, history, width = tokens.shape
        if history != self.history or width != 640 or decoded_raw.shape != (batch, history, 1, 4, 160):
            raise ValueError("requires tokens [B,T,640] and decoded RAW [B,T,1,4,160]")
        if not decoded_raw.is_floating_point():
            raise ValueError("decoded RAW must be float, normalized to [0,1]")
        high = upscale(decoded_raw.reshape(-1, 1, 4, 160), phases.reshape(-1))
        high = high.reshape(batch, history, 1, 16, 640)
        current = high[:, 0]
        fused = (high*self.time_weights.view(1, history, 1, 1, 1)).sum(1)
        coded = self.encoder(self.embedding(tokens.reshape(-1, 640)).transpose(1, 2))
        coded = (coded.reshape(batch, history, 8)*self.time_weights.view(1, history, 1)).sum(1)
        conditioning = torch.cat((coded, self.phase(phases[:, 0]), lengths[:, :1].float()/638), 1)
        conditioning = conditioning[:, :, None, None].expand(-1, -1, 8, 320)
        features = torch.cat((F.pixel_unshuffle(current, 2), F.pixel_unshuffle(fused, 2), conditioning), 1)
        residual = F.pixel_shuffle(self.refine(features), 2)*self.residual_scale
        return (current+residual).clamp(0, 1)


class SimpleRawModel(nn.Module):
    """v9: current strip with optional previous-frame strip, learned concatenation."""
    requires_decoded = True

    def __init__(self, history=1, layers=2, variant="relu8", kernel_width=3, v9_upper_strip=False):
        super().__init__()
        if history not in (1, 2):
            raise ValueError("v9 requires history=1 or 2")
        if layers not in (1, 2, 3, 4, 5):
            raise ValueError("v9 supports one to five convolution layers")
        if kernel_width not in (3, 9):
            raise ValueError("v9 kernel width must be 3 or 9")
        if variant not in ("plain4", "plain8", "plain16", "plain32", "relu4", "relu8", "relu16", "relu24", "relu32", "relu64", "relu128", "relu256"):
            raise ValueError("unknown v9 variant")
        self.history = history
        self.v9_upper_strip = v9_upper_strip
        self.input_slots = history + int(v9_upper_strip)
        input_channels = 4*self.input_slots + int(v9_upper_strip)
        self.v9_layers = layers
        self.v9_variant = variant
        self.v9_kernel_width = kernel_width
        kernel, padding = (3, kernel_width), (1, kernel_width//2)
        if layers == 1:
            self.refine = nn.Sequential(nn.Conv2d(input_channels, 4, kernel, padding=padding))
        else:
            hidden = {"plain4": 4, "plain8": 8, "plain16": 16, "plain32": 32, "relu4": 4, "relu8": 8, "relu16": 16, "relu24": 24, "relu32": 32, "relu64": 64, "relu128": 128, "relu256": 256}[variant]
            body = [nn.Conv2d(input_channels, hidden, kernel, padding=padding)]
            if variant in ("relu4", "relu8", "relu16", "relu24", "relu32", "relu64", "relu128", "relu256"):
                body.append(nn.ReLU())
            for _ in range(layers-2):
                body.append(nn.Conv2d(hidden, hidden, kernel, padding=padding))
                if variant in ("relu4", "relu8", "relu16", "relu24", "relu32", "relu64", "relu128", "relu256"):
                    body.append(nn.ReLU())
            body.append(nn.Conv2d(hidden, 4, kernel, padding=padding))
            self.refine = nn.Sequential(*body)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, tokens, phases, lengths, decoded_raw):
        # tokens/lengths are accepted only for compatibility with the shared CLI.
        # Each frame is phase-aligned separately; no fixed temporal averaging.
        from baseline import upscale
        if decoded_raw.ndim != 5 or decoded_raw.shape[1:] != (self.input_slots, 1, 4, 160):
            raise ValueError("v9 requires decoded RAW [B,history,1,4,160]")
        if not decoded_raw.is_floating_point() or phases.shape != (len(decoded_raw), self.input_slots):
            raise ValueError("v9 requires normalized floating RAW and phases [B,history]")
        base = upscale(decoded_raw[:, 0], phases[:, 0])
        planes = F.pixel_unshuffle(base, 2)
        inputs = torch.cat([planes] + [F.pixel_unshuffle(upscale(decoded_raw[:, k], phases[:, k]), 2)
                                      for k in range(1, self.history)], dim=1)
        if self.v9_upper_strip:
            valid = (phases[:, -1] >= 0).to(planes.dtype).view(-1, 1, 1, 1)
            upper = F.pixel_unshuffle(upscale(decoded_raw[:, -1], phases[:, -1].clamp_min(0)), 2)
            inputs = torch.cat([inputs, upper*valid, valid.expand(-1, 1, *planes.shape[-2:])], dim=1)
        return F.pixel_shuffle(planes+self.refine(inputs), 2).clamp(0, 1)


def resolve_v8_variant(config):
    # Historical v8 checkpoints already save residual_layers=4. Bare/new configs
    # select the requested two-layer 24->96->24 backbone.
    variant = config.get("v8_variant", "legacy_four_layer" if config.get("residual_layers") == 4
                         else "expand96_two_layer")
    if variant not in ("expand96_two_layer", "legacy_four_layer"):
        raise ValueError("unknown v8 variant")
    return variant


def build_model(config):
    if config.get("previous_block", False):
        raise ValueError("upper-strip checkpoints are retired; use the original v7 checkpoint or archived experiment source")
    if config.get("architecture") == "v9":
        # Existing v9 checkpoints have no v9_layers field and use two layers.
        return SimpleRawModel(config["history"], layers=config.get("v9_layers", 2),
                              variant=config.get("v9_variant", "relu8"),
                              kernel_width=config.get("v9_kernel_width", 3),
                              v9_upper_strip=config.get("v9_upper_strip", False))
    if config.get("architecture") == "v8":
        if resolve_v8_variant(config) == "expand96_two_layer":
            return HybridRawModel(config["history"], config["decay"], config.get("residual_scale", 0.5),
                                  residual_blocks=5, residual_expansion=4, residual_layers=2)
        return HybridRawModel(config["history"], config["decay"], config.get("residual_scale", 0.5),
                              residual_blocks=5, residual_expansion=2, residual_layers=4)
    if config.get("architecture") == "v7":
        return HybridRawModel(config["history"], config["decay"], config.get("residual_scale", 0.5),
                              residual_blocks=5, residual_expansion=2)
    if config.get("architecture") == "v6":
        return HybridRawModel(config["history"], config["decay"], config.get("residual_scale", 0.5), residual_blocks=6)
    if config.get("architecture") == "v5":
        return HybridRawModel(config["history"], config["decay"], config.get("residual_scale", 0.5), residual_blocks=5)
    if config.get("architecture") == "v4":
        return HybridRawModel(config["history"], config["decay"], config.get("residual_scale", 0.5), residual_blocks=4)
    if config.get("architecture") == "v3":
        return HybridRawModel(config["history"], config["decay"], config.get("residual_scale", 0.5))
    if config.get("architecture", "v1") == "v2":
        return ByteRawModelV2(config["history"], config["decay"], config.get("residual_scale", 0.1))
    return ByteRawModel(config["history"], config["decay"])
