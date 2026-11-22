"""Inference-only adapter for upstream RGB EchoSR-Lite x4, without BasicSR setup.

The upstream architecture and pretrained state dict are unchanged. Only registry,
benchmark-only tqdm and timm utility imports are replaced at load time. DropPath
is identity in evaluation, and training is deliberately unsupported here.
"""
from pathlib import Path
import subprocess

import torch
from torch import nn
from torch.nn import functional as F

from baseline import fingerprint


class InferenceDropPath(nn.Identity):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob:
            raise RuntimeError("EchoSR adapter is inference-only; call eval()")
        return x


class EchoSRLite:
    def __init__(self, root, weights, device, guard=8):
        root, weights = Path(root), Path(weights)
        source_path = root/"basicsr/archs/EchoSR_arch.py"
        source = source_path.read_text(encoding="utf-8")
        substitutions = {
            "import tqdm": "# benchmark-only import omitted",
            "from timm.models.layers import DropPath, to_2tuple, trunc_normal_": "# evaluation-compatible utilities supplied by adapter",
            "from basicsr.utils.registry import ARCH_REGISTRY": "# standalone loading needs no BasicSR registry",
            "@ARCH_REGISTRY.register()": "# upstream registry decorator omitted",
        }
        for old, new in substitutions.items():
            if source.count(old) != 1:
                raise ValueError("upstream source changed; review import adaptation: "+old)
            source = source.replace(old, new)
        namespace = {"__name__": "echosr_upstream", "__file__": str(source_path),
                     "DropPath": InferenceDropPath, "to_2tuple": lambda x: (x, x),
                     "trunc_normal_": nn.init.trunc_normal_}
        exec(compile(source, str(source_path), "exec"), namespace)
        self.model = namespace["EchoSR"](upscale=4, in_chans=3, img_size=72, img_range=1.,
            d_state=16, depths=[2, 3, 2, 3], kernel_sizes=[0, 5, 11, 17], embed_dim=36,
            mlp_ratio=1.5, upsampler="pixelshuffledirect", resi_connection="1conv")
        saved = torch.load(weights, map_location="cpu", weights_only=True)
        key = "params_ema" if "params_ema" in saved else "params" if "params" in saved else None
        self.model.load_state_dict(saved[key] if key else saved, strict=True)
        self.model.to(device).eval()
        self.guard = guard
        self.metadata = {
            "repository": "https://github.com/funnyWang-Echoes/EchoSR",
            "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            "architecture": str(source_path), "architecture_sha256": fingerprint(source_path),
            "weights": str(weights), "weights_sha256": fingerprint(weights), "state_key": key,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "configuration": "official EchoSR_lite_SRx4_DF2K; embed_dim36 depths[2,3,2,3] mlp_ratio1.5",
            "training": "upstream DF2K pretrained; no project training or fine-tuning",
            "input": "phase-aware current-strip RGB, PNG8-equivalent quantization, replicated guard8 on all sides",
            "output": "RGB x4, crop guards, clamp [0,1]; RAW metrics use RGGB projection",
            "dependency_adapter": "registry/tqdm imports omitted; eval DropPath identity; torch.nn.init.trunc_normal_; no architecture/weight changes",
        }

    @torch.inference_mode()
    def __call__(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("EchoSR requires RGB BCHW")
        _, _, h, w = rgb.shape
        g = self.guard
        padded = F.pad(rgb, (g, g, g, g), mode="replicate")
        if min(padded.shape[-2:]) < 8:
            raise ValueError("EchoSR global pooling requires padded height/width >=8")
        output = self.model(padded)
        if output.shape[-2:] != (4*(h+2*g), 4*(w+2*g)) or not torch.isfinite(output).all():
            raise ValueError("invalid EchoSR output shape or nonfinite output")
        return output[:, :, 4*g:4*(g+h), 4*g:4*(g+w)].clamp(0, 1)
