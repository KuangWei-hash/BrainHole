"""Estimate how much of the remaining development error could be fixed by scaling R2."""
import argparse
from pathlib import Path

import torch

from evaluate_v13 import model_predictions
from record_experiment import digest, run_logged, write_json
from v13.data import fingerprints, labels


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("db", "scene-plan", "cache", "checkpoint", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    torch.set_num_threads(3)
    provenance = fingerprints(args.db, args.scene_plan)
    cache = torch.load(args.cache, weights_only=True)
    if cache["split"] != "S2-dev" or cache["provenance"] != provenance:
        raise ValueError("expected matching S2 development data")
    x = cache["features"]
    target = labels(args.db, cache["manifest"]).short()
    baseline = x[:, :1].short()
    full, checkpoint = model_predictions(args.checkpoint, x, "cpu", 1., provenance)
    full = full.short()
    low, high = torch.minimum(baseline, full), torch.maximum(baseline, full)
    best_pixel = torch.maximum(low - target, target - high).clamp_min_(0)
    strengths = {}
    tiles = []
    for strength in (0, 0.25, 0.5, 0.75, 1):
        # Work on the quantized endpoints, with exact v12 and full-R2 bytes.
        restored = torch.lerp(baseline.float(), full.float(), strength).round().short()
        error = (restored - target).abs()
        strengths[str(strength)] = error.float().mean().item()
        tiles.append(error.reshape(len(x), 16, 40, 16).sum(dim=(1, 3)))
    tile_error = torch.stack(tiles).amin(dim=0).sum().item() / target.numel()
    report = {"split": cache["split"], "frames": [736, 779], "strips": len(x),
              "checkpoint": checkpoint, "cache_sha256": digest(args.cache), "strength_mae": strengths,
              "oracle_tile_mae": tile_error, "oracle_pixel_mae": best_pixel.float().mean().item(),
              "target_bracketed_fraction": (best_pixel == 0).float().mean().item(),
              "scope": "label-assisted diagnosis on quantized endpoints; no proposed inference output"}
    write_json(args.out / "report.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    run_logged("checks", main)
