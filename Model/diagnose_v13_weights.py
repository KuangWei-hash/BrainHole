"""Inspect learned age weights on development input without reading RAW targets."""
import argparse
import json
import math
from pathlib import Path

import torch

from record_experiment import digest, run_logged, write_json
from v13.model import ResidualCNN


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    torch.set_num_threads(3)
    cache = torch.load(args.cache, weights_only=True)
    if cache["split"] != "S2-dev":
        raise ValueError("use the S2 development window")
    manifest = cache["manifest"]
    value = cache["features"]
    roi = Path(__file__).with_name("v13") / "roi-bbb-q5.json"
    region = json.loads(roi.read_text(encoding="utf-8"))["regions"]["title-when-visible"]
    x0, y0, x1, y1 = [i // 2 for i in region]
    results = {}
    for name in args.checkpoint:
        state = torch.load(name, weights_only=True, map_location="cpu")
        config = state["config"]
        if config["fingerprints"] != cache["provenance"]:
            raise ValueError("checkpoint provenance mismatch")
        model = ResidualCNN(config["variant"], config.get("width", 32), config.get("context", "compact"))
        model = model.to(memory_format=torch.channels_last)
        model.load_state_dict(state["model"])
        model.eval()
        captured = []
        totals = {key: torch.zeros(18, dtype=torch.float64) for key in ("full", "title-from-750")}

        def collect(_module, _inputs, logits):
            captured.append(logits.reshape(-1, 16, 8, 320).softmax(dim=1))

        def accumulate(totals_row, weights):
            totals_row[:16] += weights.sum(dim=(1, 2), dtype=torch.float64)
            totals_row[16] += -(weights * weights.clamp_min(1e-20).log()).sum().item()
            totals_row[17] += weights.shape[1] * weights.shape[2]

        with model.fusion.score.register_forward_hook(collect):
            for index in range(0, len(value), 8):
                model(value[index:index + 8])
                for row, weights in zip(manifest[index:index + 8], captured.pop()):
                    accumulate(totals["full"], weights)
                    first_y = row["block"] * 8
                    if row["frame"] >= 750 and first_y < y1 and first_y + 8 > y0:
                        cropped = weights[:, max(y0 - first_y, 0):min(y1 - first_y, 8), x0:x1]
                        accumulate(totals["title-from-750"], cropped)
        summary = {}
        for name_roi, row in totals.items():
            mean = row[:16] / row[17]
            summary[name_roi] = {"cells": int(row[17]), "mean_by_age": mean.tolist(),
                                 "youngest_four": mean[:4].sum().item(), "oldest_four": mean[-4:].sum().item(),
                                 "entropy_fraction": (row[16] / row[17] / math.log(16)).item()}
        results[config["variant"]] = {"sha256": digest(name), "epoch": state["epoch"], "weights": summary}
        print(config["variant"], summary["title-from-750"], flush=True)
    write_json(args.out / "report.json", {"split": cache["split"], "cache_sha256": digest(args.cache),
               "roi_sha256": digest(roi), "methods": results})


if __name__ == "__main__":
    run_logged("checks", main)
