"""Predict one RAW8 strip from decoded history without reading target data."""
import argparse
from contextlib import closing
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from evaluate import demosaic
from record_experiment import digest, run_logged, write_json
from v11.data import connect
from v13.base import features
from v13.data import fingerprints, load_inputs
from v13.model import ResidualCNN, mix_residual_bands, quantize


def window(rows, frame, block, first):
    header, _, _ = rows[frame, block]
    records = []
    for t in range(frame, max(first, frame - 15) - 1, -1):
        item = rows.get((t, block))
        if item is None or item[0] != header:
            break
        records.append(item)
    _, phases, decoded = zip(*records)
    return torch.stack(decoded), phases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    strength = parser.add_mutually_exclusive_group()
    strength.add_argument("--alpha", type=float, choices=(0., 0.5, 1.), default=1.)
    strength.add_argument("--band-alpha", type=float, choices=(0., 0.5, 0.75, 1.), nargs=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.block < 30 or not 0 <= args.first <= args.frame:
        parser.error("invalid block or source frame range")
    if any(args.out.with_suffix(suffix).exists() for suffix in (".raw", ".png", ".json")):
        raise ValueError("select a new output basename")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    provenance = fingerprints(args.db, args.scene_plan)
    if state["config"]["fingerprints"] != provenance:
        raise ValueError("incompatible checkpoint")
    plan = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    first = max(args.first, max((cut for cut in plan["scene_starts"] if cut <= args.frame), default=0))
    with closing(connect(args.db)) as conn:
        rows = load_inputs(conn, max(first, args.frame - 15), args.frame)
    decoded, phases = window(rows, args.frame, args.block, first)
    sample = features(decoded, phases).unsqueeze(0)
    fallback = "warmup" if len(phases) < 16 else "phase_coverage" if len(set(phases)) < 16 else None
    with torch.inference_mode():
        if fallback or args.alpha == 0 or args.band_alpha == [0., 0.]:
            raw = sample[:, :1]
        else:
            config = state["config"]
            model = ResidualCNN(config["variant"], config.get("width", 32), config.get("context", "compact"))
            model = model.to(args.device, memory_format=torch.channels_last)
            model.load_state_dict(state["model"])
            model.eval()
            payload = sample.to(args.device)
            if args.band_alpha:
                base = payload[:, :1].float() / 255
                raw = quantize(mix_residual_bands(base, model(payload), *args.band_alpha)).cpu()
            else:
                raw = model.predict(payload, args.alpha).cpu()
        rgb = demosaic(raw.float() / 255).squeeze(0).permute(1, 2, 0).numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".raw").write_bytes(raw.numpy().tobytes())
    Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(args.out.with_suffix(".png"))
    write_json(args.out.with_suffix(".json"), {"frame": args.frame, "block": args.block,
                 "alpha": None if args.band_alpha else args.alpha, "band_alpha": args.band_alpha,
                 "history": len(phases), "fallback": fallback, "fingerprints": provenance,
                 "checkpoint_sha256": digest(args.checkpoint)})


if __name__ == "__main__":
    run_logged("inference", main)
