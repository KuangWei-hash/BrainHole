"""Target-free V10 inference for one current/previous strip pair."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from baseline import DecodedCache, fingerprint
from evaluate import demosaic
from record_experiment import run_logged
from v10.data import inference_inputs
from v10.diffusion import Schedule, anchor_image, compose, generate_residual
from v10.model import load_detail_anchor, load_v10, validate_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--decoded-cache", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-policy", choices=("per_frame", "fixed_screen"), default="per_frame")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("select new output directory")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, saved = load_v10(args.checkpoint, device)
    anchor = load_detail_anchor(saved, device)
    config = saved["config"]
    cache = DecodedCache(args.decoded_cache, args.db)
    if cache.metadata["header_id"] != config["header_id"]:
        raise ValueError("checkpoint/header mismatch")
    low, high, phases = inference_inputs(args.db, cache, args.frame, args.block, config["header_id"])
    validate_inputs(low, high, phases, check_pair=True)
    low, high, phases = low.to(device), high.to(device), phases.to(device)
    base = anchor_image(anchor, low, high, phases)
    schedule = Schedule(config["timesteps"]).to(device)
    residual = generate_residual(model, schedule, low, high, phases, [(args.frame, args.block)],
        seed=args.seed, video_id=cache.metadata["database_sha256"], steps=args.steps,
        policy=args.noise_policy, mode=config["mode"], base=base)
    output = compose(high, residual, model.residual_scale, args.gamma, base=base)
    args.out.mkdir(parents=True)
    (args.out/"generated.raw").write_bytes((output[0, 0]*255).round().byte().cpu().numpy().tobytes())
    rgb = demosaic(output)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    Image.fromarray((rgb*255).round().astype(np.uint8)).save(args.out/"generated.png")
    report = {"frame": args.frame, "block": args.block, "steps": args.steps, "gamma": args.gamma,
              "seed": args.seed, "noise_policy": args.noise_policy, "mode": config["mode"],
              "target_read": False, "checkpoint": str(args.checkpoint), "checkpoint_sha256": fingerprint(args.checkpoint),
              "decoded_cache": cache.metadata, "scope": "single strip inference, no quality scoring"}
    (args.out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run_logged("evaluation", main)
