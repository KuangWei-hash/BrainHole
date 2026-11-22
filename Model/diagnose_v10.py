"""Frozen-checkpoint, labeled timestep diagnostics and target-free sampling traces."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from baseline import DecodedCache, fingerprint
from record_experiment import run_logged
from v10.data import V10Dataset, stratified_indices
from v10.diffusion import Schedule, anchor_image, compose, generate_residual, keyed_noise, target_residual
from v10.model import load_detail_anchor, load_v10, make_bilinear


@torch.inference_mode()
def traced_sample(model, schedule, decoded, high, phases, keys, video_id, steps, base_raw=None):
    condition = model.prepare_condition(decoded, high, phases)
    noisy = keyed_noise(keys, 0, video_id, decoded.device)
    grid = torch.linspace(schedule.timesteps-1, 0, steps, device=decoded.device).round().long()
    base = F.pixel_unshuffle(high[:, 0] if base_raw is None else base_raw, 2)
    lower, upper = -base/model.residual_scale, (1-base)/model.residual_scale
    trace = []
    for i, t in enumerate(grid):
        times = t.expand(len(decoded))
        prediction = model(noisy, times, condition)
        clean, _ = schedule.decode_prediction(noisy, times, prediction, model.prediction_type)
        image = base+model.residual_scale*clean
        trace.append({"t": int(t), "alpha_bar": float(schedule.alpha_bar[t]),
                      "clip_fraction": float(((image < 0) | (image > 1)).float().mean()),
                      "unclipped_raw_min": float(image.min()), "unclipped_raw_max": float(image.max()),
                      "residual_rms": float(clean.square().mean().sqrt())})
        clean = clean.clamp(lower, upper)
        if i == len(grid)-1:
            return clean, trace
        a, next_a = schedule.alpha_bar[t], schedule.alpha_bar[grid[i+1]]
        epsilon = (noisy-a.sqrt()*clean)/(1-a).sqrt()
        noisy = next_a.sqrt()*clean+(1-next_a).sqrt()*epsilon


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--decoded-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--steps", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if min(args.count, args.sample_count, *args.steps) < 1 or args.out.exists():
        raise ValueError("invalid diagnostic budget or preexisting output")
    args.out.mkdir(parents=True)
    # Freeze bytes because the training process may replace best.pt concurrently.
    snapshot = args.out/"source-checkpoint.pt"
    snapshot.write_bytes(args.checkpoint.read_bytes())
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, saved = load_v10(snapshot, device)
    anchor = load_detail_anchor(saved, device)
    config = saved["config"]
    if config["mode"] != "diffusion":
        raise ValueError("diagnostic requires diffusion checkpoint")
    cache = DecodedCache(args.decoded_cache, args.db)
    if cache.metadata["database_sha256"] != config["database_sha256"]:
        raise ValueError("diagnostic must use original validation source")
    dataset = V10Dataset(args.db, cache, frame_start=config["split"]["boundary"],
                         frame_stop=config["split"].get("val_stop_frame"))
    lookup = {key: i for i, key in enumerate(dataset.keys)}
    candidates = [tuple(k) for k in config["validation_keys"]]
    selected = [candidates[i] for i in stratified_indices(candidates, min(args.count, len(candidates))) ]
    indices = [lookup[k] for k in selected]
    schedule = Schedule(config["timesteps"]).to(device)
    times = sorted({min(t, schedule.timesteps-1) for t in (0, 100, 300, 500, 700, 900, 950, 975, 990, 999)})
    totals = {t: [0., 0., 0., 0., 0] for t in times}
    batches = []
    for decoded, phases, target, keys in DataLoader(Subset(dataset, indices), batch_size=8):
        decoded, phases, target = decoded.to(device), phases.to(device), target.to(device)
        high = make_bilinear(decoded, phases)
        base = anchor_image(anchor, decoded, high, phases)
        condition = model.prepare_condition(decoded, high, phases)
        signal = target_residual(target, base, detail_only=anchor is not None)
        residual = signal/model.residual_scale
        base_planes = F.pixel_unshuffle(base, 2)
        truth = (F.pixel_unshuffle(target, 2) if anchor is None else base_planes+signal)
        noise = keyed_noise(keys.tolist(), 777, config["database_sha256"], device)
        for t in times:
            labels = torch.full((len(decoded),), t, dtype=torch.long, device=device)
            noisy = schedule.add_noise(residual, labels, noise)
            prediction = model(noisy, labels, condition)
            clean, epsilon = schedule.decode_prediction(noisy, labels, prediction, model.prediction_type)
            raw = base_planes+model.residual_scale*clean
            stats = totals[t]
            stats[0] += (epsilon-noise).square().sum().item()
            stats[1] += (raw-truth).abs().sum().item()
            stats[2] += (raw.clamp(0, 1)-truth).abs().sum().item()
            stats[3] += ((raw < 0) | (raw > 1)).sum().item()
            stats[4] += raw.numel()
        batches.append((decoded, high, phases, target, base, keys.tolist()))
    teacher = {str(t): {"alpha_bar": float(schedule.alpha_bar[t]), "epsilon_mse": a/n,
                       "x0_unclipped_raw_mae": b/n*255, "x0_clipped_raw_mae": c/n*255,
                       "clip_fraction": d/n} for t, (a, b, c, d, n) in totals.items()}
    sampling = {}
    for steps in args.steps:
        errors, count, traces = 0., 0, []
        started = time.perf_counter()
        for decoded, high, phases, target, base, keys in batches:
            if count >= args.sample_count:
                break
            size = min(len(decoded), args.sample_count-count)
            decoded, high, phases, target, keys = decoded[:size], high[:size], phases[:size], target[:size], keys[:size]
            base = base[:size]
            residual, trace = traced_sample(model, schedule, decoded, high, phases, keys, config["database_sha256"], steps, base)
            if count == 0:
                shipped = generate_residual(model, schedule, decoded, high, phases, keys,
                    video_id=config["database_sha256"], steps=steps, base=base)
                torch.testing.assert_close(residual, shipped, atol=1e-5, rtol=1e-5)
            errors += (compose(high, residual, model.residual_scale, base=base)-target).abs().sum().item()
            count += size
            traces.append({"keys": keys, "steps": trace})
        sampling[str(steps)] = {"samples": count, "raw_mae": errors/(count*10240)*255,
                                "seconds": time.perf_counter()-started, "trace": traces}
    report = {"samples": len(indices), "epoch": saved["epoch"], "prediction_type": model.prediction_type,
              "checkpoint_source": str(args.checkpoint), "frozen_checkpoint": str(snapshot),
              "checkpoint_sha256": fingerprint(snapshot), "keys": [list(k) for k in selected],
              "teacher_forced": teacher, "sampling": sampling,
              "teacher_reference": "B plus labeled high-pass residual" if anchor is not None else "original RAW",
              "scope": "Labeled timestep diagnostic and limited sampling, not full holdout or quality acceptance"}
    (args.out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"epoch": saved["epoch"], "teacher_forced": teacher,
                      "sampling": {k: {n: v for n, v in s.items() if n != "trace"} for k, s in sampling.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    run_logged("evaluation", main)
