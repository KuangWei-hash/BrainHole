"""Train V11 phase-aware multi-frame fusion, or run a bounded flow check."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from record_experiment import digest, run_logged, update_active_record
from v11.data import V11Dataset, build_windows
from v11.model import FusionNet, make_bilinear


def atomic_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def baseline_mae(loader, device, steps=0):
    """Phase-aligned bilinear of the current frame: the reference every run is compared to."""
    total, count = 0.0, 0
    for index, (decoded, phases, target, _) in enumerate(loader, 1):
        decoded, phases, target = decoded.to(device), phases.to(device), target.to(device)
        base = make_bilinear(decoded[:, :1], phases[:, :1])
        total += float((base[:, 0] - target).abs().sum())*255
        count += target.numel()
        if steps and index >= steps:
            break
    return total / max(count, 1)


def validate(model, loader, device, steps=0):
    model.eval()
    total, count, squared, samples = 0.0, 0, 0.0, 0
    with torch.inference_mode():
        for index, (decoded, phases, target, _) in enumerate(loader, 1):
            decoded, phases, target = decoded.to(device), phases.to(device), target.to(device)
            output = model(decoded, phases)
            difference = output - target
            total += float(difference.abs().sum())*255
            squared += float(difference.square().sum())*255*255
            count += target.numel()
            samples += len(target)
            if steps and index >= steps:
                break
    mse = squared/max(count, 1)
    psnr = 10*math.log10(255.0*255.0/mse) if mse > 0 else float("inf")
    return {"mae_8bit": total / count, "psnr_raw": psnr, "samples": samples}


def split_windows(path, cohort_frames, train_range, validation_range, scene_starts):
    if not 0 <= train_range[0] <= train_range[1] < validation_range[0] <= validation_range[1]:
        raise ValueError("training and validation frame ranges must be disjoint and ordered")
    train = build_windows(path, cohort_frames, *train_range, scene_starts=scene_starts)
    validation = build_windows(path, cohort_frames, *validation_range, scene_starts=scene_starts)
    if not train or not validation:
        raise ValueError("both training and validation require complete windows")
    def target_key(window):
        return window[1][-1], window[0]
    return sorted(train, key=target_key), sorted(validation, key=target_key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--validation-count", type=int, default=0, help="0 evaluates the complete cohort")
    parser.add_argument("--train-range", type=int, nargs=2, default=(0, 719))
    parser.add_argument("--validation-range", type=int, nargs=2, default=(721, 899))
    parser.add_argument("--cohort-frames", type=int, default=4)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--alignment", choices=("phase", "motion"), default="phase")
    parser.add_argument("--anchor", choices=("current", "phase-mosaic"), default="current")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.frames <= args.cohort_frames <= 16:
        parser.error("require 1 <= frames <= cohort-frames <= 16")
    if min(args.epochs, args.batch_size, args.cpu_threads) < 1 or min(args.validation_count, args.max_steps) < 0:
        parser.error("epochs, batch size and thread count must be positive, limits must be nonnegative")
    if args.out.exists():
        raise ValueError("select a new output directory")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)

    scene_plan = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    database_sha256 = digest(args.db)
    if database_sha256 != scene_plan["database_sha256"]:
        raise ValueError("scene plan belongs to a different database")
    scene_starts = sorted(set(scene_plan["scene_starts"]))
    train_windows, validation_windows = split_windows(
        args.db, args.cohort_frames, args.train_range, args.validation_range, scene_starts
    )
    validation_count = min(args.validation_count or len(validation_windows), len(validation_windows))
    selected = [validation_windows[i * len(validation_windows) // validation_count]
                for i in range(validation_count)]
    train_subset = V11Dataset(args.db, args.frames, windows=train_windows)
    validation_subset = V11Dataset(args.db, args.frames, windows=selected)
    keys = {"train": [[ids[-1], block] for block, ids in train_windows],
            "validation_pool": [[ids[-1], block] for block, ids in validation_windows],
            "validation": [[ids[-1], block] for block, ids in selected]}
    args.out.mkdir(parents=True)
    keys_path = args.out/"keys.json"
    keys_path.write_text(json.dumps(keys, separators=(",", ":")), encoding="utf-8")

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(architecture="v11", history=args.frames,
                  input_contract="multi_frame_phase_aware_fusion_v2",
                  database_sha256=database_sha256, scene_plan_sha256=digest(args.scene_plan),
                  keys_sha256=digest(keys_path), scene_starts=scene_starts,
                  train_windows=len(train_windows), validation_windows=len(validation_windows),
                  validation_samples=len(validation_windows),
                  val_steps=math.ceil(len(selected)/args.batch_size) if args.validation_count else 0,
                  validation_selected=len(selected),
                  validation_frame_range=[selected[0][1][-1], selected[-1][1][-1]],
                  device_used=str(device), torch=str(torch.__version__),
                  loss="L1 on normalized RAW", precision="FP32",
                  coverage="flow_check" if args.max_steps else "full_training")
    model = FusionNet(args.frames, args.channels, args.blocks,
                      alignment=args.alignment, anchor=args.anchor).to(device)
    config["parameters"] = model.parameter_count()
    (args.out/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(validation_subset, batch_size=args.batch_size)
    baseline = baseline_mae(validation_loader, device)
    print(json.dumps({"baseline_bilinear_mae_8bit": baseline}), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    for epoch in range(1, args.epochs+1):
        model.train()
        total, count, steps = 0.0, 0, 0
        started = time.perf_counter()
        for decoded, phases, target, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            decoded, phases, target = decoded.to(device), phases.to(device), target.to(device)
            output = model(decoded, phases)
            loss = (output - target).abs().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite V11 training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss)*len(decoded)
            count += len(decoded)
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        metrics = validate(model, validation_loader, device)
        record = {"epoch": epoch, "train_loss_normalized": total/max(count, 1),
                  "seconds": time.perf_counter()-started, "steps": steps, "train_samples": count,
                  "validation": metrics, "baseline_bilinear_mae_8bit": baseline}
        with (args.out/"metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record)+"\n")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                 "epoch": epoch, "torch_rng_state": torch.get_rng_state()}
        atomic_checkpoint(args.out/"last.pt", state)
        if metrics["mae_8bit"] < best:
            best = metrics["mae_8bit"]
            atomic_checkpoint(args.out/"best.pt", state)
        update_active_record()
        print(json.dumps(record), flush=True)
        if args.max_steps:
            break


if __name__ == "__main__":
    run_logged("training", main)
