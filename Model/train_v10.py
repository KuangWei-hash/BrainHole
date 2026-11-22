"""Train V10 dual-input RAW residual diffusion, or matched non-generative controls."""
import argparse
from contextlib import closing
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from baseline import DecodedCache, fingerprint
from data import connect
from evaluate import demosaic
from record_experiment import run_logged, update_active_record
from v10.data import V10Dataset, fit_residual_scale, stratified_indices
from v10.model import ResidualDenoiser, load_v10, make_bilinear
from v10.diffusion import Schedule, anchor_image, compose, generate_residual, low_pass, target_residual


def atomic_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train_epoch(model, schedule, loader, optimizer, device, args, anchor=None):
    model.train()
    total, count, steps = 0., 0, 0
    started = time.perf_counter()
    for decoded, phases, target, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        size = len(decoded)
        for first in range(0, size, args.microbatch):
            low = decoded[first:first+args.microbatch].to(device)
            phase = phases[first:first+args.microbatch].to(device)
            truth = target[first:first+args.microbatch].to(device)
            high = make_bilinear(low, phase)
            base = anchor_image(anchor, low, high, phase)
            residual = target_residual(truth, base, detail_only=anchor is not None)/model.residual_scale
            condition = model.prepare_condition(low, high, phase)
            noise = torch.randn_like(residual)
            if args.mode == "diffusion":
                t = torch.randint(schedule.timesteps, (len(low),), device=device)
                noisy = schedule.add_noise(residual, t, noise)
                predicted = model(noisy, t, condition)
                objective_target = schedule.training_target(residual, t, noise, args.prediction_type)
                loss = F.mse_loss(predicted, objective_target)
                if args.low_frequency_weight:
                    active = t < int(schedule.timesteps*0.8)
                    if active.any():
                        clean, _ = schedule.decode_prediction(noisy[active], t[active], predicted[active], args.prediction_type)
                        out = F.pixel_unshuffle(base[active], 2)+model.residual_scale*clean
                        guard = F.mse_loss(low_pass(out), low_pass(F.pixel_unshuffle(truth[active], 2)))
                        loss = loss+args.low_frequency_weight*guard
            else:
                t = torch.full((len(low),), 0 if args.mode == "deterministic" else schedule.timesteps-1,
                               dtype=torch.long, device=device)
                predicted = model(torch.zeros_like(noise) if args.mode == "deterministic" else noise, t, condition)
                delta = (predicted-residual)*model.residual_scale
                loss = (delta.square()+1e-6).sqrt().mean() if args.mode == "deterministic" else delta.abs().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite V10 training loss")
            (loss*len(low)/size).backward()
            total += loss.item()*len(low)
            count += len(low)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite V10 gradient")
        optimizer.step()
        steps += 1
        if steps % 100 == 0:
            print(f"  optimizer_step={steps} loss={total/count:.6f}", flush=True)
        if args.max_steps and steps >= args.max_steps:
            break
    return {"objective": total/count, "samples": count, "optimizer_steps": steps,
            "seconds": time.perf_counter()-started}


@torch.inference_mode()
def validate(model, schedule, loader, device, config, max_steps=0, anchor=None):
    model.eval()
    totals = {str(g): [0., 0., 0] for g in config["gammas"]}
    start = time.perf_counter()
    preview = None
    for step, (decoded, phases, target, keys) in enumerate(loader, 1):
        decoded, phases, target = decoded.to(device), phases.to(device), target.to(device)
        bilinear = make_bilinear(decoded, phases)
        base = anchor_image(anchor, decoded, bilinear, phases)
        residual = generate_residual(model, schedule, decoded, bilinear, phases, keys.tolist(),
            seed=0, video_id=config["database_sha256"], steps=config["sample_steps"], mode=config["mode"], base=base)
        for gamma in config["gammas"]:
            output = compose(bilinear, residual, model.residual_scale, gamma, base=base)
            delta = output-target
            values = totals[str(gamma)]
            values[0] += delta.abs().sum().item()
            values[1] += delta.square().sum().item()
            values[2] += delta.numel()
            if preview is None and gamma == 1.:
                preview = torch.cat((target[:1], base[:1], output[:1]))
        if max_steps and step >= max_steps:
            break
    metrics = {g: {"mae_8bit": a/n*255, "psnr_raw": -10*math.log10(max(s/n, 1e-12)),
                   "samples": n//10240} for g, (a, s, n) in totals.items()}
    return {**metrics["1.0"], "gammas": metrics, "seconds": time.perf_counter()-start,
            "note": "Generated output at seed0, gamma1. Fixed validation subset; not full holdout or perceptual acceptance."}, preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--decoded-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("diffusion", "deterministic", "single_noise"), default="diffusion")
    parser.add_argument("--condition", choices=("both", "decoded", "bilinear"), default="both")
    parser.add_argument("--prediction-type", choices=("epsilon", "v"), default="epsilon")
    parser.add_argument("--anchor-checkpoint", type=Path, help="freeze deterministic V10 and learn high-pass target-minus-anchor")
    parser.add_argument("--train-start-frame", type=int, default=0)
    parser.add_argument("--val-stop-frame", type=int)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--low-frequency-weight", type=float, default=0.)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--validation-count", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--val-steps", type=int, default=0)
    parser.add_argument("--scale-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.mode != "diffusion" and args.prediction_type != "epsilon":
        parser.error("prediction-type v applies only to diffusion")
    if args.anchor_checkpoint and (args.mode != "diffusion" or args.prediction_type != "v"):
        parser.error("the frozen detail anchor requires v-prediction diffusion")
    if (min(args.epochs, args.batch_size, args.microbatch, args.lr, args.validation_count) <= 0 or
            min(args.max_steps, args.val_steps, args.scale_steps, args.low_frequency_weight) < 0 or
            not 1 <= args.sample_steps <= args.timesteps):
        parser.error("invalid budget, rate or diffusion settings")
    if args.out.exists():
        raise ValueError("select a new output directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    cache = DecodedCache(args.decoded_cache, args.db)
    with closing(connect(args.db)) as db:
        frames = [r[0] for r in db.execute("SELECT DISTINCT frame_index FROM samples ORDER BY frame_index")]
    boundary = frames[int(len(frames)*0.8)]
    if not 0 <= args.train_start_frame < boundary-1 or (args.val_stop_frame is not None and args.val_stop_frame <= boundary+1):
        parser.error("training and validation frame ranges need at least two frames each")
    train = V10Dataset(args.db, cache, frame_start=args.train_start_frame, frame_stop=boundary)
    validation = V10Dataset(args.db, cache, frame_start=boundary, frame_stop=args.val_stop_frame)
    if train.headers != validation.headers or len(train.headers) != 1:
        raise ValueError("one shared codec/header required")
    indices = stratified_indices(validation.keys, args.validation_count)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(architecture="v10", history=2, input_contract="decoded_and_phase_aligned_bilinear_v1",
        split={"kind": "temporal_smoke_only", "boundary": boundary, "train_start_frame": args.train_start_frame,
               "val_stop_frame": args.val_stop_frame}, header_id=next(iter(train.headers)),
        database_sha256=cache.metadata["database_sha256"], train_samples=len(train),
        validation_samples=len(indices), validation_full_samples=len(validation),
        validation_keys=[list(validation.keys[i]) for i in indices], device_used=str(device),
        torch=str(torch.__version__), gammas=[0., 0.25, 0.5, 0.75, 1.],
        selection="provisional generated RAW MAE at gamma1 seed0 on fixed subset; no perceptual acceptance",
        sampler="DDIM eta0; x0 clipped to physical RAW bounds; corrected epsilon", noise_schedule="cosine_s0.008_beta_max0.999",
        sample_seed=0, noise_policy="per_frame", precision="FP32", weight_decay=1e-4,
        coverage="flow_check" if args.max_steps or args.val_steps or args.scale_steps else "full_training_fixed_subset_validation")
    anchor, anchor_state = None, None
    if args.anchor_checkpoint:
        anchor, saved = load_v10(args.anchor_checkpoint, device)
        anchor_config = saved["config"]
        if (anchor_config["mode"] != "deterministic" or anchor_config["database_sha256"] != config["database_sha256"] or
                anchor_config["header_id"] != config["header_id"] or anchor_config["split"]["boundary"] > boundary):
            raise ValueError("detail anchor must be deterministic and trained on the same source with no holdout overlap")
        anchor.requires_grad_(False)
        anchor_state = {"model": saved["model"], "config": anchor_config}
        config.update(residual_target="cfa_gaussian_highpass_target_minus_frozen_anchor", anchor_epoch=saved["epoch"],
                      anchor_sha256=fingerprint(args.anchor_checkpoint))
    args.out.mkdir(parents=True)
    (args.out/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Fitting residual scale using train split only", flush=True)
    stats = fit_residual_scale(DataLoader(train, batch_size=64), device, args.scale_steps, anchor)
    config["residual_statistics"] = stats
    model = ResidualDenoiser(args.channels, args.condition, args.prediction_type).to(device)
    model.residual_scale.copy_(torch.tensor(stats["std"], device=device).view(1, 4, 1, 1))
    config["parameters"] = sum(p.numel() for p in model.parameters())
    (args.out/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in config.items() if k != "validation_keys"}, indent=2), flush=True)
    schedule = Schedule(args.timesteps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(Subset(validation, indices), batch_size=args.batch_size)
    best = float("inf")
    for epoch in range(1, args.epochs+1):
        training = train_epoch(model, schedule, train_loader, optimizer, device, args, anchor)
        metrics, preview = validate(model, schedule, val_loader, device, config, args.val_steps, anchor)
        record = {"epoch": epoch, "train": training, "validation": metrics}
        with (args.out/"metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record)+"\n")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                 "epoch": epoch, "torch_rng_state": torch.get_rng_state(),
                 "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        if anchor_state is not None:
            state["detail_anchor"] = anchor_state
        atomic_checkpoint(args.out/"last.pt", state)
        if metrics["mae_8bit"] < best:
            best = metrics["mae_8bit"]
            atomic_checkpoint(args.out/"best.pt", state)
        if epoch in (5, 10, 20) or epoch == args.epochs:
            atomic_checkpoint(args.out/f"epoch-{epoch:03d}.pt", state)
        if preview is not None:
            rgb = demosaic(preview).clamp(0, 1).cpu()
            pixels = torch.cat(list(rgb), dim=1).permute(1, 2, 0).numpy()
            Image.fromarray((pixels*255).round().astype(np.uint8)).save(args.out/f"epoch-{epoch:03d}-target-base-generated.png")
        update_active_record()
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    run_logged("training", main)
