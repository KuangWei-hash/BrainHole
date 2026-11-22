"""Train one preregistered v13 variant or run the 32-strip S0 fit check."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from evaluate import demosaic
from record_experiment import digest, run_logged, update_active_record, write_json
from v13.data import atomic_save, fingerprints, labels
from v13.fusion import FUSION_VARIANTS, GUIDED_VARIANT, GUIDE_TEMPERATURE, GUIDE_WEIGHT, NATIVE_VARIANTS
from v13.model import ResidualCNN, loss_for, quantize
from v13.pairs import adjacent_groups, pair_loss, previous_indices


@torch.inference_mode()
def validation(model, x, y):
    candidate_abs = baseline_abs = candidate_sq = baseline_sq = clipped = 0
    rgb_abs = rgb_base = rgb_sq = rgb_base_sq = 0
    by_age = torch.zeros(16, 2, device=x.device)
    by_block = torch.zeros(30, 2, device=x.device)
    by_phase = torch.zeros(16, 2, device=x.device)
    phase_counts = torch.zeros(16, device=x.device)
    fine = torch.arange(len(x), device=x.device)
    for batch in fine.split(16):
        inp, target = x[batch], y[batch]
        raw = model(inp)
        clipped += ((raw < 0) | (raw > 1)).sum().item()
        predicted, base = quantize(raw).float(), inp[:, :1].float()
        error, base_error = predicted - target, base - target
        abs_error, abs_base = error.abs(), base_error.abs()
        candidate_abs += abs_error.sum().item()
        baseline_abs += abs_base.sum().item()
        candidate_sq += error.square().sum().item()
        baseline_sq += base_error.square().sum().item()
        rgb_target = demosaic(target.float() / 255)
        corr = (demosaic(predicted / 255) - rgb_target) * 255
        old = (demosaic(base / 255) - rgb_target) * 255
        rgb_abs += corr.abs().sum().item()
        rgb_base += old.abs().sum().item()
        rgb_sq += corr.square().sum().item()
        rgb_base_sq += old.square().sum().item()
        for column, err in enumerate((abs_error, abs_base)):
            by_age[:, column].scatter_add_(0, inp[:, 2:3].reshape(-1).long(), err.reshape(-1))
        key_means = torch.stack((abs_error.mean((1, 2, 3)), abs_base.mean((1, 2, 3))), dim=1)
        by_block.index_add_(0, batch % 30, key_means)
        # Complete BBB windows contain one current owner cell in the top-left 8x8 tile.
        phase = (inp[:, 2, :8:2, :8:2] == 0).reshape(-1, 16).long().argmax(dim=1)
        by_phase.index_add_(0, phase, key_means)
        phase_counts.index_add_(0, phase, torch.ones_like(phase, dtype=torch.float))
    pixels = y.numel()
    return {"mae_8bit": candidate_abs / pixels, "v12_mae_8bit": baseline_abs / pixels,
            "psnr_raw": -10 * math.log10(max(candidate_sq / pixels / 255**2, 1e-12)),
            "v12_psnr": -10 * math.log10(max(baseline_sq / pixels / 255**2, 1e-12)),
            "rgb_mae_8bit": rgb_abs / (3 * pixels), "v12_rgb_mae_8bit": rgb_base / (3 * pixels),
            "rgb_psnr": -10 * math.log10(max(rgb_sq / (3 * pixels) / 255**2, 1e-12)),
            "v12_rgb_psnr": -10 * math.log10(max(rgb_base_sq / (3 * pixels) / 255**2, 1e-12)),
            "clipped_fraction": clipped / pixels,
            "by_age_mae_candidate_v12": (by_age / (pixels / 16)).tolist(),
            "by_block_mae_candidate_v12": (by_block / (len(x) / 30)).tolist(),
            "by_phase_mae_candidate_v12": (by_phase / phase_counts[:, None].clamp_min(1)).tolist(),
            "phase_counts": phase_counts.int().tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("S0-fit", "S1", "S2"), required=True)
    parser.add_argument("--variant", choices=("R0", "R1", "R2", *FUSION_VARIANTS), default="R1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, choices=(32, 48, 64), default=32)
    parser.add_argument("--context", choices=("compact", "extended"), default="compact")
    parser.add_argument("--gradient-weight", type=float, choices=(0.1, 0.2, 0.3), default=0.1)
    parser.add_argument("--band-weight", type=float, choices=(0., 0.15), default=0.)
    parser.add_argument("--display-weight", type=float, choices=(0., 0.1), default=0.)
    parser.add_argument("--paired-order", action="store_true")
    parser.add_argument("--random-pair-supervision", action="store_true")
    parser.add_argument("--temporal-weight", type=float, choices=(0., 0.05), default=0.)
    parser.add_argument("--lr", type=float, choices=(1e-4, 5e-5, 2e-5), default=1e-4)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--native-checkpoint", type=Path)
    parser.add_argument("--fade-cache", type=Path)
    parser.add_argument("--deadline", type=datetime.fromisoformat)
    args = parser.parse_args()
    if args.stage == "S0-fit" and args.variant != "R1":
        parser.error("S0-fit uses R1")
    if args.warm_start and args.stage != "S2":
        parser.error("warm-start requires S2")
    if args.fade_cache and args.stage != "S2":
        parser.error("fade-cache requires S2")
    if args.paired_order and args.random_pair_supervision:
        parser.error("choose paired order or random companion supervision")
    if args.deadline and (args.stage != "S2" or args.deadline.utcoffset() is None):
        parser.error("deadline requires S2 and a UTC offset")
    if (args.native_checkpoint and args.variant not in NATIVE_VARIANTS
            or args.variant in NATIVE_VARIANTS and not (args.native_checkpoint or args.warm_start)):
        parser.error("clean fusion requires native-checkpoint or warm-start")
    out = args.out
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    provenance = fingerprints(args.db, args.scene_plan)
    stage = "S1" if args.stage == "S0-fit" else args.stage
    cache_info = {}

    def load(role):
        path = args.cache / f"{stage}-{role}.pt"
        saved = torch.load(path, weights_only=True)
        if saved["provenance"] != provenance or saved["split"] != f"{stage}-{role}":
            raise ValueError(f"incompatible cache {path}")
        cache_info[role] = {"path": str(path.resolve()), "sha256": digest(path), "source": saved["source"]}
        return saved["features"], labels(args.db, saved["manifest"]), saved["manifest"]

    train_x, train_y, keys = load("train")
    fade_info = None
    replaced = []
    if args.fade_cache:
        synthetic = torch.load(args.fade_cache, map_location="cpu", weights_only=True)
        indices = synthetic["indices"]
        if (synthetic["provenance"] != provenance or synthetic["train_count"] != len(train_x)
                or not torch.equal(torch.tensor([(keys[i]["frame"], keys[i]["block"]) for i in indices]),
                                   torch.tensor([(row["frame"], row["block"]) for row in synthetic["manifest"]]))):
            raise ValueError("synthetic fades do not match the training keys")
        train_x[indices] = synthetic["features"]
        train_y[indices] = synthetic["targets"]
        replaced = indices.tolist()
        fade_info = {"path": str(args.fade_cache.resolve()), "sha256": digest(args.fade_cache),
                     "samples": len(indices)}
    diagnostic = args.stage == "S0-fit"
    if diagnostic:
        subset = torch.arange(32) * len(keys) // 32
        train_x, train_y = train_x[subset], train_y[subset]
        keys = [keys[i] for i in subset]
        dev_x, dev_y = train_x, train_y
    else:
        dev_x, dev_y, _ = load("dev")
    dev_x, dev_y = dev_x.to(device), dev_y.to(device)
    cache_bytes = train_x.numel() + train_y.numel()
    if device.type == "cuda" and cache_bytes <= torch.cuda.mem_get_info(device)[0] // 2:
        train_x, train_y = train_x.to(device), train_y.to(device)
    epochs = 100 if diagnostic else 40 if stage == "S1" else 20
    milestones = (10, 20, 40) if stage == "S1" else (5, 10, 20)
    if diagnostic:
        milestones = (100,)
    config = {"stage": args.stage, "variant": args.variant, "seed": args.seed, "epochs": epochs,
              "device": str(device), "precision": "FP32", "lr": args.lr, "betas": [0.9, 0.999],
              "weight_decay": 0, "batch_size": 16, "microbatch": 4, "clip_norm": 1,
              "architecture": f"v13-{args.variant}", "width": args.width, "context": args.context,
              "gradient_weight": args.gradient_weight,
              "band_weight": args.band_weight, "display_gradient_weight": args.display_weight,
              "history": 16, "train_cache_device": str(train_x.device),
              "train_samples": len(train_x), "validation_samples": len(dev_x), "cache": cache_info,
              "fingerprints": provenance, "fit_keys": keys if diagnostic else None}
    paired = args.paired_order or (args.temporal_weight > 0 and not args.random_pair_supervision)
    if paired:
        pairs, singles = adjacent_groups(keys, replaced)
        first_in_pair = torch.zeros(len(keys), dtype=torch.bool, device=train_x.device)
        first_in_pair[pairs[:, 0].to(train_x.device)] = True
        config["temporal_pairs"] = {"pairs_per_epoch": len(pairs), "singles_per_epoch": len(singles),
                                    "weight": args.temporal_weight}
    if args.random_pair_supervision:
        prev = previous_indices(keys, replaced).to(train_x.device)
        config["temporal_companion"] = {"eligible_keys": int((prev >= 0).sum()),
                                        "weight": args.temporal_weight, "per_microbatch": 1}
    guided = args.variant == GUIDED_VARIANT
    if guided:
        config["fusion_guidance"] = {"weight": GUIDE_WEIGHT, "temperature": GUIDE_TEMPERATURE}
    if args.deadline:
        config["deadline"] = args.deadline.isoformat()
    if fade_info:
        config["fade_cache"] = fade_info
    model = ResidualCNN(args.variant, args.width, args.context).to(device, memory_format=torch.channels_last)
    if args.warm_start:
        initial = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        settings = initial["config"]
        original = (settings["variant"], settings.get("width", 32), settings.get("context", "compact"),
                    settings["fingerprints"])
        if original != (args.variant, args.width, args.context, provenance):
            raise ValueError("warm-start checkpoint architecture or provenance mismatch")
        model.load_state_dict(initial["model"])
        config["warm_start"] = {"sha256": digest(args.warm_start), "epoch": initial["epoch"],
                                "path": str(args.warm_start.resolve()), "optimizer_reset": True}
        if "native_checkpoint" in settings:
            config["native_checkpoint"] = settings["native_checkpoint"]
    if args.native_checkpoint:
        native = torch.load(args.native_checkpoint, map_location="cpu", weights_only=True)
        if native["config"]["db_sha256"] != provenance["db"]:
            raise ValueError("native checkpoint belongs to a different database")
        model.fusion.native.load_state_dict(native["model"])
        config["native_checkpoint"] = {"sha256": digest(args.native_checkpoint), "epoch": native["epoch"],
                                       "path": str(args.native_checkpoint.resolve())}
    write_json(out / "config.json", config)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0)
    order = torch.Generator().manual_seed(args.seed)
    best, updates = float("inf"), 0
    metrics_path = out / "metrics.jsonl"
    completed_epochs = 0
    for epoch in range(1, epochs + 1):
        if args.deadline and datetime.now().astimezone() >= args.deadline:
            write_json(out / "budget_status.json", {"status": "deadline", "completed_epochs": completed_epochs,
                                                   "planned_epochs": epochs, "updates": updates})
            break
        started = time.perf_counter()
        model.train()
        losses = torch.zeros((), device=device)
        if paired:
            pair_order = pairs[torch.randperm(len(pairs), generator=order)].flatten()
            single_order = singles[torch.randperm(len(singles), generator=order)]
            shuffled = torch.cat((pair_order, single_order)).to(train_x.device)
        else:
            shuffled = torch.randperm(len(train_x), generator=order).to(train_x.device)
        for keys in shuffled.split(16):
            optimizer.zero_grad(set_to_none=True)
            for micro in keys.split(4):
                inp = train_x[micro].to(device)
                target = train_y[micro].to(device).float() / 255
                if guided:
                    output, guidance = model(inp, fusion_target=target)
                    loss = loss_for(args.variant, output, target, args.gradient_weight, args.band_weight)
                    loss = loss + GUIDE_WEIGHT * guidance
                else:
                    output = model(inp)
                    loss = loss_for(args.variant, output, target, args.gradient_weight, args.band_weight)
                if args.display_weight:
                    display_error = demosaic(output - target)
                    display_gradient = display_error.diff(dim=-1).abs().mean() + display_error.diff(dim=-2).abs().mean()
                    loss = loss + args.display_weight * display_gradient
                if args.temporal_weight:
                    if args.random_pair_supervision:
                        previous = prev[micro[:1]]
                        prior_index = previous.clamp_min(0)
                        prior_input = train_x[prior_index].to(device)
                        prior_target = train_y[prior_index].to(device).float() / 255
                        delta = (output[:1] - target[:1]) - (model(prior_input) - prior_target)
                        temporal_loss = delta.abs().mean() * (previous >= 0).to(device).float().mean()
                    else:
                        pair_mask = first_in_pair[micro[:len(micro) // 2 * 2:2]].to(device)
                        temporal_loss = pair_loss(output, target, pair_mask)
                    loss = loss + args.temporal_weight * temporal_loss
                (loss * (len(micro) / len(keys))).backward()
                losses += loss.detach() * len(micro)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
        model.eval()
        if diagnostic:
            with torch.inference_mode():
                prediction = model.predict(dev_x).float()
                scores = {"mae_8bit": (prediction - dev_y).abs().mean().item(),
                          "v12_mae_8bit": (dev_x[:, :1].float() - dev_y).abs().mean().item()}
        else:
            scores = validation(model, dev_x, dev_y)
        result = {"epoch": epoch, "updates": updates, "loss": losses.item() / len(train_x),
                  "validation": scores, "seconds": time.perf_counter() - started}
        if not all(math.isfinite(number) for number in (result["loss"], scores["mae_8bit"])):
            raise ValueError("nonfinite training result")
        checkpoint = {"config": config, "epoch": epoch, "model": model.state_dict()}
        atomic_save(checkpoint, out / "last.pt")
        if scores["mae_8bit"] < best:
            best = scores["mae_8bit"]
            atomic_save(checkpoint, out / "best-mae.pt")
        if epoch in milestones:
            atomic_save(checkpoint, out / f"epoch-{epoch}.pt")
        with metrics_path.open("a", encoding="utf-8") as metrics:
            metrics.write(json.dumps(result, allow_nan=False) + "\n")
        completed_epochs = epoch
        update_active_record()
        if not diagnostic or epoch == 1 or epoch % 10 == 0:
            print(f"{args.stage} {args.variant} epoch={epoch} updates={updates} "
                  f"mae={scores['mae_8bit']:.4f} v12={scores['v12_mae_8bit']:.4f}", flush=True)


if __name__ == "__main__":
    run_logged("training", main)
