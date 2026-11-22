"""Train a small bytes-to-RAW baseline; see README.md for commands."""
import argparse
from contextlib import closing
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import DecodedDataset, RawDataset, connect
from network import build_model, resolve_v8_variant


def preview(path, prediction, target):
    # RAW mosaic, not demosaiced RGB. Top=target, bottom=prediction.
    pixels = torch.cat((target[0, 0], prediction[0, 0]), dim=0)
    Image.fromarray((pixels.clamp(0, 1).cpu().numpy()*255).round().astype(np.uint8)).save(path)


def run_epoch(model, loader, device, optimizer=None, max_steps=0, auxiliary_weight=1.0):
    model.train(optimizer is not None)
    absolute = squared = count = 0
    start = time.perf_counter()
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(loader, 1):
            tokens, phases, lengths, target = batch[:4]
            tokens, phases, lengths, target = [x.to(device) for x in (tokens, phases, lengths, target)]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if getattr(model, "requires_decoded", False):
                prediction = model(tokens, phases, lengths, batch[4].to(device))
                low_loss = 0
            elif len(batch) == 5:
                prediction, low = model(tokens, phases, lengths, return_low=True)
                low_loss = F.l1_loss(low, batch[4].to(device))
            else:
                prediction = model(tokens, phases, lengths)
                low_loss = 0
            loss = F.l1_loss(prediction, target)
            objective = loss+auxiliary_weight*low_loss
            if not torch.isfinite(objective):
                raise RuntimeError("non-finite loss")
            if optimizer is not None:
                objective.backward()
                optimizer.step()
            absolute += loss.item() * target.numel()
            squared += F.mse_loss(prediction, target).item() * target.numel()
            count += target.numel()
            if step % 50 == 0:
                print(f"  step={step} RAW_MAE={absolute/count*255:.3f}", flush=True)
            if max_steps and step >= max_steps:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"mae_8bit": absolute/count*255,
            "psnr_raw": -10*math.log10(max(squared/count, 1e-12)),
            "seconds": time.perf_counter()-start}, prediction.detach(), target.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--val-db", type=Path)
    parser.add_argument("--history", type=int, choices=(1, 2, 16), default=None,
                        help="total frames; default 1 for v9, 2 for earlier models")
    parser.add_argument("--architecture", choices=("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"), default="v9")
    parser.add_argument("--v9-layers", type=int, choices=(1, 2, 3, 4, 5), default=2,
                        help="v9 convolution count; layers beyond two add hidden-to-hidden convolutions")
    parser.add_argument("--v9-variant", choices=("plain4", "plain8", "plain16", "plain32", "relu4", "relu8", "relu16", "relu24", "relu32", "relu64", "relu128", "relu256"), default="relu16",
                        help="multi-layer v9: plain4/plain8/plain16/plain32=no activation; relu4/relu8/relu16/relu24/relu32/relu64/relu128/relu256=with ReLU")
    parser.add_argument("--v9-kernel-width", type=int, choices=(3, 9), default=3,
                        help="v9 kernel is 3 x width; 9 expands horizontal context within the current strip")
    parser.add_argument("--v9-upper-strip", action="store_true", help="add current-frame upper strip with validity mask")
    parser.add_argument("--v8-variant", choices=("expand96_two_layer", "legacy_four_layer"),
                        default="expand96_two_layer", help="v8 backbone; legacy option only for historical reproduction")
    parser.add_argument("--decoded-cache", type=Path)
    parser.add_argument("--val-decoded-cache", type=Path)
    parser.add_argument("--auxiliary-weight", type=float, default=1.0, help="legacy v2 only; v3-v9 use high-resolution RAW loss only")
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--warm-start", type=Path, help="initialize weights; optimizer starts fresh")
    parser.add_argument("--decay", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=0, help="per epoch; 0=full epoch")
    parser.add_argument("--val-steps", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "runs" / "baseline")
    args = parser.parse_args()
    if args.v9_upper_strip and args.architecture != "v9":
        parser.error("--v9-upper-strip requires v9")
    if args.history is None:
        args.history = 1 if args.architecture == "v9" else 2
    if args.architecture == "v9" and args.history not in (1, 2):
        parser.error("v9 uses current and optional previous frame; history must be 1 or 2")
    if args.residual_scale is None:
        args.residual_scale = (1.0 if args.architecture == "v9" else
                               0.5 if args.architecture in ("v3", "v4", "v5", "v6", "v7", "v8") else 0.1)
    if args.architecture == "v9" and args.residual_scale != 1:
        parser.error("v9 has an unscaled linear residual; residual-scale must be 1")
    if min(args.epochs, args.batch_size, args.lr) <= 0 or min(args.max_steps, args.val_steps) < 0:
        parser.error("epochs, batch-size and lr must be positive; steps must be nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if args.val_db:
        if args.db.resolve() == args.val_db.resolve():
            parser.error("validation DB must differ from training DB")
        train = RawDataset(args.db, args.history, v9_upper_strip=args.v9_upper_strip)
        validation = RawDataset(args.val_db, args.history, v9_upper_strip=args.v9_upper_strip)
        split = {"kind": "separate_databases", "note": "Sources must be different videos/scenes."}
    else:
        with closing(connect(args.db)) as db:
            frames = db.execute("SELECT DISTINCT frame_index FROM samples ORDER BY frame_index").fetchall()
        boundary = frames[int(len(frames)*0.8)][0]
        train = RawDataset(args.db, args.history, frame_stop=boundary, v9_upper_strip=args.v9_upper_strip)
        validation = RawDataset(args.db, args.history, frame_start=boundary, v9_upper_strip=args.v9_upper_strip)
        split = {"kind": "temporal_smoke_only", "boundary": boundary,
                 "note": "Same-video holdout does not establish unseen-video generalization."}
        print("WARNING: same-video temporal holdout; use --val-db for independent validation.", flush=True)
    if train.headers != validation.headers or len(train.headers) != 1:
        raise ValueError("training and validation must share one fixed codec/header configuration")
    if args.architecture in ("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"):
        from baseline import DecodedCache
        if args.decoded_cache is None or (args.val_db and args.val_decoded_cache is None):
            parser.error("v2-v9 require decoded caches for each database")
        if args.architecture == "v2" and args.auxiliary_weight <= 0:
            parser.error("auxiliary-weight must be positive")
        cache = DecodedCache(args.decoded_cache, args.db)
        val_cache = DecodedCache(args.val_decoded_cache, args.val_db) if args.val_db else cache
        train = DecodedDataset(train, cache)
        validation = DecodedDataset(validation, val_cache)
    model = build_model(vars(args)).to(device)
    if args.warm_start:
        prior = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        previous = prior["config"]
        if previous.get("v9_upper_strip", False) != args.v9_upper_strip:
            raise ValueError("warm-start requires the same upper-strip input contract")
        if previous.get("previous_block", False):
            raise ValueError("upper-strip checkpoints are retired; use the original v7 checkpoint")
        if args.architecture == "v8" and resolve_v8_variant(previous) != args.v8_variant:
            raise ValueError("warm-start requires the same v8 variant")
        if args.architecture == "v9" and previous.get("v9_layers", 2) != args.v9_layers:
            raise ValueError("warm-start requires the same v9 layer count")
        if args.architecture == "v9" and previous.get("v9_kernel_width", 3) != args.v9_kernel_width:
            raise ValueError("warm-start requires the same v9 kernel width")
        if (args.architecture == "v9" and args.v9_layers >= 2 and
                previous.get("v9_variant", "relu8") != args.v9_variant):
            raise ValueError("warm-start requires the same v9 variant")
        if (previous.get("architecture", "v1") != args.architecture or
                previous["history"] != args.history or previous["header_id"] not in train.headers or
                Path(previous["db"]).resolve() != args.db.resolve() or previous["split"] != split):
            raise ValueError("warm-start requires same architecture, history, header, source DB and split")
        model.load_state_dict(prior["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # num_workers=0 keeps read-only SQLite connections local on Windows.
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=0)
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("output already contains a run; select a new --out directory")
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "split": split, "torch": str(torch.__version__), "device_used": str(device),
              "header_id": next(iter(train.headers)), "train_samples": len(train),
              "validation_samples": len(validation),
              "parameters": sum(p.numel() for p in model.parameters())}
    config["input_contract"] = ("encoded_and_decoded_raw_v1" if args.architecture in ("v3", "v4", "v5", "v6", "v7", "v8")
                                else "encoded_only_legacy")
    config["residual_blocks"] = getattr(model, "residual_blocks", 0)
    config["residual_expansion"] = getattr(model, "residual_expansion", 1)
    config["residual_layers"] = getattr(model, "residual_layers", 2)
    if args.architecture == "v9":
        config.update(input_contract="decoded_raw_only_v1", byte_encoder=False,
                      temporal_fusion="phase_aligned_channel_concat" if args.history == 2 else False,
                      hidden_channels=model.refine[0].out_channels if args.v9_layers >= 2 else 0,
                      activation="relu" if args.v9_layers >= 2 and args.v9_variant in ("relu4", "relu8", "relu16", "relu24", "relu32", "relu64", "relu128", "relu256") else "none",
                      image_convolutions=args.v9_layers, residual_layers=args.v9_layers)
        config.update(kernel_size=[3, args.v9_kernel_width],
                      plane_receptive_field=[1+2*args.v9_layers, 1+(args.v9_kernel_width-1)*args.v9_layers])
        if args.v9_upper_strip:
            config.update(input_contract="decoded_raw_with_upper_strip_v1",
                          spatial_fusion="channel_concat_with_validity_mask", input_channels=model.refine[0].in_channels,
                          upper_boundary="zero_with_phase_minus_one_and_mask", input_order=["current_strip", "previous_frame_same_strip", "current_frame_upper_strip"] if args.history == 2 else ["current_strip", "current_frame_upper_strip"])
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(device), flush=True)
    best = float("inf")
    for epoch in range(1, args.epochs+1):
        training, _, _ = run_epoch(model, train_loader, device, optimizer, args.max_steps, args.auxiliary_weight)
        metrics, prediction, target = run_epoch(model, val_loader, device, max_steps=args.val_steps,
                                               auxiliary_weight=args.auxiliary_weight)
        record = {"epoch": epoch, "train": training, "validation": metrics}
        print(json.dumps(record), flush=True)
        with (args.out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record)+"\n")
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "config": config, "epoch": epoch}
        torch.save(checkpoint, args.out / "last.pt")
        if metrics["mae_8bit"] < best:
            best = metrics["mae_8bit"]
            torch.save(checkpoint, args.out / "best.pt")
        preview(args.out / f"epoch-{epoch:03d}-raw.png", prediction, target)
        from record_experiment import update_active_record
        update_active_record()


if __name__ == "__main__":
    from record_experiment import run_logged
    run_logged("training", main, Path(__file__).parent/"runs"/"baseline")
