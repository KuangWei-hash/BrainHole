"""Isolate JPEG XS sample restoration using targets at the actual sampled phase."""
import argparse
from contextlib import closing
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from record_experiment import digest, run_logged, update_active_record, write_json
from v11.data import connect
from v11.phase import scatter_indices
from v13.data import atomic_save, labels, load_inputs
from v13.model import quantize
from v13.native import NativeRestorer


def sampled(db, first, last):
    keys = [{"frame": frame, "block": block} for frame in range(first, last + 1) for block in range(30)]
    full = labels(db, keys).reshape(last - first + 1, 30, 1, 16, 640)
    with closing(connect(db)) as conn:
        inputs = load_inputs(conn, first, last)
    grids = [scatter_indices(p) for p in range(16)]
    decoded, target = [], []
    for index, frame in enumerate(range(first, last + 1)):
        items = [inputs[frame, block] for block in range(30)]
        phase = {row[1] for row in items}
        if len(phase) != 1:
            raise ValueError(f"varied phase within frame {frame}")
        rows, cols = grids[phase.pop()]
        decoded.append(torch.stack([row[2] for row in items]))
        target.append(full[index][..., rows[:, None], cols])
    return torch.cat(decoded), torch.cat(target)


@torch.inference_mode()
def score(model, decoded, target):
    truth = target.float()
    predicted = torch.cat([quantize(model(x)) for x in decoded.split(128)]).float()
    error = predicted - truth
    baseline = decoded.float() - truth
    def gradients(residual):
        pack = F.pixel_unshuffle(residual, 2)
        return (pack.diff(dim=-1).abs().mean() + pack.diff(dim=-2).abs().mean()).item() / 2
    return {"mae_8bit": error.abs().mean().item(), "decoded_mae_8bit": baseline.abs().mean().item(),
            "psnr_raw": -10 * math.log10(max(error.square().mean().item() / 255**2, 1e-12)),
            "cfa_gradient_error": gradients(error), "decoded_gradient_error": gradients(baseline)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("db", "scene-plan", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    torch.set_num_threads(3)
    torch.manual_seed(42)
    torch.backends.cudnn.allow_tf32 = False
    db_hash = digest(args.db)
    scene = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    if scene["database_sha256"] != db_hash:
        raise ValueError("scene plan belongs to another database")
    train, truth = [x.to(args.device) for x in sampled(args.db, 0, 719)]
    validation, labels_dev = [x.to(args.device) for x in sampled(args.db, 736, 779)]
    model = NativeRestorer().to(args.device, memory_format=torch.channels_last)
    epochs = 10
    config = {"architecture": "v13-native-cfa", "history": 1, "train_frames": [0, 719],
              "validation_frames": [736, 779], "train_samples": len(train), "validation_samples": len(validation),
              "db_sha256": db_hash, "scene_sha256": digest(args.scene_plan), "epochs": epochs,
              "batch_size": 64, "seed": 42, "lr": 0.001, "device": args.device}
    write_json(args.out / "config.json", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    order = torch.Generator().manual_seed(42)
    best = float("inf")
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total = 0.
        for idx in torch.randperm(len(train), generator=order).to(args.device).split(64):
            loss = F.l1_loss(model(train[idx]), truth[idx].float() / 255)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(idx)
        model.eval()
        numbers = score(model, validation, labels_dev)
        checkpoint = {"config": config, "epoch": epoch, "model": model.state_dict()}
        atomic_save(checkpoint, args.out / "last.pt")
        if numbers["mae_8bit"] < best:
            best = numbers["mae_8bit"]
            atomic_save(checkpoint, args.out / "best-mae.pt")
        with (args.out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch, "loss": total / len(train), "validation": numbers,
                                    "seconds": time.perf_counter() - started}, allow_nan=False) + "\n")
        update_active_record()
        print(epoch, numbers, flush=True)


if __name__ == "__main__":
    run_logged("training", main)
