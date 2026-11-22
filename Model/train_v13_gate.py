"""Fit a small convex gate over two frozen v13 residual outputs."""
import argparse
import json
import math
from pathlib import Path

import torch

from evaluate_v13 import raw_predictions
from record_experiment import digest, run_logged, update_active_record, write_json
from v13.data import atomic_save, fingerprints, labels
from v13.ensemble import EnsembleGate
from v13.model import loss_for, quantize


@torch.inference_mode()
def assess(model, tensors, device):
    errors, base_errors = [], []
    for group in zip(*(x.split(24) for x in tensors)):
        x, trend, spatial, target = [item.to(device) for item in group]
        error = quantize(model(x, trend, spatial)).float() - target
        baseline = quantize((trend + spatial) / 2).float() - target
        errors.append(error.cpu())
        base_errors.append(baseline.cpu())
    error = torch.cat(errors)
    baseline = torch.cat(base_errors)
    return {"mae_8bit": error.abs().mean().item(), "uniform_mae_8bit": baseline.abs().mean().item(),
            "psnr_raw": -10 * math.log10(max(error.square().mean().item() / 255**2, 1e-12))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("db", "scene-plan", "cache", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, nargs=2, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    torch.set_num_threads(3)
    torch.manual_seed(42)
    torch.backends.cudnn.allow_tf32 = False
    provenance = fingerprints(args.db, args.scene_plan)
    inputs = []
    histories = {}
    for role in ("train", "dev"):
        cache = torch.load(args.cache / f"S2-{role}.pt", weights_only=True)
        if cache["split"] != f"S2-{role}" or cache["provenance"] != provenance:
            raise ValueError("training cache provenance mismatch")
        measurements = cache["features"]
        target = labels(args.db, cache["manifest"])
        outputs = []
        for path in args.checkpoint:
            values, info = raw_predictions(path, measurements, args.device, provenance)
            # Inference tensors need ordinary CPU storage for the gate's backward pass.
            outputs.append(values.clone())
            if role == "train":
                inputs.append(info)
        histories[role] = (measurements, *outputs, target)
        print(role, len(measurements), "frozen predictions ready", flush=True)
    del cache
    model = EnsembleGate().to(args.device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    order = torch.Generator().manual_seed(42)
    epochs, batch = 12, 32
    config = {"architecture": "v13-ensemble-gate", "history": 16, "seed": 42, "epochs": epochs,
              "device": args.device, "precision": "FP32",
              "train_samples": len(histories["train"][0]), "validation_samples": len(histories["dev"][0]),
              "batch_size": batch, "lr": 0.001, "fingerprints": provenance, "models": inputs,
              "cache_sha256": {key: digest(args.cache / f"S2-{key}.pt") for key in histories}}
    write_json(args.out / "config.json", config)
    minimum = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for idx in torch.randperm(config["train_samples"], generator=order).split(batch):
            x, trend, spatial, target = [tensor[idx].to(args.device) for tensor in histories["train"]]
            loss = loss_for("R2", model(x, trend, spatial), target.float() / 255)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(idx)
        model.eval()
        numbers = assess(model, histories["dev"], args.device)
        result = {"epoch": epoch, "validation": numbers, "loss": total / config["train_samples"]}
        checkpoint = {"config": config, "epoch": epoch, "model": model.state_dict()}
        atomic_save(checkpoint, args.out / "last.pt")
        if numbers["mae_8bit"] < minimum:
            minimum = numbers["mae_8bit"]
            atomic_save(checkpoint, args.out / "best-mae.pt")
        with (args.out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
        update_active_record()
        print(epoch, numbers, flush=True)


if __name__ == "__main__":
    run_logged("training", main)
