"""Train position-specific median prototypes for complete RAW8 strips."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_v14_codebook_storage import CODECS, encode
from evaluate_v13 import display
from evaluate_v14 import REFERENCE, export
from evaluate_v14_strips import manifest
from record_experiment import digest, run_logged, update_active_record, write_json
from train_v14_codebook import score
from v13.data import atomic_save, fingerprints, labels


CAPACITIES = (4, 8, 16)
UPDATES = 8


def assign(strips, prototypes):
    distance = torch.cdist(strips, prototypes.flatten(-2).float(), p=1).int()
    return distance.min(dim=-1)


def update_prototypes(raw, prototypes, indices):
    updated = prototypes.clone()
    for position in range(len(prototypes)):
        for index in range(prototypes.shape[1]):
            members = raw[position, indices[position] == index]
            if len(members):
                updated[position, index] = members.median(dim=0).values
    return updated


def measurements(distance, pixels):
    strip_mae = distance.flatten().float() / pixels
    return {**score(distance, pixels), "strip_mae_p95": strip_mae.quantile(0.95).item(),
            "strip_mae_max": strip_mae.max().item()}


def storage(prototypes, anchor_frames):
    order = np.argsort(anchor_frames.numpy(), axis=1, kind="stable")
    chronological = np.take_along_axis(prototypes.numpy(), order[..., None, None], axis=1)
    deltas = encode(chronological, "temporal")
    pack, _ = CODECS["xz-6"]
    return {"raw_bytes": prototypes.numel(),
            "xz_payload_bytes": sum(len(pack(position.tobytes())) for position in deltas)}


def distortion_and_storage(raw, prototypes, distance, indices, anchors):
    pred = prototypes[torch.arange(len(prototypes))[:, None], indices]
    error = (pred.short() - raw.short()).transpose(0, 1)
    if not torch.equal(error.abs().sum(dim=(-2, -1)).transpose(0, 1), distance.long()):
        raise ValueError("nearest-distance scores differ from reconstructed RAW")
    numbers = {**measurements(distance, raw.shape[-2] * raw.shape[-1]), **storage(prototypes, anchors),
               "temporal_residual_difference": error.diff(dim=0).abs().float().mean().item()}
    return numbers, pred.transpose(0, 1).reshape(-1, 1, 16, 640)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="greedy exemplar codebook for initialization")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)
    provenance = fingerprints(args.db, args.scene_plan)
    initial = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if initial["config"]["provenance"] != provenance or tuple(initial["config"]["frame_range"]) != REFERENCE:
        raise ValueError("initialization codebook must match the RAW source and frame range")

    samples = manifest(*REFERENCE)
    truth = labels(args.db, samples)
    raw = truth.reshape(-1, 30, 16, 640).transpose(0, 1).contiguous()
    strips = raw.flatten(-2).float()
    pixels = strips.shape[-1]
    config = {"architecture": "v14-raw-k-medians", "frame_range": REFERENCE, "strip_dimensions": [640, 16],
              "capacities": CAPACITIES, "updates_per_capacity": UPDATES, "epochs": UPDATES * len(CAPACITIES),
              "training_samples": len(samples), "validation_samples": len(samples),
              "validation_scope": "same finite corpus used to train the codebook", "provenance": provenance,
              "initialization_sha256": digest(args.checkpoint)}
    write_json(args.out / "config.json", config)
    export(args.out, "target", display(truth), REFERENCE[0])

    report = {"training_range": REFERENCE, "samples": len(samples), "capacities": {}}
    best_mae = float("inf")
    epoch = 0
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as metrics:
        for count in CAPACITIES:
            prototypes = initial["raw"][:count].transpose(0, 1).contiguous()
            anchors = initial["source_frames"][:count].transpose(0, 1).contiguous()
            distance, indices = assign(strips, prototypes)
            baseline, _ = distortion_and_storage(raw, prototypes, distance, indices, anchors)
            for step in range(1, UPDATES + 1):
                prototypes = update_prototypes(raw, prototypes, indices)
                updated, indices = assign(strips, prototypes)
                if updated.sum() > distance.sum():
                    raise ValueError("median update increased total RAW error")
                distance = updated
                epoch += 1
                row = {"epoch": epoch, "iteration": step, "entries_per_position": count,
                       "validation": measurements(distance, pixels)}
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                update_active_record()

            result, predicted = distortion_and_storage(raw, prototypes, distance, indices, anchors)
            checkpoint = {"raw": prototypes.transpose(0, 1).contiguous(),
                          "anchor_frames": anchors.transpose(0, 1).contiguous(), "config": config}
            atomic_save(checkpoint, args.out / f"k{count:02d}.pt")
            atomic_save(checkpoint, args.out / "last.pt")
            if result["mae_8bit"] < best_mae:
                best_mae = result["mae_8bit"]
                atomic_save(checkpoint, args.out / "best-mae.pt")
            report["capacities"][str(count)] = {"exemplars": baseline, "trained_medians": result}
            write_json(args.out / "report.json", report)
            export(args.out, f"k{count:02d}", display(predicted), REFERENCE[0])
            print(f"K={count}: exemplars {baseline['mae_8bit']:.4f}, trained {result['mae_8bit']:.4f}, "
                  f"P95 {result['strip_mae_p95']:.4f}, XZ payload {result['xz_payload_bytes']:,} bytes", flush=True)


if __name__ == "__main__":
    run_logged("training", main)
