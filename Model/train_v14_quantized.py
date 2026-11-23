"""Refine a fixed-size whole-strip codebook directly on its final RAW value grid."""
import argparse
import json
from pathlib import Path

import torch

from evaluate_v14 import REFERENCE
from evaluate_v14_strips import manifest
from record_experiment import digest, run_logged, update_active_record, write_json
from train_v14_lossy import assign, measurements, storage
from v13.data import atomic_save, fingerprints, labels


def refine(raw, codes, groups, levels):
    updated = codes.clone()
    for position in range(len(codes)):
        for index in range(codes.shape[1]):
            members = raw[position, groups[position] == index]
            if not len(members):
                continue
            median = members.median(dim=0).values
            upper = torch.searchsorted(levels, median)
            lower = (upper - 1).clamp_min(0)
            lower_error = (members - levels[lower]).abs().sum(dim=0)
            upper_error = (members - levels[upper]).abs().sum(dim=0)
            best = torch.where(lower_error <= upper_error, lower, upper)
            original = codes[position, index].long()
            original_error = (members - levels[original]).abs().sum(dim=0)
            improved = torch.minimum(lower_error, upper_error) < original_error
            updated[position, index] = torch.where(improved, best, original).byte()
    return updated


def stats(distances, pixels, codes, anchors):
    return {**measurements(distances, pixels), "xz_payload_bytes": storage(codes, anchors)["xz_payload_bytes"]}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)
    provenance = fingerprints(args.db, args.scene_plan)
    initial = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if initial["config"]["provenance"] != provenance or tuple(initial["config"]["frame_range"]) != REFERENCE:
        raise ValueError("initial codebook does not match the RAW source")
    frames = manifest(*REFERENCE)
    raw = labels(args.db, frames).reshape(-1, 30, 16, 640).transpose(0, 1).short().contiguous()
    strips = raw.flatten(-2).float()
    pixels = strips.shape[-1]
    bits, rounds = 6, 32
    span = (1 << bits) - 1
    levels = ((torch.arange(span + 1) * 255 + span // 2) // span).short()
    prototypes = initial["raw"].transpose(0, 1).int()
    anchors = initial["anchor_frames"].transpose(0, 1)
    codes = ((prototypes * span + 127) // 255).byte()
    config = {"architecture": "v14-raw-quantized-k-medians", "frame_range": REFERENCE,
              "entries_per_position": codes.shape[1], "bits_per_pixel": bits, "epochs": rounds,
              "training_samples": len(frames), "validation_samples": len(frames),
              "validation_scope": "same finite corpus used to train the codebook", "provenance": provenance,
              "initialization_sha256": digest(args.checkpoint)}
    write_json(args.out / "config.json", config)

    distance, groups = assign(strips, levels[codes.long()])
    report = {"baseline": stats(distance, pixels, codes, anchors), "rounds": {}}
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as metrics:
        for epoch in range(1, rounds + 1):
            updated = refine(raw, codes, groups, levels)
            improved, groups = assign(strips, levels[updated.long()])
            if improved.sum() > distance.sum():
                raise ValueError("grid-aware training increased total RAW error")
            row = {"epoch": epoch, "validation": measurements(improved, pixels),
                   "updated_values": int((codes != updated).sum())}
            codes, distance = updated, improved
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            update_active_record()
            if epoch in (1, 8, rounds):
                report["rounds"][str(epoch)] = stats(distance, pixels, codes, anchors)
                print(f"round {epoch}: {report['rounds'][str(epoch)]}", flush=True)

    checkpoint = {"codes": codes.transpose(0, 1).contiguous(), "levels": levels.byte(),
                  "anchor_frames": initial["anchor_frames"], "config": config}
    atomic_save(checkpoint, args.out / "best-mae.pt")
    atomic_save(checkpoint, args.out / "last.pt")
    write_json(args.out / "report.json", report)


if __name__ == "__main__":
    run_logged("training", main)
