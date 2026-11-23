"""Learn a ranked codebook of complete RAW strips by minimizing nearest-entry MAE."""
import argparse
import json
from pathlib import Path

import torch

from evaluate_v14 import REFERENCE
from evaluate_v14_strips import manifest, roundtrip
from record_experiment import run_logged, update_active_record, write_json
from v13.data import atomic_save, fingerprints, labels


def strip_distances(raw):
    strips = raw.transpose(0, 1).flatten(-2).float()
    # Integer-valued L1 sums fit exactly in float32 at 640x16 and 8 bits per sample.
    return torch.cdist(strips, strips, p=1).int()


def append_best(distances, nearest, used):
    losses = torch.minimum(distances, nearest[:, None, :]).sum(dim=-1)
    selected = losses.masked_fill(used, torch.iinfo(losses.dtype).max).argmin(dim=1)
    positions = torch.arange(distances.shape[0])
    used[positions, selected] = True
    return selected, torch.minimum(nearest, distances[positions, selected])


def score(distances, pixels):
    return {"mae_8bit": int(distances.sum()) / (distances.numel() * pixels),
            "exact_strips": int((distances == 0).sum()), "strips": distances.numel()}


def checkpoint(raw, ranking, config, path):
    indices = torch.stack(ranking)
    entries = raw[indices, torch.arange(raw.shape[1])]
    atomic_save({"raw": entries, "source_frames": indices + REFERENCE[0], "config": config}, path)
    return entries


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)
    provenance = fingerprints(args.db, args.scene_plan)
    scene = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    if scene["database_sha256"] != provenance["db"] or [f for f in scene["scene_starts"] if f >= REFERENCE[0]] != [REFERENCE[0]]:
        raise ValueError("expected the BBB scene plan paired with the input database")

    samples = manifest(*REFERENCE)
    raw = labels(args.db, samples).reshape(-1, 30, 16, 640)
    frames, positions, height, width = raw.shape
    pixels = height * width
    config = {"architecture": "v14-raw-codebook", "objective": "greedy minimum in-corpus nearest-strip L1",
              "data_column": "raw_bayer_target", "frame_range": REFERENCE, "strip_dimensions": [width, height],
              "training_samples": len(samples), "validation_samples": len(samples),
              "validation_scope": "same finite corpus used to train the codebook",
              "epochs": frames, "entries_per_position_at_end": frames, "provenance": provenance}
    write_json(args.out / "config.json", config)

    distances = strip_distances(raw)
    used = torch.zeros((positions, frames), dtype=torch.bool)
    nearest = torch.full((positions, frames), torch.iinfo(distances.dtype).max, dtype=distances.dtype)
    ranking = []
    best_epoch = None
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as metrics:
        for epoch in range(1, frames + 1):
            selected, nearest = append_best(distances, nearest, used)
            ranking.append(selected)
            uniform_ids = torch.linspace(0, frames - 1, epoch).round().long() if epoch > 1 else torch.tensor([frames // 2])
            uniform = distances[:, uniform_ids].amin(dim=1)
            learned = score(nearest, pixels)
            row = {"epoch": epoch, "entries_per_position": epoch, "entries_total": epoch * positions,
                   "validation": learned, "uniform_frame_baseline": score(uniform, pixels)}
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            update_active_record()
            if learned["exact_strips"] == len(samples) and best_epoch is None:
                best_epoch = epoch
            if epoch in (1, 4, 8, 16, 24, frames):
                print(f"{epoch} per position: RAW MAE {learned['mae_8bit']:.4f}, "
                      f"exact {learned['exact_strips']}/{len(samples)}", flush=True)

    if best_epoch is None or nearest.any():
        raise ValueError("the complete RAW codebook must cover every training strip exactly")
    entries = checkpoint(raw, ranking, config, args.out / "last.pt")
    control = roundtrip(entries, entries.short(), range(frames))
    if control["absolute_error"] or control["exact_strip_count"] != len(samples):
        raise ValueError("the trained complete codebook failed exact lookup")
    checkpoint(raw, ranking[:best_epoch], config, args.out / "best-mae.pt")
    write_json(args.out / "report.json", {"first_zero_error_epoch": best_epoch, "full_roundtrip": control})


if __name__ == "__main__":
    run_logged("training", main)
