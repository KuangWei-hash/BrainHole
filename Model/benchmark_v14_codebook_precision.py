"""Measure how reducing RAW prototype precision trades storage for reconstruction error."""
import argparse
from pathlib import Path

import torch

from evaluate_v14 import REFERENCE
from evaluate_v14_strips import manifest
from record_experiment import digest, run_logged, write_json
from train_v14_lossy import assign, measurements, storage
from v13.data import fingerprints, labels


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    args = parser.parse_args()
    torch.set_num_threads(4)
    provenance = fingerprints(args.db, args.scene_plan)
    samples = manifest(*REFERENCE)
    strips = labels(args.db, samples).reshape(-1, 30, 16, 640).transpose(0, 1).flatten(-2).float()
    report = {"frame_range": REFERENCE, "samples": len(samples), "provenance": provenance, "codebooks": []}
    for path in args.checkpoint:
        book = torch.load(path, map_location="cpu", weights_only=True)
        config = book["config"]
        if config["provenance"] != provenance or tuple(config["frame_range"]) != REFERENCE:
            raise ValueError(f"codebook does not match the source data: {path}")
        prototypes = book["raw"].transpose(0, 1).int()
        anchors = book["anchor_frames"].transpose(0, 1)
        rows = []
        for bits in (8, 7, 6, 5, 4):
            levels = (1 << bits) - 1
            codes = ((prototypes * levels + 127) // 255).byte()
            decoded = (codes.int() * 255 + levels // 2) // levels
            distance, _ = assign(strips, decoded)
            frame_mae = distance.sum(dim=0).double() / (strips.shape[0] * strips.shape[-1])
            packed = storage(codes, anchors)
            row = {"bits_per_pixel": bits, "bit_packed_bytes": (codes.numel() * bits + 7) // 8,
                   "xz_payload_bytes": packed["xz_payload_bytes"],
                   **measurements(distance, strips.shape[-1]), "max_frame_mae": frame_mae.max().item()}
            rows.append(row)
            print(f"K={prototypes.shape[1]}, {bits}-bit: {row['xz_payload_bytes']:,} bytes, "
                  f"RAW MAE {row['mae_8bit']:.4f}", flush=True)
        report["codebooks"].append({"checkpoint": str(path), "sha256": digest(path),
                                     "entries_per_position": prototypes.shape[1], "measurements": rows})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)


if __name__ == "__main__":
    run_logged("benchmark", main)
