"""Build a persistent v14 codebook for demo.mp4 using the validated recorded_phase_keys method.

Each of the 30 strip positions gets its own dictionary: key = (phase, 160x4 RAW8
sample) as observed when a frame was actually captured at that phase; value = the
complete 640x16 RAW8 strip at that position and frame. Duplicate keys (content
repeating under the same phase) collapse to the first occurrence, so the saved
codebook is automatically deduplicated.
"""
import argparse
from contextlib import closing
from pathlib import Path

import torch

from evaluate_v14_current_phase import phase_ids, sample
from evaluate_v14_strips import manifest
from record_experiment import digest, run_logged, write_json
from v11.data import connect
from v13.data import atomic_save, labels


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)

    with closing(connect(args.db)) as conn:
        first, last = conn.execute("SELECT min(frame_index), max(frame_index) FROM samples").fetchone()
    phases = phase_ids(args.db, first, last)
    strips_manifest = manifest(first, last)
    raw = labels(args.db, strips_manifest).reshape(-1, 30, 16, 640)
    frames, positions, height, width = raw.shape
    phase_tensor = torch.tensor(phases)

    position_tables, position_entries = [], []
    total_lookups = total_unique = 0
    for position in range(positions):
        references = raw[:, position]
        keys_by_phase = {phase: sample(references, phase) for phase in range(16)}
        index = {}
        order = []
        for frame in range(frames):
            phase = phases[frame]
            key = keys_by_phase[phase][frame].numpy().tobytes()
            full_key = (phase, key)
            if full_key not in index:
                index[full_key] = len(order)
                order.append(frame)
        unique_raw = references[order].clone()
        position_tables.append(index)
        position_entries.append(unique_raw)
        total_lookups += frames
        total_unique += len(order)

    book = {"position_tables": position_tables, "position_entries": position_entries,
            "frame_range": [first, last], "positions": positions,
            "key_format": "(phase:int, 160x4 RAW8 bytes); value: complete 640x16 RAW8 strip"}
    path = args.out / "codebook.pt"
    atomic_save(book, path)

    naive_bytes = total_lookups * height * width
    unique_bytes = total_unique * height * width
    key_bytes = total_unique * 640
    report = {
        "db_sha256": digest(args.db), "frame_range": [first, last], "frames": frames, "positions": positions,
        "total_strips": total_lookups, "unique_entries": total_unique,
        "dedup_ratio": 1 - total_unique / total_lookups,
        "naive_storage_bytes": naive_bytes, "deduped_value_bytes": unique_bytes,
        "key_storage_bytes": key_bytes, "deduped_total_bytes": unique_bytes + key_bytes,
        "codebook_path": str(path), "codebook_sha256": digest(path),
    }
    write_json(args.out / "report.json", report)
    print(f"strips: {total_lookups}, unique entries: {total_unique} "
          f"({report['dedup_ratio']*100:.2f}% deduplicated)", flush=True)
    print(f"naive storage: {naive_bytes/1e6:.2f} MB -> deduped (values+keys): "
          f"{report['deduped_total_bytes']/1e6:.2f} MB", flush=True)
    print(f"codebook saved to {path}", flush=True)


if __name__ == "__main__":
    run_logged("training", main)
