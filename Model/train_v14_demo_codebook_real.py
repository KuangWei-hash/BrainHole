"""Build and save the real-signal v14 codebook: key = (position, actual phase,
JPEG XS encoded payload bytes); value = the complete 640x16 RAW8 strip. Same
recorded_phase_keys rule as train_v14_demo_codebook.py, but keyed on what a
receiver can actually have (the encoded payload), not a clean RAW sample.
"""
import argparse
from contextlib import closing
from pathlib import Path

import torch

from evaluate_v14_current_phase import phase_ids
from evaluate_v14_strips import manifest
from record_experiment import digest, run_logged, write_json
from v11.data import connect
from v13.data import atomic_save, labels


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--column", choices=("jpegxs_payload", "decoded"), default="jpegxs_payload")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)

    with closing(connect(args.db)) as conn:
        first, last = conn.execute("SELECT min(frame_index), max(frame_index) FROM samples").fetchone()
        rows = conn.execute(f"SELECT frame_index, block_index, {args.column} FROM samples "
                            "WHERE frame_index BETWEEN ? AND ?", (first, last)).fetchall()
    phases = phase_ids(args.db, first, last)
    strips_manifest = manifest(first, last)
    raw = labels(args.db, strips_manifest).reshape(-1, 30, 16, 640)
    frames, positions, height, width = raw.shape

    payloads_by_position = [[None] * frames for _ in range(30)]
    for frame, block, payload in rows:
        payloads_by_position[block][frame - first] = payload
    if any(p is None for column in payloads_by_position for p in column):
        raise ValueError("missing signal for some strip")

    position_tables, position_entries = [], []
    total_lookups = total_unique = key_bytes_total = 0
    for position in range(positions):
        references = raw[:, position]
        keys_all = payloads_by_position[position]
        index, order = {}, []
        for frame in range(frames):
            full_key = (phases[frame], keys_all[frame])
            if full_key not in index:
                index[full_key] = len(order)
                order.append(frame)
        unique_raw = references[order].clone()
        position_tables.append(index)
        position_entries.append(unique_raw)
        total_lookups += frames
        total_unique += len(order)
        key_bytes_total += sum(len(keys_all[frame]) for frame in order)

    book = {"position_tables": position_tables, "position_entries": position_entries,
            "frame_range": [first, last], "positions": positions, "key_column": args.column,
            "key_format": f"(phase:int, {args.column} bytes); value: complete 640x16 RAW8 strip"}
    path = args.out / "codebook.pt"
    atomic_save(book, path)

    value_bytes = total_unique * height * width
    report = {
        "db_sha256": digest(args.db), "frame_range": [first, last], "frames": frames, "positions": positions,
        "key_column": args.column, "total_strips": total_lookups, "unique_entries": total_unique,
        "dedup_ratio": 1 - total_unique / total_lookups,
        "value_bytes": value_bytes, "key_bytes": key_bytes_total, "logical_total_bytes": value_bytes + key_bytes_total,
        "codebook_path": str(path), "codebook_on_disk_bytes": path.stat().st_size, "codebook_sha256": digest(path),
    }
    write_json(args.out / "report.json", report)
    print(f"key column: {args.column}; strips: {total_lookups}, unique entries: {total_unique} "
          f"({report['dedup_ratio']*100:.2f}% deduplicated)", flush=True)
    print(f"value bytes: {value_bytes/1e6:.2f} MB, key bytes: {key_bytes_total/1e6:.2f} MB, "
          f"logical total: {report['logical_total_bytes']/1e6:.2f} MB", flush=True)
    print(f"codebook saved to {path}: {path.stat().st_size/1e6:.2f} MB on disk", flush=True)


if __name__ == "__main__":
    run_logged("training", main)
