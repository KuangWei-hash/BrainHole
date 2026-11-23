"""Same whole-video exact-lookup check as check_v14_exact_video.py, but keyed on
the actual JPEG XS encoded payload bytes (jpegxs_payload) instead of a RAW pixel
sample. Encoding is deterministic, so two identical payloads must decode to
identical pixels; the question is whether using the much shorter encoded bytes
as the index changes anything versus the 640-byte decoded-sample key, and how
much smaller the key storage becomes.
"""
import argparse
from contextlib import closing
from pathlib import Path

import torch

from evaluate_v14_current_phase import phase_ids
from evaluate_v14_strips import manifest
from record_experiment import digest, run_logged, write_json
from v11.data import connect
from v13.data import labels


def earliest_by_key(keys, candidates):
    index = {}
    for i in candidates:
        index.setdefault(keys[i], i)
    return index


def check(raw, phases, payloads_by_position):
    frames, positions, height, width = raw.shape
    expected = frames * positions
    phase_frames = [[i for i, p in enumerate(phases) if p == phase] for phase in range(16)]
    modes = ("recorded_phase_keys", "all_reference_phases")
    matched = {name: torch.ones(frames, dtype=torch.bool) for name in modes}
    exact = dict.fromkeys(modes, 0)
    absolute_error = dict.fromkeys(modes, 0)
    key_bytes_total = payload_bytes_total = unique_entries_total = 0
    for position in range(positions):
        references = raw[:, position]
        keys_all = payloads_by_position[position]
        payload_bytes_total += sum(len(k) for k in keys_all)
        for phase, queries in enumerate(phase_frames):
            if not queries:
                continue
            for name, candidates in zip(modes, (queries, range(frames))):
                table = earliest_by_key(keys_all, candidates)
                if name == "recorded_phase_keys":
                    unique_entries_total += len(table)
                    key_bytes_total += sum(len(keys_all[i]) for i in set(table.values()))
                found = [table[keys_all[i]] for i in queries]
                difference = (references[found].short() - references[queries].short()).abs().flatten(1).sum(dim=-1)
                perfect = difference == 0
                matched[name][queries] &= perfect
                exact[name] += int(perfect.sum())
                absolute_error[name] += int(difference.sum())
    return ({name: {"frames_exact": int(matched[name].sum()), "strips_exact": exact[name],
                    "strips_with_nonzero_error": expected - exact[name], "absolute_error": absolute_error[name],
                    "raw_mae": absolute_error[name] / (expected * height * width)} for name in modes},
            payload_bytes_total, key_bytes_total, unique_entries_total)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--column", choices=("jpegxs_payload", "decoded"), default="jpegxs_payload",
                        help="which real (lossy) per-block signal to key on")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)

    with closing(connect(args.db)) as conn:
        first, last = conn.execute("SELECT min(frame_index), max(frame_index) FROM samples").fetchone()
        rows = conn.execute(f"SELECT frame_index, block_index, {args.column} FROM samples "
                            "WHERE frame_index BETWEEN ? AND ?", (first, last)).fetchall()
    if first is None or last is None:
        raise ValueError("no source frames")
    phases = phase_ids(args.db, first, last)
    strips_manifest = manifest(first, last)
    raw = labels(args.db, strips_manifest).reshape(-1, 30, 16, 640)
    frames = raw.shape[0]

    payloads_by_position = [[None] * frames for _ in range(30)]
    for frame, block, payload in rows:
        payloads_by_position[block][frame - first] = payload
    if any(p is None for column in payloads_by_position for p in column):
        raise ValueError("missing jpegxs_payload for some sample")

    results, payload_bytes_total, deduped_key_bytes, unique_entries = check(raw, phases, payloads_by_position)
    fixed_key_bytes_equivalent = unique_entries * 640
    report = {
        "start_frame": first, "first_evaluated_frame": first, "last_evaluated_frame": last,
        "scope": "whole source video in-codebook roundtrip, keyed on the JPEG XS encoded payload",
        "db_sha256": digest(args.db), "frames": len(phases), "samples": len(strips_manifest),
        "key": "position, current phase and the variable-length JPEG XS payload bytes",
        "tie_breaker": "earliest reference frame",
        "mean_payload_bytes": payload_bytes_total / (frames * 30),
        "recorded_phase_keys_unique_entries": unique_entries,
        "deduped_key_storage_bytes_encoded": deduped_key_bytes,
        "deduped_key_storage_bytes_if_fixed_640byte_decoded_key": fixed_key_bytes_equivalent,
        "modes": results,
    }
    write_json(args.out / "report.json", report)
    for name, result in results.items():
        print(f"{name}: {result['frames_exact']}/{len(phases)} frames, "
              f"{result['strips_exact']}/{len(strips_manifest)} strips, RAW MAE {result['raw_mae']:.6f}", flush=True)
    print(f"mean JPEG XS payload size: {payload_bytes_total/(frames*30):.2f} bytes "
          f"(vs fixed 640 bytes for a decoded sample)", flush=True)
    print(f"deduped key storage: encoded payload = {deduped_key_bytes/1e6:.3f} MB, "
          f"vs fixed 640-byte decoded key = {fixed_key_bytes_equivalent/1e6:.3f} MB "
          f"({unique_entries} unique entries)", flush=True)


if __name__ == "__main__":
    run_logged("evaluation", main)
