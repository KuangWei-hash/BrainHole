"""Verify exact whole-video lookup using current-phase RAW8 keys from a finite source."""
import argparse
from contextlib import closing
from pathlib import Path

import torch

from evaluate_v14_current_phase import phase_ids, sample
from evaluate_v14_strips import manifest
from record_experiment import digest, run_logged, write_json
from v11.data import connect
from v13.data import labels


def earliest_by_key(keys, candidates):
    index = {}
    for i in candidates:
        index.setdefault(keys[i], i)
    return index


def check(raw, phases):
    frames, positions, height, width = raw.shape
    expected = frames * positions
    phase_frames = [[i for i, frame_phase in enumerate(phases) if frame_phase == phase] for phase in range(16)]
    modes = ("recorded_phase_keys", "all_reference_phases")
    matched = {name: torch.ones(frames, dtype=torch.bool) for name in modes}
    exact = dict.fromkeys(modes, 0)
    absolute_error = dict.fromkeys(modes, 0)
    for position in range(positions):
        references = raw[:, position]
        for phase, queries in enumerate(phase_frames):
            if not queries:
                continue
            keys = [row.tobytes() for row in sample(references, phase).numpy()]
            for name, candidates in zip(modes, (queries, range(frames))):
                table = earliest_by_key(keys, candidates)
                found = [table[keys[i]] for i in queries]
                difference = (references[found].short() - references[queries].short()).abs().flatten(1).sum(dim=-1)
                perfect = difference == 0
                matched[name][queries] &= perfect
                exact[name] += int(perfect.sum())
                absolute_error[name] += int(difference.sum())
    return {name: {"frames_exact": int(matched[name].sum()), "strips_exact": exact[name],
                   "strips_with_nonzero_error": expected - exact[name], "absolute_error": absolute_error[name],
                   "raw_mae": absolute_error[name] / (expected * height * width)} for name in modes}


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
    if first is None or last is None:
        raise ValueError("no source frames")
    phases = phase_ids(args.db, first, last)
    strips = manifest(first, last)
    raw = labels(args.db, strips).reshape(-1, 30, 16, 640)
    results = check(raw, phases)
    report = {"start_frame": first, "first_evaluated_frame": first, "last_evaluated_frame": last,
              "scope": "whole source video in-codebook roundtrip", "db_sha256": digest(args.db),
              "frames": len(phases), "samples": len(strips), "raw_bytes_checked_per_mode": raw.numel(),
              "key": "position, current phase and 640 sampled bytes", "tie_breaker": "earliest reference frame",
              "modes": results}
    write_json(args.out / "report.json", report)
    for name, result in results.items():
        print(f"{name}: {result['frames_exact']}/{len(phases)} frames, "
              f"{result['strips_exact']}/{len(strips)} strips, RAW MAE {result['raw_mae']:.6f}", flush=True)


if __name__ == "__main__":
    run_logged("evaluation", main)
