"""Same as export_v14_exact_lookup_video.py, but keyed on the real, lossy
per-block signal (the actual `decoded` JPEG XS reconstruction a receiver would
have) instead of a clean sample taken from the ground-truth RAW. This is the
honest version: it shows the recorded_phase_keys lookup's real achievable
accuracy (91.6% exact strips), not the idealized oracle-key result.
"""
import argparse
from contextlib import closing
from pathlib import Path

import torch

from evaluate import demosaic
from evaluate_v10 import SingleVideo
from evaluate_v13 import verify_video
from evaluate_v14_current_phase import phase_ids
from evaluate_v14_strips import manifest
from v11.data import connect
from v13.data import labels


def earliest_by_key(keys, candidates):
    index = {}
    for i in candidates:
        index.setdefault(keys[i], i)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--error-gain", type=float, default=10.0, help="multiplier for the error-visualization video")
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)

    with closing(connect(args.db)) as conn:
        first, last = conn.execute("SELECT min(frame_index), max(frame_index) FROM samples").fetchone()
        rows = conn.execute("SELECT frame_index, block_index, decoded FROM samples "
                            "WHERE frame_index BETWEEN ? AND ?", (first, last)).fetchall()
    phases = phase_ids(args.db, first, last)
    strips_manifest = manifest(first, last)
    raw = labels(args.db, strips_manifest).reshape(-1, 30, 16, 640)
    frames, positions, height, width = raw.shape

    payloads_by_position = [[None] * frames for _ in range(30)]
    for frame, block, payload in rows:
        payloads_by_position[block][frame - first] = payload
    if any(p is None for column in payloads_by_position for p in column):
        raise ValueError("missing decoded sample for some strip")

    phase_frames = [[i for i, p in enumerate(phases) if p == phase] for phase in range(16)]
    found = torch.arange(frames).unsqueeze(1).repeat(1, positions)
    mismatches = []
    for position in range(positions):
        references = raw[:, position]
        keys = payloads_by_position[position]
        for phase, queries in enumerate(phase_frames):
            if not queries:
                continue
            table = earliest_by_key(keys, queries)
            for i in queries:
                match = table[keys[i]]
                found[i, position] = match
                if match != i and not torch.equal(references[match], references[i]):
                    mismatches.append((i + first, position, match + first))

    print(f"content-wrong strips: {len(mismatches)}/{frames * positions}", flush=True)
    for frame, position, source in mismatches[:20]:
        print(f"  frame {frame} block {position} <- retrieved from frame {source}", flush=True)

    videos = {name: SingleVideo(args.out, name) for name in
              ("target", "recorded-phase-lookup-real", "error-x%g" % args.error_gain)}
    written = 0
    with torch.inference_mode():
        for i in range(frames):
            truth = raw[i]
            retrieved = raw[found[i], torch.arange(positions)]
            error = (truth.short() - retrieved.short()).abs().mul(args.error_gain).clamp(0, 255).byte()
            for name, value in (("target", truth), ("recorded-phase-lookup-real", retrieved),
                                 ("error-x%g" % args.error_gain, error)):
                rgb = demosaic(value.unsqueeze(1).float() / 255)
                videos[name].write(dict(enumerate(rgb)))
            written += 1
    for video in videos.values():
        video.close()
    for name in videos:
        verify_video(args.out / f"{name}.mp4", written)
    print(f"wrote {written} frames to {args.out}", flush=True)


if __name__ == "__main__":
    main()
