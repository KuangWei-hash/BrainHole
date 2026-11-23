"""Export target / recorded-phase-key lookup / amplified-error videos for the whole source video.

Reproduces check_v14_exact_video.py's "recorded_phase_keys" mode (each table entry
keyed only by the phase it was actually captured at) but also renders the result
as video, so the rare collision strips can be seen directly instead of only counted.
"""
import argparse
from contextlib import closing
from pathlib import Path

import torch

from evaluate import demosaic
from evaluate_v10 import SingleVideo
from evaluate_v13 import verify_video
from evaluate_v14_current_phase import phase_ids, sample
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
    phases = phase_ids(args.db, first, last)
    strips_manifest = manifest(first, last)
    raw = labels(args.db, strips_manifest).reshape(-1, 30, 16, 640)
    frames, positions, height, width = raw.shape

    phase_frames = [[i for i, p in enumerate(phases) if p == phase] for phase in range(16)]
    found = torch.arange(frames).unsqueeze(1).repeat(1, positions)
    mismatches = []
    for position in range(positions):
        references = raw[:, position]
        for phase, queries in enumerate(phase_frames):
            if not queries:
                continue
            keys = [row.tobytes() for row in sample(references, phase).numpy()]
            table = earliest_by_key(keys, queries)
            for i in queries:
                match = table[keys[i]]
                found[i, position] = match
                if match != i and not torch.equal(references[match], references[i]):
                    mismatches.append((i + first, position, match + first))

    print(f"content-wrong strips: {len(mismatches)}/{frames * positions}", flush=True)
    for frame, position, source in mismatches[:20]:
        print(f"  frame {frame} block {position} <- retrieved from frame {source}", flush=True)

    videos = {name: SingleVideo(args.out, name) for name in ("target", "recorded-phase-lookup", "error-x%g" % args.error_gain)}
    written = 0
    with torch.inference_mode():
        for i in range(frames):
            truth = raw[i]
            retrieved = raw[found[i], torch.arange(positions)]
            error = (truth.short() - retrieved.short()).abs().mul(args.error_gain).clamp(0, 255).byte()
            for name, value in (("target", truth), ("recorded-phase-lookup", retrieved),
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
