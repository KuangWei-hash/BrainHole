"""Export target / v13 full-resolution videos for the whole demo.mp4 source video.

Counterpart to export_v14_exact_lookup_video.py's whole-video scope, but for v13's
fixed-base-plus-residual-CNN reconstruction instead of v14's recorded-phase lookup.
Runs every frame 0..last (not just a fixed scored window) through the same causal,
per-block history v13 uses at inference time, so the two methods' videos cover the
identical demo.mp4 segment and can be compared frame for frame.
"""
import argparse
from collections import deque
from contextlib import ExitStack, closing
import json
from pathlib import Path

import torch

from evaluate import demosaic
from evaluate_v10 import SingleVideo
from evaluate_v13 import load_restorer, verify_video
from v11.data import BLOCKS, connect
from v12 import HISTORY
from v13.base import features
from v13.data import fingerprints


def strips(blobs, height, width):
    raw = bytearray(b"".join(blobs))
    return torch.frombuffer(raw, dtype=torch.uint8).view(-1, 1, height, width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    scene_plan = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    cuts = set(scene_plan["scene_starts"])
    provenance = fingerprints(args.db, args.scene_plan)
    device = torch.device(args.device)
    model, info = load_restorer(args.checkpoint, device, provenance)

    history = {}
    videos = {name: SingleVideo(args.out, name) for name in ("target", "v13")}
    written = 0
    with ExitStack() as cleanup, torch.inference_mode():
        for video in videos.values():
            cleanup.callback(video.close)
        conn = cleanup.enter_context(closing(connect(args.db)))
        first, last = conn.execute("SELECT min(frame_index), max(frame_index) FROM samples").fetchone()
        for frame in range(first, last + 1):
            if frame in cuts:
                history.clear()
            inputs = conn.execute("SELECT block_index, phase_id, decoded FROM samples "
                                  "WHERE frame_index=? ORDER BY block_index", (frame,)).fetchall()
            ids = [row[0] for row in inputs]
            if ids != list(range(BLOCKS)):
                raise ValueError(f"frame {frame}: expected all {BLOCKS} blocks, found {ids}")
            for block, phase, decoded_blob in inputs:
                strip = torch.frombuffer(bytearray(decoded_blob), dtype=torch.uint8).reshape(1, 4, 160)
                recent = history.get(block)
                if recent is None or recent[0][0] != frame - 1:
                    recent = history[block] = deque(maxlen=HISTORY)
                recent.appendleft((frame, phase - 1, strip))
            block_features = []
            for block in range(BLOCKS):
                _, phases, decoded = zip(*history[block])
                block_features.append(features(torch.stack(decoded), phases))
            batch = torch.stack(block_features).to(device)
            predicted = model.predict(batch, args.alpha).cpu()
            labels = dict(conn.execute("SELECT block_index, raw_bayer_target FROM samples WHERE frame_index=?",
                                       (frame,)).fetchall())
            truth = strips([labels[b] for b in range(BLOCKS)], 16, 640).float() / 255
            outputs = {"target": truth, "v13": predicted.float() / 255}
            for name, value in outputs.items():
                videos[name].write(dict(enumerate(demosaic(value))))
            written += 1
    for name in videos:
        verify_video(args.out / f"{name}.mp4", written)
    print(f"wrote {written} frames to {args.out}", flush=True)


if __name__ == "__main__":
    main()
