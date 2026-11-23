"""Export target / bayer-nearest / v12 as three separate full-resolution videos for demo.mp4."""
import argparse
from collections import deque
from contextlib import ExitStack, closing
import json
from pathlib import Path

import torch

from evaluate import demosaic
from evaluate_v10 import SingleVideo
from evaluate_v13 import verify_video
from v11.data import BLOCKS, HEIGHT, LR_HEIGHT, LR_WIDTH, WIDTH, connect
from v12 import HISTORY, blend_bytes, nearest_bayer_bytes, phase_protected_blend


def strips(blobs, height, width):
    raw = bytearray(b"".join(blobs))
    return torch.frombuffer(raw, dtype=torch.uint8).view(-1, 1, height, width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame-range", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    args = parser.parse_args()
    start, stop = args.frame_range
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    scene_plan = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    cuts = set(scene_plan["scene_starts"])

    history = {}
    videos = {name: SingleVideo(args.out, name) for name in ("target", "bayer-nearest", "v12")}
    written = 0
    with ExitStack() as cleanup, torch.inference_mode():
        for video in videos.values():
            cleanup.callback(video.close)
        conn = cleanup.enter_context(closing(connect(args.db)))
        for frame in range(start, stop + 1):
            if frame in cuts:
                history.clear()
            inputs = conn.execute("SELECT block_index, phase_id, decoded FROM samples "
                                  "WHERE frame_index=? ORDER BY block_index", (frame,)).fetchall()
            ids = [row[0] for row in inputs]
            if ids != list(range(BLOCKS)):
                raise ValueError(f"frame {frame}: expected all {BLOCKS} blocks, found {ids}")
            low = strips([row[2] for row in inputs], LR_HEIGHT, LR_WIDTH)
            phases = torch.tensor([row[1] - 1 for row in inputs])
            base = nearest_bayer_bytes(low, phases)
            by_block = dict(zip(ids, base))
            for (block, phase, _), decoded_strip in zip(inputs, low):
                recent = history.get(block)
                if recent is None or recent[0][0] != frame - 1:
                    recent = history[block] = deque(maxlen=HISTORY)
                recent.appendleft((frame, phase - 1, by_block[block], decoded_strip))
            if frame - start < HISTORY - 1:
                continue  # not enough history yet for a full-frame v12 blend
            v12_blocks = {}
            for block in range(BLOCKS):
                _, history_phases, enlarged, decoded = zip(*history[block])
                v12_blocks[block] = phase_protected_blend(enlarged, decoded, history_phases)
            labels = dict(conn.execute("SELECT block_index, raw_bayer_target FROM samples WHERE frame_index=?",
                                       (frame,)).fetchall())
            truth = strips([labels[b] for b in range(BLOCKS)], HEIGHT, WIDTH).float() / 255
            outputs = {
                "target": truth,
                "bayer-nearest": torch.stack([by_block[b] for b in range(BLOCKS)]).float() / 255,
                "v12": torch.stack([v12_blocks[b] for b in range(BLOCKS)]).float() / 255,
            }
            for name, value in outputs.items():
                videos[name].write(dict(enumerate(demosaic(value))))
            written += 1
    for name in videos:
        verify_video(args.out / f"{name}.mp4", written)
    print(f"wrote {written} frames to {args.out}", flush=True)


if __name__ == "__main__":
    main()
