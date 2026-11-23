"""Export the v14 "v12-index, similarity-gated" method as a video for demo.mp4.

Mirrors the frozen BBB setting from 實驗紀錄-v14-v12索引查表.md: table keyed by
v12's own reconstruction, values are the true RAW strip at build time; query time
uses the current v12 output as the key, retrieves the nearest table entry by RAW
MAE, and blends it into v12 with weight max(0, 1 - mae/threshold) * strength
(threshold=24, strength=1.0, the setting frozen there). The current frame's
actually-measured phase pixels are restored after blending, same as v12 itself.
"""
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
from v11.phase import scatter_indices
from v12 import HISTORY, nearest_bayer_bytes, phase_protected_blend

THRESHOLD, STRENGTH = 24.0, 1.0


def strips(blobs, height, width):
    raw = bytearray(b"".join(blobs))
    return torch.frombuffer(raw, dtype=torch.uint8).view(-1, 1, height, width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--table-range", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--frame-range", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    scene_plan = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    cuts = set(scene_plan["scene_starts"])

    table_first, table_last = args.table_range
    start, stop = args.frame_range
    if table_last >= start:
        raise ValueError("table range must end before the scored frame range begins")
    run_first = min(table_first, start)

    history = {}
    table_keys = {b: [] for b in range(BLOCKS)}
    table_values = {b: [] for b in range(BLOCKS)}
    video = SingleVideo(args.out, "v14")
    written = 0
    with ExitStack() as cleanup, torch.inference_mode():
        cleanup.callback(video.close)
        conn = cleanup.enter_context(closing(connect(args.db)))
        for frame in range(run_first, stop + 1):
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
            if frame - run_first < HISTORY - 1:
                continue

            v12_blocks = {}
            for block in range(BLOCKS):
                _, history_phases, enlarged, decoded = zip(*history[block])
                v12_blocks[block] = phase_protected_blend(enlarged, decoded, history_phases)

            if table_first <= frame <= table_last:
                labels = dict(conn.execute("SELECT block_index, raw_bayer_target FROM samples WHERE frame_index=?",
                                           (frame,)).fetchall())
                for block in range(BLOCKS):
                    truth = torch.frombuffer(bytearray(labels[block]), dtype=torch.uint8).view(1, HEIGHT, WIDTH)
                    table_keys[block].append(v12_blocks[block].clone())
                    table_values[block].append(truth.clone())

            if frame < start + HISTORY - 1:
                continue  # align with the scored window used by the v12/target exports
            predicted = {}
            for block in range(BLOCKS):
                query = v12_blocks[block].short()
                keys = torch.stack(table_keys[block]).short()
                values = torch.stack(table_values[block])
                mae = (keys - query).abs().float().mean(dim=(1, 2, 3))
                best = mae.argmin()
                weight = max(0., 1 - mae[best].item() / THRESHOLD) * STRENGTH
                blended = (v12_blocks[block].float() + weight * (values[best].float() - v12_blocks[block].float()))
                result = blended.round().clamp(0, 255).byte()
                phase = int(phases[ids.index(block)])
                rows, cols = scatter_indices(phase)
                result[:, rows[:, None], cols[None, :]] = low[block, 0]
                predicted[block] = result
            rgb = demosaic(torch.stack([predicted[b] for b in range(BLOCKS)]).float() / 255)
            video.write(dict(enumerate(rgb)))
            written += 1
    verify_video(args.out / "v14.mp4", written)
    print(f"table entries per position: {len(next(iter(table_keys.values())))}; wrote {written} frames", flush=True)


if __name__ == "__main__":
    main()
