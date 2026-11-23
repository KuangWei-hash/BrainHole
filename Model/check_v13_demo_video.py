"""Whole-video RAW MAE summary for the v13 demo.mp4 export (counterpart to check_v14_exact_video.py's report)."""
import argparse
from collections import deque
from contextlib import closing
import json
from pathlib import Path

import torch

from evaluate_v13 import load_restorer
from v11.data import BLOCKS, connect
from v12 import HISTORY
from v13.base import features
from v13.data import fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    scene_plan = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    cuts = set(scene_plan["scene_starts"])
    provenance = fingerprints(args.db, args.scene_plan)
    device = torch.device(args.device)
    model, info = load_restorer(args.checkpoint, device, provenance)

    history = {}
    abs_sum = sq_sum = count = 0
    warmup_abs_sum = warmup_count = 0
    with torch.inference_mode(), closing(connect(args.db)) as conn:
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
            predicted = model.predict(batch, 1.0).cpu()
            labels = dict(conn.execute("SELECT block_index, raw_bayer_target FROM samples WHERE frame_index=?",
                                       (frame,)).fetchall())
            truth = torch.frombuffer(bytearray(b"".join(labels[b] for b in range(BLOCKS))),
                                     dtype=torch.uint8).view(BLOCKS, 1, 16, 640)
            error = predicted.float() - truth.float()
            abs_sum += error.abs().sum().item()
            sq_sum += error.square().sum().item()
            count += error.numel()
            full_history = all(len(history[block]) == HISTORY for block in range(BLOCKS))
            if not full_history:
                warmup_abs_sum += error.abs().sum().item()
                warmup_count += error.numel()
    print(json.dumps({
        "frames": last - first + 1,
        "samples": count // (16 * 640),
        "raw_mae": abs_sum / count,
        "raw_rmse": (sq_sum / count) ** 0.5,
        "warmup_strips": warmup_count // (16 * 640),
        "warmup_raw_mae": (warmup_abs_sum / warmup_count) if warmup_count else None,
        "checkpoint": info["checkpoint"],
        "epoch": info["epoch"],
    }, indent=2))


if __name__ == "__main__":
    main()
