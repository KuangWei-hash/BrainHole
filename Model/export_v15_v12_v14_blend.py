"""v15: compute v12 and v14 (compact codebook + phase-protected 30/70 recursive
blend) completely independently, then average their two finished 640x16
frames 50/50. Motivation: visual inspection found v12 keeps the sharpest
texture in static regions (real-pixel temporal accumulation) while v14 keeps
shapes intact under motion but with softer (DCT-residual-compressed) texture;
this tries to keep more of each method's strength. See v15/__init__.py.
"""
import argparse
from collections import deque
from contextlib import ExitStack, closing
import json
from pathlib import Path

import torch

from evaluate import demosaic
from evaluate_fsr import update_stats
from evaluate_v10 import SingleVideo
from evaluate_v13 import verify_video
import v14 as compact
import v15
from v11.data import BLOCKS, HEIGHT, LR_HEIGHT, LR_WIDTH, WIDTH, connect
from v11.phase import scatter_indices
from v12 import HISTORY, nearest_bayer_bytes, phase_protected_blend


def strips(blobs, height, width):
    raw = bytearray(b"".join(blobs))
    return torch.frombuffer(raw, dtype=torch.uint8).view(-1, 1, height, width)


def simple_phase_reset_blend(current, previous, phase, current_weight=0.3):
    """current, previous: (1,16,640) uint8. previous may be None (first frame)."""
    if previous is None:
        return current.clone()
    mixed = (current.float() * current_weight + previous.float() * (1 - current_weight)).round().byte()
    rows, cols = scatter_indices(phase)
    mixed[:, rows[:, None], cols[None, :]] = current[:, rows[:, None], cols[None, :]]
    return mixed


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame-range", type=int, nargs=2, default=(1733, 1852), metavar=("FIRST", "LAST"))
    parser.add_argument("--score-start", type=int, default=1763)
    parser.add_argument("--v14-current-weight", type=float, default=0.3)
    parser.add_argument("--v15-weight", type=float, default=0.5, help="v12's share in the final 50/50 blend")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)
    start, stop = args.frame_range

    book = torch.load(args.codebook, map_location="cpu", weights_only=True)
    compact.RANK = book["config"].get("dct_rank", compact.RANK)
    width = book["config"].get("patch_width", 40)
    tiles = compact.tiles_for(width)
    basis = compact.dct_basis(width)
    gate = book["gate"]

    previous_simple = {b: None for b in range(BLOCKS)}
    history_nearest = {}
    names = ("v12", "v14", "v15")
    raw_stats = {n: [0., 0., 0] for n in names}
    videos = {n: SingleVideo(args.out, n) for n in (*names, "target")}
    written = 0
    with ExitStack() as cleanup:
        for v in videos.values():
            cleanup.callback(v.close)
        conn = cleanup.enter_context(closing(connect(args.db)))
        for frame in range(start, stop + 1):
            inputs = conn.execute("SELECT block_index, phase_id, decoded FROM samples "
                                  "WHERE frame_index=? ORDER BY block_index", (frame,)).fetchall()
            ids = [row[0] for row in inputs]
            if ids != list(range(BLOCKS)):
                raise ValueError(f"frame {frame}: expected all {BLOCKS} blocks")
            low = strips([row[2] for row in inputs], LR_HEIGHT, LR_WIDTH)
            phases = torch.tensor([row[1] - 1 for row in inputs])

            flat_samples = low.squeeze(1)
            base = compact.bilinear(flat_samples, phases)
            positions = torch.arange(BLOCKS)
            patches, distance = compact.retrieve(book, flat_samples, positions, phases, basis)
            compact_pred = compact.reconstruct(base, patches, distance, phases, **gate)
            compact_by_block = dict(enumerate(compact_pred))

            nearest_pred = nearest_bayer_bytes(low, phases)
            nearest_by_block = dict(enumerate(nearest_pred))
            for block, phase in zip(ids, phases.tolist()):
                recent = history_nearest.get(block)
                if recent is None or recent[0][0] != frame - 1:
                    recent = history_nearest[block] = deque(maxlen=HISTORY)
                recent.appendleft((frame, phase, nearest_by_block[block], low[block]))

            v14_by_block = {}
            for block, phase in zip(ids, phases.tolist()):
                v14_by_block[block] = simple_phase_reset_blend(
                    compact_by_block[block], previous_simple[block], phase, args.v14_current_weight)
                previous_simple[block] = v14_by_block[block]

            if frame < args.score_start:
                continue
            v12_by_block = {}
            for block in range(BLOCKS):
                _, hp, enlarged_n, decoded_n = zip(*history_nearest[block])
                v12_by_block[block] = phase_protected_blend(enlarged_n, decoded_n, hp)

            v15_by_block = {block: v15.blend(v12_by_block[block], v14_by_block[block], args.v15_weight)
                            for block in range(BLOCKS)}

            outputs = {"v12": v12_by_block, "v14": v14_by_block, "v15": v15_by_block}
            labels_row = dict(conn.execute("SELECT block_index, raw_bayer_target FROM samples "
                                           "WHERE frame_index=?", (frame,)).fetchall())
            truth = strips([labels_row[b] for b in range(BLOCKS)], HEIGHT, WIDTH).float() / 255
            videos["target"].write(dict(enumerate(demosaic(truth))))
            for name in names:
                pred = torch.stack([outputs[name][b] for b in range(BLOCKS)]).float() / 255
                update_stats(raw_stats, name, pred, truth)
                videos[name].write(dict(enumerate(demosaic(pred))))
            written += 1

    for name in (*names, "target"):
        verify_video(args.out / f"{name}.mp4", written)
    report = {"frame_range": [start, stop], "score_start": args.score_start, "scored_frames": written,
              "methods": {n: {"raw_mae": s[0] / s[2] * 255} for n, s in raw_stats.items()}}
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in names:
        print(f"{name}: RAW MAE {report['methods'][name]['raw_mae']:.4f}", flush=True)


if __name__ == "__main__":
    main()
