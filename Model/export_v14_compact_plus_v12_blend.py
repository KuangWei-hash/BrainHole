"""Combine the compact (patch+DCT) single-frame reconstruction with v12's
16-frame phase-protected temporal blend: the per-frame "enlarged" estimate fed
into the blend is the compact codebook's output instead of a crude nearest-
Bayer fill. Compares three methods on the same frames: compact alone (no
history), v12 (nearest-Bayer + history), and this hybrid (compact + history).
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
from v11.data import BLOCKS, HEIGHT, LR_HEIGHT, LR_WIDTH, WIDTH, connect
from v12 import HISTORY, nearest_bayer_bytes, phase_protected_blend


def strips(blobs, height, width):
    raw = bytearray(b"".join(blobs))
    return torch.frombuffer(raw, dtype=torch.uint8).view(-1, 1, height, width)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame-range", type=int, nargs=2, default=(1748, 1852), metavar=("FIRST", "LAST"))
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

    history_compact, history_nearest = {}, {}
    names = ("compact-alone", "v12", "compact+v12-blend")
    raw_stats = {n: [0., 0., 0] for n in names}
    videos = {n: SingleVideo(args.out, n) for n in (*names, "target")}
    written = 0
    with ExitStack() as cleanup, torch.inference_mode():
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
                for hist, value in ((history_compact, compact_by_block[block]),
                                    (history_nearest, nearest_by_block[block])):
                    recent = hist.get(block)
                    if recent is None or recent[0][0] != frame - 1:
                        recent = hist[block] = deque(maxlen=HISTORY)
                    recent.appendleft((frame, phase, value, low[block]))

            if frame - start < HISTORY - 1:
                continue
            outputs = {}
            for block in range(BLOCKS):
                _, hp, enlarged_c, decoded_c = zip(*history_compact[block])
                _, _, enlarged_n, decoded_n = zip(*history_nearest[block])
                outputs.setdefault("compact-alone", {})[block] = compact_by_block[block]
                outputs.setdefault("v12", {})[block] = phase_protected_blend(enlarged_n, decoded_n, hp)
                outputs.setdefault("compact+v12-blend", {})[block] = phase_protected_blend(enlarged_c, decoded_c, hp)

            labels_row = dict(conn.execute("SELECT block_index, raw_bayer_target FROM samples "
                                           "WHERE frame_index=?", (frame,)).fetchall())
            truth = strips([labels_row[b] for b in range(BLOCKS)], HEIGHT, WIDTH).float() / 255
            truth_rgb = demosaic(truth)
            videos["target"].write(dict(enumerate(truth_rgb)))
            for name in names:
                pred = torch.stack([outputs[name][b] for b in range(BLOCKS)]).float() / 255
                update_stats(raw_stats, name, pred, truth)
                videos[name].write(dict(enumerate(demosaic(pred))))
            written += 1

    for name in (*names, "target"):
        verify_video(args.out / f"{name}.mp4", written)
    report = {"frame_range": [start, stop], "scored_frames": written,
              "methods": {n: {"raw_mae": s[0] / s[2] * 255} for n, s in raw_stats.items()}}
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in names:
        print(f"{name}: RAW MAE {report['methods'][name]['raw_mae']:.4f}", flush=True)


if __name__ == "__main__":
    main()
