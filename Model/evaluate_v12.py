"""Evaluate the deterministic V12 blend against current-frame Bayer nearest on BBB Q5."""
import argparse
from collections import defaultdict, deque
from contextlib import ExitStack, closing
import json
from pathlib import Path
import subprocess

import torch

from evaluate import demosaic
from evaluate_fsr import RgbVideo, summarize, update_stats
from evaluate_v10 import SingleVideo
from record_experiment import digest, run_logged
from v11.data import BLOCKS, HEIGHT, LR_HEIGHT, LR_WIDTH, WIDTH, build_windows, connect
from v12 import HISTORY, PHASE_DECAY_POINTS, blend_bytes, nearest_bayer_bytes, phase_protected_blend


def strips(blobs, height, width):
    raw = bytearray(b"".join(blobs))
    return torch.frombuffer(raw, dtype=torch.uint8).view(-1, 1, height, width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame-range", type=int, nargs=2, default=(721, 899), metavar=("FIRST", "LAST"))
    parser.add_argument("--blend-mode", choices=("phase-protected", "fixed"), default="phase-protected")
    parser.add_argument("--phase-decay-points", type=int, choices=range(100 // (HISTORY - 1) + 1),
                        default=PHASE_DECAY_POINTS, help="integer percentage points lost per older frame")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    start, stop = args.frame_range
    if not 0 <= start <= stop:
        parser.error("invalid inclusive frame range")
    if args.out.exists():
        raise ValueError("select a new output directory")
    torch.set_num_threads(4)
    scene_plan = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    database_hash = digest(args.db)
    if database_hash != scene_plan["database_sha256"]:
        raise ValueError("scene plan belongs to a different database")
    cuts = sorted(set(scene_plan["scene_starts"]))
    windows = build_windows(args.db, HISTORY, start, stop, scene_starts=cuts)
    if not windows:
        raise ValueError("no complete sixteen-frame windows")
    scored_by_frame = defaultdict(list)
    for block, frames in windows:
        scored_by_frame[frames[-1]].append(block)
    args.out.mkdir(parents=True)
    names = ("bayer-nearest", "v12")
    raw_stats = defaultdict(lambda: [0., 0., 0])
    rgb_stats = defaultdict(lambda: [0., 0., 0])
    temporal = defaultdict(float)
    temporal_pixels = wins = ties = fallback = 0
    protected_pixels = 0
    previous_errors, previous_frame = {}, None
    history = {}
    with ExitStack() as cleanup, torch.inference_mode():
        conn = cleanup.enter_context(closing(connect(args.db)))
        detail = cleanup.enter_context((args.out/"blocks.jsonl").open("w", encoding="utf-8"))
        video = candidate = None
        if args.video:
            video = RgbVideo(args.out, ["target", *names])
            cleanup.callback(video.close)
            candidate = SingleVideo(args.out)
            cleanup.callback(candidate.close)
        for frame in range(start, stop + 1):
            if frame in cuts:
                history.clear()
            inputs = conn.execute("SELECT block_index, phase_id, decoded FROM samples "
                                  "WHERE frame_index=? ORDER BY block_index", (frame,)).fetchall()
            if not inputs and not video:
                continue
            ids = [row[0] for row in inputs]
            if video and ids != list(range(BLOCKS)):
                raise ValueError(f"video needs all {BLOCKS} decoded strips at frame {frame}")
            low = strips([row[2] for row in inputs], LR_HEIGHT, LR_WIDTH)
            phases = torch.tensor([row[1] - 1 for row in inputs])
            base = nearest_bayer_bytes(low, phases)
            by_block = dict(zip(ids, base))
            for (block, phase, _), decoded_strip in zip(inputs, low):
                recent = history.get(block)
                if recent is None or recent[0][0] != frame - 1:
                    recent = history[block] = deque(maxlen=HISTORY)
                recent.appendleft((frame, phase - 1, by_block[block], decoded_strip))
            scored = scored_by_frame.get(frame, [])
            if not scored and not video:
                continue
            selected = {}
            for block in scored:
                _, history_phases, enlarged, decoded = zip(*history[block])
                if args.blend_mode == "phase-protected":
                    selected[block] = phase_protected_blend(enlarged, decoded, history_phases,
                                                            decay_points=args.phase_decay_points)
                    protected_pixels += len(set(history_phases)) * LR_HEIGHT * LR_WIDTH
                else:
                    selected[block] = blend_bytes(enlarged)
            blocks = ids if video else scored
            outputs = {
                "bayer-nearest": torch.stack([by_block[b] for b in blocks]).float() / 255,
                "v12": torch.stack([selected.get(b, by_block[b]) for b in blocks]).float() / 255,
            }
            # Read labels only after the blend is ready.
            labels = dict(conn.execute("SELECT block_index, raw_bayer_target FROM samples WHERE frame_index=?",
                                       (frame,)).fetchall())
            truth = strips([labels[b] for b in blocks], HEIGHT, WIDTH).float() / 255
            rgb = {name: demosaic(value) for name, value in outputs.items()}
            truth_rgb = demosaic(truth)
            if scored:
                index = [blocks.index(b) for b in scored]
                errors = {n: update_stats(raw_stats, n, outputs[n][index], truth[index]) for n in names}
                rgb_errors = {n: update_stats(rgb_stats, n, rgb[n][index], truth_rgb[index]) for n in names}
                wins += sum(a < b for a, b in zip(errors["v12"], errors["bayer-nearest"]))
                ties += sum(a == b for a, b in zip(errors["v12"], errors["bayer-nearest"]))
                for i, block in enumerate(scored):
                    detail.write(json.dumps({"frame": frame, "block": block,
                                             "raw_projection_mae": {n: e[i] * 255 for n, e in errors.items()},
                                             "rgb_reference_mae": {n: e[i] * 255 for n, e in rgb_errors.items()}}) + "\n")
                if scored == list(range(BLOCKS)):
                    residuals = {name: outputs[name][index] - truth[index] for name in names}
                    if previous_frame == frame - 1:
                        for name in names:
                            temporal[name] += (residuals[name] - previous_errors[name]).abs().sum().item()
                        temporal_pixels += residuals["v12"].numel()
                    previous_errors, previous_frame = residuals, frame
            if video:
                images = {name: dict(zip(blocks, value)) for name, value in {"target": truth_rgb, **rgb}.items()}
                video.write(frame, images)
                candidate.write(images["v12"])
                fallback += BLOCKS - len(scored)
    video_info = None
    if video:
        if video.frames != stop - start + 1 or candidate.frames != video.frames:
            raise RuntimeError("video frame count differs from the requested range")
        for name in ("generated.mp4", "comparison.mp4"):
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror", "-i", str(args.out/name),
                            "-f", "null", "-"], check=True)
        candidates = [start] + [(clip["start"] + clip["stop"]) // 2 for clip in scene_plan.get("previews", [])]
        preview_frames = sorted({frame for frame in candidates if start <= frame <= stop})
        for frame in preview_frames:
            png = args.out/f"preview-{frame}.png"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
                            "-i", str(args.out/"comparison.mp4"), "-vf", f"select=eq(n\\,{frame - start})",
                            "-frames:v", "1", "-update", "1", str(png)], check=True)
            if not png.is_file():
                raise RuntimeError(f"could not read preview frame {frame}")
        video_info = {"frames": video.frames, "nearest_fallback_strips": fallback,
                      "preview_frames": preview_frames}
    report = {
        "architecture": "v12-deterministic", "samples": len(windows),
        "start_frame": start, "stop_frame_inclusive": stop,
        "first_evaluated_frame": min(scored_by_frame), "last_evaluated_frame": max(scored_by_frame),
        "history_frames": HISTORY, "scene_starts": cuts, "db_sha256": database_hash,
        "blend_mode": args.blend_mode,
        "phase_weight_percent_by_age": ([100 - age * args.phase_decay_points for age in range(HISTORY)]
                                        if args.blend_mode == "phase-protected" else None),
        "phase_weight_remainder": "50/50 fold of other frames, excluding the phase owner",
        "phase_protected_fraction": protected_pixels / (len(windows) * HEIGHT * WIDTH),
        "training_steps": 0, "fallback_blend": "newest-first; floor((accumulator + older_byte) / 2) at each step",
        "baseline": "copy the nearest complete 2x2 Bayer cell; ties choose the higher index",
        "raw_projection_metrics": summarize(raw_stats),
        "rgb_reference_metrics": summarize(rgb_stats),
        "v12_win_rate": wins / len(windows), "ties": ties,
        "consecutive_frame_residual_mae": {n: value / temporal_pixels * 255 for n, value in temporal.items()},
        "video": video_info,
    }
    (args.out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    run_logged("evaluation", main)
