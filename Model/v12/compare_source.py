"""Compare rendered V12 variants against the matching original RGB frames."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from evaluate_fsr import RgbVideo, summarize
from record_experiment import digest, run_logged
from v11.data import BLOCKS, HEIGHT, WIDTH, connect


FRAME_HEIGHT = BLOCKS * HEIGHT
NAMES = ("original", "v12 fixed 50-50", "v12 phase weight 1", "v12 phase decay 0.02")


def decode_range(path, first, last, *, scale_original=False):
    filters = [f"select=between(n\\,{first}\\,{last})"]
    if scale_original:
        filters.append(f"scale={WIDTH}:{FRAME_HEIGHT}:flags=bilinear")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-xerror",
               "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
               "-vf", ",".join(filters), "-fps_mode", "passthrough",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    data = subprocess.run(command, capture_output=True, check=True).stdout
    if len(data) != (last - first + 1) * WIDTH * FRAME_HEIGHT * 3:
        raise ValueError(f"decoded RGB frame count or size differs for {path}")
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, FRAME_HEIGHT, WIDTH, 3), command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--scene-plan", required=True, type=Path)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--candidate-video", required=True, type=Path, nargs=3,
                        metavar=("FIXED", "PHASE_ONE", "PHASE_DECAY"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("select a new output directory")
    torch.set_num_threads(4)
    plan = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    database_sha, source_sha = digest(args.db), digest(args.source_video)
    if plan["database_sha256"] != database_sha or plan["source_video_sha256"] != source_sha:
        raise ValueError("source video or database differs from scene plan")
    candidate_reports = [path.with_name("report.json") for path in args.candidate_video]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in candidate_reports]
    start = max(report["first_evaluated_frame"] for report in reports)
    stop = min(report["last_evaluated_frame"] for report in reports)
    for report in reports:
        if (report["db_sha256"] != database_sha or
                report["video"]["frames"] != report["stop_frame_inclusive"] - report["start_frame"] + 1 or
                report["samples"] != BLOCKS * (stop - start + 1)):
            raise ValueError("candidate does not cover the same complete window")
    if start > stop:
        raise ValueError("no common output frames")
    with closing(connect(args.db)) as conn:
        rows = conn.execute(
            "SELECT frame_index, min(source_frame_index), max(source_frame_index), count(*) "
            "FROM samples WHERE frame_index BETWEEN ? AND ? GROUP BY frame_index ORDER BY frame_index",
            (start, stop),
        ).fetchall()
    if ([row[0] for row in rows] != list(range(start, stop + 1)) or
            any(first != last or count != BLOCKS for _, first, last, count in rows)):
        raise ValueError("incomplete output-to-original frame mapping")
    sources = [row[1] for row in rows]
    original, source_ffmpeg = decode_range(args.source_video, min(sources), max(sources), scale_original=True)
    images_by_method = {}
    provenance = {}
    for name, path, report, report_path in zip(NAMES[1:], args.candidate_video, reports, candidate_reports):
        frames, command = decode_range(path, start - report["start_frame"], stop - report["start_frame"])
        images_by_method[name] = frames
        provenance[name] = {"report": str(report_path), "report_sha256": digest(report_path),
                            "ffmpeg_argv": command}
    args.out.mkdir(parents=True)
    stats = {name: [0., 0., 0] for name in images_by_method}
    temporal = dict.fromkeys(images_by_method, 0)
    previous_errors = {}
    frame_details = []
    video = RgbVideo(args.out, NAMES)
    try:
        for index, source_index in enumerate(sources):
            truth = original[source_index - min(sources)]
            pixels = {"original": truth, **{name: frames[index] for name, frames in images_by_method.items()}}
            frame_errors = {}
            for name, stats_row in stats.items():
                error = pixels[name].astype(np.float32) - truth
                absolute = np.abs(error).sum(dtype=np.float64)
                frame_errors[name] = absolute / error.size
                stats_row[0] += absolute / 255
                stats_row[1] += np.square(error).sum(dtype=np.float64) / (255 ** 2)
                stats_row[2] += error.size
                if name in previous_errors:
                    temporal[name] += np.abs(error - previous_errors[name]).sum(dtype=np.float64)
                previous_errors[name] = error
            frame = start + index
            frame_details.append({"frame": frame, "source_frame": source_index, "rgb_mae": frame_errors})
            panels = {}
            for name, rgb in pixels.items():
                rgb = torch.tensor(rgb).permute(2, 0, 1).float().div_(255)
                panels[name] = dict(enumerate(rgb.split(HEIGHT, dim=1)))
            video.write(frame, panels)
    finally:
        video.close()
    if video.frames != len(rows):
        raise RuntimeError("comparison frame count differs from the common window")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror", "-i",
                    str(args.out/"comparison.mp4"), "-f", "null", "-"], check=True)
    previews = [frame for frame in (779, 874) if start <= frame <= stop]
    for frame in previews:
        png = args.out/f"preview-{frame}.png"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
                        "-i", str(args.out/"comparison.mp4"), "-vf", f"select=eq(n\\,{frame - start})",
                        "-frames:v", "1", "-update", "1", str(png)], check=True)
        if not png.is_file():
            raise RuntimeError(f"could not read back comparison frame {frame}")
    (args.out/"frames.jsonl").write_text(
        "".join(json.dumps(frame) + "\n" for frame in frame_details), encoding="utf-8"
    )
    report = {
        "samples": len(rows) * BLOCKS, "start_frame": start, "first_evaluated_frame": start,
        "last_evaluated_frame": stop, "frames": len(rows), "unique_source_frames": len(set(sources)),
        "reference": "Original video decoded to RGB24 with the dataset's 640x480 scaler and source frame mapping",
        "candidate_input": "previously recorded V12-generated H.264 files",
        "source_ffmpeg_argv": source_ffmpeg, "candidates": provenance,
        "rgb_metrics": summarize(stats),
        "rgb_consecutive_residual_mae": {n: value / ((len(rows) - 1) * WIDTH * FRAME_HEIGHT * 3)
                                         for n, value in temporal.items()} if len(rows) > 1 else {},
        "preview_frames": previews,
    }
    (args.out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    run_logged("evaluation", main)
