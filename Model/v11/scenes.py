"""Identify scene cuts in the source video and map them to Q5 output frames."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import re
import subprocess

from record_experiment import digest, run_logged
from v11.data import connect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=8.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.threshold <= 100:
        parser.error("threshold must be within (0, 100]")
    if args.out.exists():
        raise ValueError("select a new output file")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2", "-filter_threads", "1",
        "-i", str(args.source_video), "-an",
        "-vf", "scale=320:240,format=yuv420p,scdet,metadata=print:key=lavfi.scd.score:file=-",
        "-f", "null", "-",
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    scores = {int(frame): float(score) for frame, score in re.findall(
        r"frame:\s*(\d+)[^\n]*\nlavfi\.scd\.score=([0-9.]+)", output
    )}
    with closing(connect(args.db)) as conn:
        rows = conn.execute(
            "SELECT frame_index, min(source_frame_index), max(source_frame_index) "
            "FROM samples GROUP BY frame_index ORDER BY frame_index"
        ).fetchall()
    if not rows or any(first != last or first not in scores for _, first, last in rows):
        raise ValueError("source scene scores do not match database frame ordinals")
    if any(a[1] > b[1] for a, b in zip(rows, rows[1:])):
        raise ValueError("source frame ordinals must be monotonic")
    source_cuts = sorted(frame for frame, score in scores.items() if score >= args.threshold)
    pairs = [(a[0], b[0], b[1], scores[b[1]]) for a, b in zip(rows, rows[1:])
             if a[0] >= 721 and b[0] == a[0] + 1]
    cuts = [b[0] for a, b in zip(rows, rows[1:]) if any(a[1] < cut <= b[1] for cut in source_cuts)]
    previews = []
    # Fix preview windows before training and leave room for motion and T=4 history.
    candidates = [frame for frame in range(724, rows[-1][0] - 29 + 1, 5)
                  if all(any(f == k for _, f, _, _ in pairs) for k in range(frame, frame + 30))]
    if candidates:
        means = {start: sum(s for _, f, _, s in pairs if start <= f < start + 30) / 30
                 for start in candidates if not any(start - 3 <= cut < start + 30 for cut in cuts)}
        if means:
            quiet = min(means, key=means.get)
            active = [start for start in means if abs(start - quiet) >= 30]
            previews.append({"name": "low-change", "start": quiet, "stop": quiet + 30})
            if active:
                busy = max(active, key=means.get)
                previews.append({"name": "high-change", "start": busy, "stop": busy + 30})
    cut = next((frame for frame in cuts if frame - 15 >= 721 and frame + 15 <= rows[-1][0] + 1), None)
    if cut is not None:
        previews.append({"name": "scene-cut", "start": cut - 15, "stop": cut + 15})
    report = {
        "database_sha256": digest(args.db), "source_video_sha256": digest(args.source_video),
        "ffmpeg_argv": command, "threshold": args.threshold, "scene_starts": cuts,
        "source_scene_starts": source_cuts, "source_scores": scores, "previews": previews,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("scene_starts", "source_scene_starts", "previews")}), flush=True)


if __name__ == "__main__":
    run_logged("diagnostic", main)
