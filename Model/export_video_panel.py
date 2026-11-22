"""Export one 640x480 image panel from a recorded comparison video."""
import argparse
import json
from pathlib import Path
import subprocess

from baseline import fingerprint
from record_experiment import run_logged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("select a new output directory")
    report_path = args.evaluation/"report.json"
    source = args.evaluation/"comparison.mp4"
    original = json.loads(report_path.read_text(encoding="utf-8"))
    index = original["video"]["panel_order"].index(args.panel)
    x, y = index % 2*640, index//2*508+28
    args.out.mkdir(parents=True)
    output = args.out/"echosr-lite.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-i", str(source),
               "-vf", f"crop=640:480:{x}:{y}", "-an", "-c:v", "libx264", "-preset", "fast",
               "-crf", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    with (args.out/"ffmpeg.log").open("wb") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,r_frame_rate,nb_read_frames:format=duration", "-of", "json", str(output)], text=True))
    stream = probe["streams"][0]
    if ((stream["width"], stream["height"]) != (640, 480) or stream["r_frame_rate"] != "30/1"
            or int(stream["nb_read_frames"]) != original["video"]["frames"]):
        raise ValueError("export dimensions, frame rate or frame count mismatch")
    record = {"operation": "crop comparison panel; no model inference", "panel": args.panel,
              "source_video": str(source), "source_sha256": fingerprint(source),
              "source_report": str(report_path), "source_report_sha256": fingerprint(report_path),
              "crop": {"width": 640, "height": 480, "x": x, "y": y},
              "command": command, "output": str(output), "output_sha256": fingerprint(output),
              "probe": probe, "note": "Lossless re-encoding of decoded comparison video crop; source video already used lossy H264."}
    (args.out/"report.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "probe": probe}, indent=2))


if __name__ == "__main__":
    run_logged("evaluation", main)
