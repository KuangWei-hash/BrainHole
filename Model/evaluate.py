"""Compare RAW reconstruction models with decoded + nearest/bilinear."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from baseline import DecodedCache, upscale
from data import RawDataset, decoded_window
from network import build_model


def demosaic(raw):
    """Fixed bilinear CFA interpolation; same display pipeline for every method."""
    height, width = raw.shape[-2:]
    masks = torch.zeros((1, 3, height, width), device=raw.device)
    masks[:, 0, ::2, ::2] = 1
    masks[:, 1, ::2, 1::2] = 1
    masks[:, 1, 1::2, ::2] = 1
    masks[:, 2, 1::2, 1::2] = 1
    kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=raw.device)
    kernel = kernel.expand(3, 1, 3, 3)
    numerator = F.conv2d(raw*masks, kernel, padding=1, groups=3)
    denominator = F.conv2d(masks, kernel, padding=1, groups=3)
    return numerator/denominator.clamp_min(1)


class ComparisonVideo:
    """Offline visualization only; preserves the dataset's 30 FPS timeline."""
    def __init__(self, folder, names):
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise ValueError("--video requires ffmpeg on PATH")
        self.names, self.frames, self.last_frame = names, 0, None
        self.previous_errors, self.temporal_sum, self.temporal_count = {}, defaultdict(float), defaultdict(int)
        self.size = (1280, ((len(names)+1)//2)*508)
        self.log = (folder/"video-ffmpeg.log").open("wb")
        try:
            self.process = subprocess.Popen([
                executable, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", f"{self.size[0]}x{self.size[1]}", "-r", "30",
                "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "20", "-pix_fmt", "yuv420p", str(folder/"comparison.mp4")],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        except Exception:
            self.log.close()
            raise

    def write(self, frame, methods):
        if self.last_frame is not None and frame != self.last_frame+1:
            raise ValueError("video requires consecutive complete frames")
        sheet = Image.new("RGB", self.size, "white")
        draw = ImageDraw.Draw(sheet)
        target = None
        for index, name in enumerate(self.names):
            if set(methods[name]) != set(range(30)):
                raise ValueError("video requires all 30 blocks per frame")
            mosaic = torch.cat([methods[name][block] for block in range(30)], dim=1)[None]
            if name == "target":
                target = mosaic
            else:
                error = mosaic-target
                if name in self.previous_errors:
                    self.temporal_sum[name] += (error-self.previous_errors[name]).abs().sum().item()
                    self.temporal_count[name] += error.numel()
                self.previous_errors[name] = error
            rgb = demosaic(mosaic)[0].permute(1, 2, 0).clamp(0, 1).numpy()
            x, y = index % 2*640, index//2*508
            draw.text((x+8, y+8), f"{name} | frame {frame}", fill="black")
            sheet.paste(Image.fromarray((rgb*255).round().astype(np.uint8)), (x, y+28))
        self.process.stdin.write(sheet.tobytes())
        self.frames += 1
        self.last_frame = frame

    def close(self):
        try:
            self.process.stdin.close()
        finally:
            code = self.process.wait()
            self.log.close()
        if code != 0:
            raise RuntimeError("ffmpeg video encoding failed; see video-ffmpeg.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--decoded-cache", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, nargs="+")
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--stop-frame", type=int)
    parser.add_argument("--preview-frame", type=int, default=750)
    parser.add_argument("--video", action="store_true", help="render complete consecutive frames to a comparison MP4")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output directory already exists")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = DecodedCache(args.decoded_cache, args.db)
    models, configs = {}, {}
    for path in args.checkpoint:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["config"]["header_id"] != cache.metadata["header_id"]:
            raise ValueError("checkpoint/header mismatch")
        name = path.parent.name
        if name in models:
            raise ValueError("checkpoints must have unique parent directory names")
        configs[name] = saved["config"]
        models[name] = build_model(saved["config"]).to(device).eval()
        models[name].load_state_dict(saved["model"])
    history = max(c["history"] for c in configs.values())
    dataset = RawDataset(args.db, history, args.start_frame, args.stop_frame,
                         v9_upper_strip=any(c.get("v9_upper_strip", False) for c in configs.values()))
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    stats = defaultdict(lambda: {"abs": 0., "sq": 0., "count": 0})
    wins = defaultdict(int)
    low_stats = defaultdict(lambda: [0., 0])
    block_rows, frames = [], defaultdict(dict)
    offset = 0
    args.out.mkdir(parents=True)
    video_names = ["target", "bilinear", *models]
    video_frames = 0
    with ExitStack() as cleanup, torch.inference_mode():
        video = ComparisonVideo(args.out, video_names) if args.video else None
        if video is not None:
            cleanup.callback(video.close)
        pending_video = defaultdict(dict)
        pending_frame = None
        for tokens, phases, lengths, target in loader:
            keys = dataset.keys[offset:offset+len(target)]
            offset += len(target)
            tokens, phases, lengths, target = [x.to(device) for x in (tokens, phases, lengths, target)]
            low = torch.stack([cache.get(key) for key in keys]).to(device)
            outputs = {mode: upscale(low, phases[:, 0], mode) for mode in ("nearest", "bilinear")}
            for name, model in models.items():
                n = configs[name]["history"]
                if getattr(model, "requires_decoded", False):
                    use_upper = configs[name].get("v9_upper_strip", False)
                    indices = list(range(n)) + ([history] if use_upper else [])
                    history_low = torch.stack([decoded_window(cache, [dataset.input_keys(frame, block)[i] for i in indices])
                                               for frame, block in keys]).to(device)
                    outputs[name] = model(tokens[:, indices], phases[:, indices], lengths[:, indices], history_low)
                elif configs[name].get("architecture") == "v2":
                    outputs[name], learned_low = model(tokens[:, :n], phases[:, :n], lengths[:, :n], return_low=True)
                    low_stats[name][0] += (learned_low[:, 0]-low).abs().sum().item()
                    low_stats[name][1] += low.numel()
                else:
                    outputs[name] = model(tokens[:, :n], phases[:, :n], lengths[:, :n])
            errors = {}
            for name, output in outputs.items():
                difference = output-target
                errors[name] = difference.abs().mean((1, 2, 3)).cpu().tolist()
                stats[name]["abs"] += difference.abs().sum().item()
                stats[name]["sq"] += difference.square().sum().item()
                stats[name]["count"] += difference.numel()
            for i, (frame, block) in enumerate(keys):
                block_rows.append({"frame": frame, "block": block,
                                   **{name: values[i]*255 for name, values in errors.items()}})
                for name in models:
                    if errors[name][i] < min(errors["nearest"][i], errors["bilinear"][i]):
                        wins[name] += 1
                if frame == args.preview_frame:
                    frames["target"][block] = target[i].cpu()
                    for name, output in outputs.items():
                        frames[name][block] = output[i].cpu()
                if video is not None:
                    if pending_video and pending_frame != frame:
                        raise ValueError("video encountered an incomplete frame")
                    pending_frame = frame
                    pending_video["target"][block] = target[i].cpu()
                    for name in video_names[1:]:
                        pending_video[name][block] = outputs[name][i].cpu()
                    if block == 29:
                        video.write(frame, pending_video)
                        if video.frames % 30 == 0:
                            print(f"Rendered {video.frames} comparison frames", flush=True)
                        pending_video = defaultdict(dict)
        if video is not None:
            if pending_video:
                raise ValueError("incomplete final video frame")
            video_frames = video.frames
    metrics = {name: {"mae_8bit": value["abs"]/value["count"]*255,
                      "psnr_raw": -10*np.log10(max(value["sq"]/value["count"], 1e-12))}
               for name, value in stats.items()}
    gates = {}
    for name in models:
        passed = all(metrics[name]["mae_8bit"] < metrics[b]["mae_8bit"] and
                     metrics[name]["psnr_raw"] > metrics[b]["psnr_raw"] for b in ("nearest", "bilinear"))
        gates[name] = {"numerical_screen_passed": bool(passed),
                       "block_win_rate_against_both": wins[name]/len(dataset)}
    report = {"samples": len(dataset), "start_frame": args.start_frame, "stop_frame": args.stop_frame,
              "first_evaluated_frame": dataset.keys[0][0], "last_evaluated_frame": dataset.keys[-1][0],
              "metrics": metrics, "gates": gates,
              "learned_low_vs_decoded_mae_8bit": {name: v[0]/v[1]*255 for name, v in low_stats.items()},
              "scope": "Same-video holdout only; visual, unseen-video and streaming gates pending.",
              "preview_order": ["target", *stats.keys()],
              "baseline": "Phase-aligned RGGB planes, border padding; nearest and bilinear",
              "decoded_cache": cache.metadata,
              "checkpoints": [str(p.resolve()) for p in args.checkpoint]}
    report["input_contracts"] = {name: c.get("input_contract", "encoded_only_legacy")
                                 for name, c in configs.items()}
    if args.video:
        report["video"] = {"file": "comparison.mp4", "frames": video_frames, "fps": 30,
                           "panel_order": video_names, "note": "Offline rendered; not a realtime benchmark."}
        report["video"]["temporal_error_change_mae_8bit"] = {
            name: video.temporal_sum[name]/count*255 for name, count in video.temporal_count.items()}
        report["video"]["temporal_metric_note"] = (
            "Mean absolute change of prediction-minus-target error between consecutive frames; "
            "no optical-flow alignment. Diagnostic only, not a standalone flicker acceptance criterion.")
    (args.out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out/"blocks.jsonl").open("w", encoding="utf-8") as stream:
        for row in block_rows:
            stream.write(json.dumps(row)+"\n")
    for name in report["preview_order"]:
        if set(frames[name]) == set(range(30)):
            mosaic = torch.cat([frames[name][b] for b in range(30)], dim=1)[None]
            rgb = demosaic(mosaic)[0].permute(1, 2, 0).clamp(0, 1).numpy()
            Image.fromarray((rgb*255).round().astype(np.uint8)).save(args.out/f"{name}.png")
    names = ["target", "bilinear", *models]
    if all((args.out/f"{name}.png").exists() for name in names):
        sheet = Image.new("RGB", (1280, ((len(names)+1)//2)*508), "white")
        draw = ImageDraw.Draw(sheet)
        for index, name in enumerate(names):
            x, y = index % 2*640, index//2*508
            draw.text((x+8, y+8), name, fill="black")
            with Image.open(args.out/f"{name}.png") as panel:
                sheet.paste(panel, (x, y+28))
        sheet.save(args.out/"comparison.png")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    from record_experiment import run_logged
    run_logged("evaluation", main)
