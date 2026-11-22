"""Compare official two-pass FSR1 RGB pipeline with Q5 v7/v9, per strip."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from baseline import DecodedCache, fingerprint, upscale
from data import RawDataset, decoded_window
from evaluate import ComparisonVideo, demosaic
from fsr_reference import fsr_passes, low_rgb, make_atlas, read_atlas, rgb_to_raw
from network import build_model


class RgbVideo(ComparisonVideo):
    def write(self, frame, methods):
        if self.last_frame is not None and frame != self.last_frame+1:
            raise ValueError("video requires consecutive frames")
        sheet = Image.new("RGB", self.size, "white")
        draw = ImageDraw.Draw(sheet)
        for index, name in enumerate(self.names):
            if set(methods[name]) != set(range(30)):
                raise ValueError("incomplete frame")
            rgb = torch.cat([methods[name][b] for b in range(30)], dim=1).permute(1, 2, 0)
            x, y = index % 2*640, index//2*508
            draw.text((x+8, y+8), f"{name} | frame {frame}", fill="black")
            sheet.paste(Image.fromarray((rgb.clamp(0, 1).numpy()*255).round().astype(np.uint8)), (x, y+28))
        self.process.stdin.write(sheet.tobytes())
        self.frames += 1
        self.last_frame = frame


def update_stats(stats, name, output, target):
    delta = output-target
    stats[name][0] += delta.abs().sum().item()
    stats[name][1] += delta.square().sum().item()
    stats[name][2] += delta.numel()
    return delta.abs().mean((1, 2, 3)).cpu().tolist()


def summarize(stats):
    return {name: {"mae_8bit": values[0]/values[2]*255,
                   "psnr": float(-10*np.log10(max(values[1]/values[2], 1e-12)))}
            for name, values in stats.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--decoded-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--single-frame-reference", type=Path,
                        help="optional v9 history=1 checkpoint evaluated on identical windows")
    parser.add_argument("--reference-checkpoint", type=Path, help="optional v9 reference without upper-strip context")
    parser.add_argument("--echosr-lite", action="store_true", help="also test official pretrained RGB EchoSR-Lite x4")
    parser.add_argument("--echosr-root", type=Path, default=Path("Model/tools/EchoSR"))
    parser.add_argument("--echosr-weights", type=Path, default=Path("Model/tools/EchoSR/experiments/EchoSR_x4_lite_DF2K/models/EchoSR_x4_lite_DF2K.pth"))
    parser.add_argument("--fsr-cli", type=Path, default=Path("Model/tools/fsr1/v1.0.3/FidelityFX_CLI.exe"))
    parser.add_argument("--start-frame", type=int, default=720)
    parser.add_argument("--stop-frame", type=int)
    parser.add_argument("--preview-frame", type=int, default=750)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("output directory already exists")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = DecodedCache(args.decoded_cache, args.db)
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    if checkpoint["config"]["header_id"] != cache.metadata["header_id"]:
        raise ValueError("checkpoint and data header mismatch")
    if checkpoint["config"]["history"] not in (1, 2) or checkpoint["config"]["architecture"] not in ("v7", "v9"):
        raise ValueError("this experiment requires one- or two-frame v7/v9")
    history = checkpoint["config"]["history"]
    model = build_model(checkpoint["config"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    echo = None
    echo_name = "EchoSR-Lite-x4-DF2K-pretrained"
    if args.echosr_lite:
        from echosr_adapter import EchoSRLite
        echo = EchoSRLite(args.echosr_root, args.echosr_weights, device)
    reference = reference_checkpoint = None
    reference_name = "v9-single-frame-reference"
    reference_path = args.reference_checkpoint or args.single_frame_reference
    if args.reference_checkpoint and args.single_frame_reference:
        raise ValueError("select one reference checkpoint option")
    if reference_path:
        reference_checkpoint = torch.load(reference_path, weights_only=True, map_location="cpu")
        reference_config = reference_checkpoint["config"]
        if (reference_config["architecture"] != "v9" or reference_config["history"] not in (1, 2) or
                reference_config["history"] > history or reference_config.get("v9_upper_strip", False) or
                (args.single_frame_reference and reference_config["history"] != 1) or
                reference_config["header_id"] != cache.metadata["header_id"]):
            raise ValueError("reference must be v9 without upper strip, compatible history and codec header")
        if reference_config["history"] == 2:
            reference_name = "v9-two-frame-reference"
        reference = build_model(reference_config).to(device).eval()
        reference.load_state_dict(reference_checkpoint["model"])
    model_name = ("v9-one-layer" if checkpoint["config"].get("v9_layers", 2) == 1
                  else "v9-" + checkpoint["config"].get("v9_variant", "relu8"))
    if checkpoint["config"].get("v9_layers", 2) > 2:
        model_name += "-layers" + str(checkpoint["config"]["v9_layers"])
    if checkpoint["config"].get("v9_kernel_width", 3) != 3:
        model_name += "-k3x" + str(checkpoint["config"]["v9_kernel_width"])
    if checkpoint["config"]["architecture"] == "v7":
        model_name = "v7"
    if history == 2:
        model_name += "-history2"
    upper_strip = checkpoint["config"].get("v9_upper_strip", False)
    if upper_strip:
        model_name += "-upper"
    dataset = RawDataset(args.db, history, args.start_frame, args.stop_frame, v9_upper_strip=upper_strip)
    if not len(dataset):
        raise ValueError("empty evaluation window")
    batch_size = 128
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    folder = args.out/"official-cli"
    folder.mkdir(parents=True)
    # Inputs use only decoded data and phase, never the target tensor.
    inputs, offset = [], 0
    with torch.inference_mode():
        for index, (_, phases, _, target) in enumerate(loader):
            keys = dataset.keys[offset:offset+len(target)]
            offset += len(target)
            low = torch.stack([cache.get(key) for key in keys]).to(device)
            path = folder/f"input-{index:03d}.png"
            make_atlas(low_rgb(low, phases[:, 0].to(device))).save(path)
            inputs.append(path)
    print(f"Prepared {len(dataset)} independent RGB strips in {len(inputs)} atlases", flush=True)
    easu_files, rcas_files = fsr_passes(args.fsr_cli, inputs, folder, sharpness=0.0)
    raw_stats, rgb_stats = defaultdict(lambda: [0., 0., 0]), defaultdict(lambda: [0., 0., 0])
    names = ["target-rgb-reference", "bilinear-raw", "bilinear-rgb-input", "fsr1-easu-2x2x", "fsr1-easu-2x2x-rcas0", model_name]
    if reference is not None:
        names.append(reference_name)
    if echo is not None:
        names.append(echo_name)
    preview, pending = defaultdict(dict), defaultdict(dict)
    offset, rows = 0, []
    with ExitStack() as cleanup, torch.inference_mode():
        video = RgbVideo(args.out, names) if args.video else None
        if video:
            cleanup.callback(video.close)
        for index, (tokens, phases, lengths, target) in enumerate(loader):
            keys = dataset.keys[offset:offset+len(target)]
            offset += len(target)
            tokens, phases, lengths, target = [t.to(device) for t in (tokens, phases, lengths, target)]
            low = torch.stack([cache.get(key) for key in keys]).to(device)
            window = torch.stack([decoded_window(cache, dataset.input_keys(f, b))
                                  for f, b in keys]).to(device)
            raw_outputs = {"bilinear-raw": upscale(low, phases[:, 0]),
                           model_name: model(tokens, phases, lengths, window)}
            if reference is not None:
                n = reference_config["history"]
                raw_outputs[reference_name] = reference(tokens[:, :n], phases[:, :n], lengths[:, :n], window[:, :n])
            # Match PNG8 input and intermediate quantization for the RGB bilinear
            # control, so EASU-vs-bilinear shares the same input conversion budget.
            source_rgb = (low_rgb(low, phases[:, 0])*255).round()/255
            half = F.interpolate(source_rgb, scale_factor=2, mode="bilinear", align_corners=False)
            half = (half*255).round()/255
            rgb_outputs = {"bilinear-rgb-input": (F.interpolate(half, scale_factor=2, mode="bilinear", align_corners=False)*255).round()/255,
                           "fsr1-easu-2x2x": read_atlas(easu_files[index], len(target)).to(device),
                           "fsr1-easu-2x2x-rcas0": read_atlas(rcas_files[index], len(target)).to(device)}
            if echo is not None:
                # Process strips independently, in small batches to fit the GTX 1650.
                rgb_outputs[echo_name] = torch.cat([echo(chunk) for chunk in source_rgb.split(16)])
            for name, rgb in rgb_outputs.items():
                raw_outputs[name] = rgb_to_raw(rgb)
            for name in (["bilinear-raw", model_name] + ([reference_name] if reference is not None else [])):
                rgb_outputs[name] = demosaic(raw_outputs[name]).clamp(0, 1)
            target_rgb = demosaic(target).clamp(0, 1)
            raw_errors = {name: update_stats(raw_stats, name, value, target) for name, value in raw_outputs.items()}
            rgb_errors = {name: update_stats(rgb_stats, name, value, target_rgb) for name, value in rgb_outputs.items()}
            for i, (frame, block) in enumerate(keys):
                rows.append({"frame": frame, "block": block,
                             "raw_projection_mae": {name: v[i]*255 for name, v in raw_errors.items()},
                             "rgb_reference_mae": {name: v[i]*255 for name, v in rgb_errors.items()}})
                methods = {names[0]: target_rgb[i].cpu(), **{name: rgb_outputs[name][i].cpu() for name in names[1:]}}
                if frame == args.preview_frame:
                    for name, image in methods.items():
                        preview[name][block] = image
                if video:
                    for name, image in methods.items():
                        pending[name][block] = image
                    if block == 29:
                        video.write(frame, pending)
                        pending = defaultdict(dict)
                        if video.frames % 30 == 0:
                            print(f"Rendered {video.frames} frames", flush=True)
        if video and pending:
            raise ValueError("incomplete final frame")
    report = {"samples": len(dataset), "start_frame": args.start_frame, "stop_frame": args.stop_frame,
              "first_evaluated_frame": dataset.keys[0][0], "last_evaluated_frame": dataset.keys[-1][0],
              "raw_projection_metrics": summarize(raw_stats), "rgb_reference_metrics": summarize(rgb_stats),
              "metric_note": "RAW metrics sample FSR RGB at RGGB positions. RGB target is demosaiced RAW, not original full RGB. Do not mix these metric spaces.",
              "scope": "Same-video holdout, per-strip; no retraining. Unseen-video, full visual and streaming acceptance pending.",
              "checkpoint": {"path": str(args.checkpoint), "sha256": fingerprint(args.checkpoint), "config": checkpoint["config"]},
              "decoded_cache": cache.metadata,
              "fsr": {"name": "AMD official FidelityFX CLI v1.0.3", "executable": str(args.fsr_cli),
                      "sha256": fingerprint(args.fsr_cli), "source": "https://github.com/GPUOpen-Effects/FidelityFX-CLI/releases/tag/v1.0.3",
                      "passes": ["EASU 2x", "EASU 2x", "RCAS sharpness 0.0"], "texture": "FP16", "intermediate": "PNG8 display-encoded RGB",
                      "guard_input_pixels": 8, "atlas_columns": 16, "tiles_per_atlas": batch_size,
                      "note": "Two EASU passes reach 4x per side. This is not a single-pass 2x Performance preset. Input is phase-aware RGB reconstruction of decoded CFA; no full-frame or neighbouring strip context."}}
    if args.video:
        report["video"] = {"frames": video.frames, "fps": 30, "panel_order": names, "note": "Offline, not a latency benchmark"}
    if echo is not None:
        import shutil
        report["echosr_lite"] = echo.metadata
        archive = args.out/"echosr-source"
        archive.mkdir()
        for source in (args.echosr_root/"basicsr/archs/EchoSR_arch.py", args.echosr_root/"LICENSE",
                       args.echosr_root/"options/EchoSR_test/test_EchoSR_lite_SRx4_DF2K.yml"):
            shutil.copyfile(source, archive/source.name)
        report["echosr_paired_comparison"] = {}
        for name in raw_stats:
            errors = np.array([row["raw_projection_mae"][name] for row in rows])
            report["echosr_paired_comparison"][name] = {
                "worst_10_percent_mae": float(np.sort(errors)[-int(np.ceil(len(errors)*0.1)):].mean()),
                "echosr_win_rate": float(np.mean([r["raw_projection_mae"][echo_name] <
                                                  r["raw_projection_mae"][name] for r in rows]))}
        report["echosr_largest_regressions_vs_fsr"] = sorted(
            [{"frame": r["frame"], "block": r["block"], "echosr_mae": r["raw_projection_mae"][echo_name],
              "fsr_mae": r["raw_projection_mae"]["fsr1-easu-2x2x-rcas0"],
              "delta": r["raw_projection_mae"][echo_name]-r["raw_projection_mae"]["fsr1-easu-2x2x-rcas0"]} for r in rows],
            key=lambda r: r["delta"], reverse=True)[:10]
    if reference is not None:
        report["reference_checkpoint"] = {"path": str(reference_path),
                                             "sha256": fingerprint(reference_path),
                                             "config": reference_checkpoint["config"]}
        report["paired_comparison"] = {}
        for name in raw_stats:
            errors = np.array([row["raw_projection_mae"][name] for row in rows])
            report["paired_comparison"][name] = {
                "worst_10_percent_mae": float(np.sort(errors)[-int(np.ceil(len(errors)*0.1)):].mean()),
                "model_win_rate": float(np.mean([
                    row["raw_projection_mae"][model_name] < row["raw_projection_mae"][name] for row in rows]))}
        regressions = sorted(rows, key=lambda row: row["raw_projection_mae"][model_name]
                             - row["raw_projection_mae"][reference_name], reverse=True)
        report["largest_regressions_vs_reference"] = [
            {"frame": row["frame"], "block": row["block"],
             "model_mae": row["raw_projection_mae"][model_name],
             "reference_mae": row["raw_projection_mae"][reference_name],
             "delta": row["raw_projection_mae"][model_name]-row["raw_projection_mae"][reference_name]}
            for row in regressions[:10]]
        report["upper_boundary_comparison"] = {}
        for group, selected in (("first_strip", [r for r in rows if r["block"] == 0]),
                                ("remaining_strips", [r for r in rows if r["block"] > 0])):
            if selected:
                report["upper_boundary_comparison"][group] = {
                    "samples": len(selected),
                    "model_mae": float(np.mean([r["raw_projection_mae"][model_name] for r in selected])),
                    "reference_mae": float(np.mean([r["raw_projection_mae"][reference_name] for r in selected])),
                    "win_rate": float(np.mean([r["raw_projection_mae"][model_name] <
                                               r["raw_projection_mae"][reference_name] for r in selected]))}
    (args.out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out/"blocks.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row)+"\n")
    if all(len(preview[name]) == 30 for name in names):
        sheet = Image.new("RGB", (1280, ((len(names)+1)//2)*508), "white")
        draw = ImageDraw.Draw(sheet)
        for index, name in enumerate(names):
            rgb = torch.cat([preview[name][b] for b in range(30)], dim=1).permute(1, 2, 0).numpy()
            panel = Image.fromarray((rgb*255).round().astype(np.uint8))
            panel.save(args.out/f"{name}.png")
            x, y = index % 2*640, index//2*508
            draw.text((x+8, y+8), name, fill="black")
            sheet.paste(panel, (x, y+28))
        sheet.save(args.out/"comparison.png")
    print(json.dumps({k: report[k] for k in ("samples", "raw_projection_metrics", "rgb_reference_metrics")}, indent=2), flush=True)


if __name__ == "__main__":
    from record_experiment import run_logged
    run_logged("evaluation", main)
