"""Evaluate V10 generation with fixed seeds/gammas and explicit source provenance."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from baseline import DecodedCache, fingerprint
from evaluate import demosaic
from evaluate_fsr import RgbVideo, summarize, update_stats
from fsr_reference import read_atlas, rgb_to_raw
from network import build_model
from record_experiment import run_logged
from v10.data import V10Dataset, stratified_indices
from v10.diffusion import Schedule, anchor_image, compose, generate_residual, high_pass, keyed_noise, low_pass
from v10.model import load_detail_anchor, load_v10, make_bilinear


class FsrReference:
    """Read already-recorded official FSR atlas outputs, never regenerate from labels."""
    def __init__(self, report_path, cache):
        self.path = Path(report_path)
        self.report = json.loads(self.path.read_text(encoding="utf-8"))
        if self.report["decoded_cache"]["database_sha256"] != cache.metadata["database_sha256"]:
            raise ValueError("FSR reference belongs to a different DB")
        rows = [json.loads(line) for line in (self.path.parent/"blocks.jsonl").read_text(encoding="utf-8").splitlines()]
        self.indices = {(r["frame"], r["block"]): i for i, r in enumerate(rows)}
        if len(self.indices) != len(rows):
            raise ValueError("duplicate reference keys")
        self.count, self.batch = len(rows), self.report["fsr"]["tiles_per_atlas"]
        self.cache_by_variant = {}

    def get(self, keys, device, variant="rcas"):
        if variant not in ("rcas", "easu4"):
            raise ValueError("unknown FSR atlas variant")
        values = []
        for key in keys:
            i = self.indices[tuple(key)]
            atlas, offset = divmod(i, self.batch)
            cached_index, cached = self.cache_by_variant.get(variant, (None, None))
            if atlas != cached_index:
                cached = read_atlas(self.path.parent/"official-cli"/f"{variant}-{atlas:03d}.png",
                                    min(self.batch, self.count-atlas*self.batch))
                self.cache_by_variant[variant] = (atlas, cached)
            values.append(cached[offset])
        return torch.stack(values).to(device)


class SingleVideo:
    def __init__(self, folder, name="generated"):
        self.frames = 0
        self.log = (folder/f"{name}-ffmpeg.log").open("wb")
        self.process = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480", "-r", "30", "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(folder/f"{name}.mp4")], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=self.log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def write(self, strips):
        rgb = torch.cat([strips[b] for b in range(30)], dim=1).permute(1, 2, 0)
        self.process.stdin.write((rgb.clamp(0, 1).numpy()*255).round().astype(np.uint8).tobytes())
        self.frames += 1

    def close(self):
        self.process.stdin.close()
        code = self.process.wait()
        self.log.close()
        if code:
            raise RuntimeError("single-video ffmpeg failed")


@torch.inference_mode()
def calibrate_noise_control(model, schedule, dataset, cache, device, config, steps, anchor=None):
    indices = stratified_indices(dataset.keys, min(64, len(dataset)))
    keys = [dataset.keys[i] for i in indices]
    generated_energy, unit_energy = torch.zeros(4, device=device), torch.zeros(4, device=device)
    for first in range(0, len(keys), 8):
        subset = keys[first:first+8]
        decoded = torch.stack([torch.stack([cache.get((f-k, b)) for k in (0, 1)]) for f, b in subset]).to(device)
        phases = torch.tensor([[f%16, (f-1)%16] for f, _ in subset], device=device)
        high = make_bilinear(decoded, phases)
        base = anchor_image(anchor, decoded, high, phases)
        residual = generate_residual(model, schedule, decoded, high, phases, subset, seed=0,
            video_id=cache.metadata["database_sha256"], steps=steps, mode=config["mode"], base=base)
        detail = F.pixel_unshuffle(compose(high, residual, model.residual_scale, base=base)-base, 2)
        if anchor is None:
            detail = high_pass(detail)
        unit = keyed_noise(subset, 10000, cache.metadata["database_sha256"], device)
        unit = unit-low_pass(unit)
        generated_energy += detail.square().sum((0, 2, 3))
        unit_energy += unit.square().sum((0, 2, 3))
    amplitude = (generated_energy/unit_energy.clamp_min(1e-12)).sqrt().view(1, 4, 1, 1)
    return amplitude, {"keys": [list(k) for k in keys], "amplitude": amplitude.flatten().cpu().tolist(),
                       "source": "Fixed train keys; matches gamma1 energy against active base, no target reads",
                       "highpass_applied_for_calibration": anchor is None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--decoded-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--control-checkpoint", type=Path)
    parser.add_argument("--fsr-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=720)
    parser.add_argument("--stop-frame", type=int)
    parser.add_argument("--preview-frame", type=int, default=750)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-policy", choices=("per_frame", "fixed_screen"), default="per_frame")
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-gamma", type=float, default=1.)
    parser.add_argument("--detail-probe", action="store_true",
                        help="compare frozen C detail and labeled oracle detail on the same B base")
    args = parser.parse_args()
    if (args.batch_size < 1 or 1. not in args.gammas or len(set(args.gammas)) != len(args.gammas)
            or any(not math.isfinite(g) or g < 0 for g in args.gammas)):
        parser.error("positive batch size and distinct nonnegative gammas including 1 required")
    if args.out.exists():
        raise ValueError("select new output directory")
    if args.detail_probe and not args.control_checkpoint:
        parser.error("--detail-probe requires --control-checkpoint")
    if args.video_gamma not in args.gammas:
        parser.error("--video-gamma must be included in --gammas")
    torch.set_num_threads(4)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    model, saved = load_v10(args.checkpoint, device)
    anchor = load_detail_anchor(saved, device)
    if anchor is not None and args.detail_probe:
        parser.error("--detail-probe compares the legacy C residual with B")
    config = saved["config"]
    cache = DecodedCache(args.decoded_cache, args.db)
    if cache.metadata["header_id"] != config["header_id"]:
        raise ValueError("checkpoint/header mismatch")
    dataset = V10Dataset(args.db, cache, args.start_frame, args.stop_frame)
    schedule = Schedule(config["timesteps"]).to(device)
    fsr = FsrReference(args.fsr_report, cache) if args.fsr_report else None
    reference, reference_saved = None, None
    if args.reference_checkpoint:
        reference_saved = torch.load(args.reference_checkpoint, weights_only=True, map_location="cpu")
        rc = reference_saved["config"]
        if (rc["architecture"] != "v9" or rc["history"] != 2 or rc.get("v9_upper_strip", False)
                or rc["header_id"] != config["header_id"]):
            raise ValueError("reference must be compatible V9 history2 without upper strip")
        reference = build_model(rc).to(device).eval()
        reference.load_state_dict(reference_saved["model"])
    control, control_saved = (load_v10(args.control_checkpoint, device) if args.control_checkpoint else (None, None))
    if control_saved and (control_saved["config"]["mode"] != "deterministic" or
                          control_saved["config"]["header_id"] != config["header_id"]):
        raise ValueError("control must be deterministic V10 with compatible header")
    args.out.mkdir(parents=True)
    calibration = None
    # Calibration is valid only on the training database used by this model.
    if cache.metadata["database_sha256"] == config["database_sha256"]:
        train = V10Dataset(args.db, cache, frame_start=config["split"].get("train_start_frame", 0),
                           frame_stop=config["split"]["boundary"])
        amplitude, calibration = calibrate_noise_control(model, schedule, train, cache, device, config, args.steps, anchor)
    else:
        raise ValueError("independent-source evaluation needs a saved training-only noise calibration")
    generation_names = {g: f"v10-gamma{g:g}" for g in args.gammas}
    primary = generation_names[1.]
    video_method = generation_names[args.video_gamma]
    noise_name = "b-plus-matched-noise" if anchor is not None else "bilinear-highpass-noise"
    names = ["target", "bilinear", *generation_names.values(), "bilinear-sharpen", noise_name]
    scaled_noise_names = {g: f"b-plus-matched-noise-gamma{g:g}" for g in args.gammas if g not in (0., 1.)}
    if anchor is not None:
        names.append("frozen-b-anchor")
        names.extend(scaled_noise_names.values())
    if reference:
        names.append("v9-reference")
    if control:
        names.append("v10-deterministic-control")
        control_schedule = Schedule(control_saved["config"]["timesteps"]).to(device)
    detail_names = ("b-plus-c-detail", "b-plus-matched-noise", "b-plus-reference-detail")
    if args.detail_probe:
        names.extend(detail_names)
    if fsr:
        names.append("official-fsr1-rcas0")
    raw_stats, rgb_stats = defaultdict(lambda: [0., 0., 0]), defaultdict(lambda: [0., 0., 0])
    detail_stats = defaultdict(lambda: [0., 0., 0])
    known_stats = defaultdict(lambda: [0., 0])
    seam_stats = defaultdict(lambda: [0., 0, 0., 0])
    temporal_stats = defaultdict(lambda: [0., 0, 0])
    previous_errors = {}
    rows, offset, preview = [], 0, defaultdict(dict)
    pending, raw_pending = defaultdict(dict), defaultdict(dict)
    with ExitStack() as cleanup, torch.inference_mode():
        video = RgbVideo(args.out, names) if args.video else None
        single = SingleVideo(args.out) if args.video else None
        if video:
            cleanup.callback(video.close)
            cleanup.callback(single.close)
        for decoded, phases, target, keys_tensor in DataLoader(dataset, batch_size=args.batch_size):
            keys = [tuple(k) for k in keys_tensor.tolist()]
            offset += len(keys)
            decoded, phases, target = decoded.to(device), phases.to(device), target.to(device)
            high = make_bilinear(decoded, phases)
            base = anchor_image(anchor, decoded, high, phases)
            residual = generate_residual(model, schedule, decoded, high, phases, keys,
                seed=args.seed, video_id=cache.metadata["database_sha256"], steps=args.steps,
                policy=args.noise_policy, mode=config["mode"], base=base)
            base_planes = F.pixel_unshuffle(high[:, 0], 2)
            unit = keyed_noise(keys, args.seed+10000, cache.metadata["database_sha256"], device, args.noise_policy)
            noise = amplitude*high_pass(unit)
            active_planes = F.pixel_unshuffle(base, 2)
            outputs = {"bilinear": high[:, 0],
                       **{name: compose(high, residual, model.residual_scale, g, base=base) for g, name in generation_names.items()},
                       "bilinear-sharpen": F.pixel_shuffle(base_planes+0.5*(base_planes-low_pass(base_planes)), 2).clamp(0, 1),
                       noise_name: F.pixel_shuffle(active_planes+noise, 2).clamp(0, 1)}
            if anchor is not None:
                outputs["frozen-b-anchor"] = base
                outputs.update({name: F.pixel_shuffle(active_planes+g*noise, 2).clamp(0, 1)
                                for g, name in scaled_noise_names.items()})
            if reference:
                tokens = torch.zeros(len(keys), 2, 640, dtype=torch.long, device=device)
                lengths = torch.zeros(len(keys), 2, dtype=torch.long, device=device)
                outputs["v9-reference"] = reference(tokens, phases, lengths, decoded)
            if control:
                cr = generate_residual(control, control_schedule, decoded, high, phases, keys,
                                       mode="deterministic", steps=1)
                control_base = compose(high, cr, control.residual_scale)
                outputs["v10-deterministic-control"] = control_base
                if args.detail_probe:
                    anchor_planes = F.pixel_unshuffle(control_base, 2)
                    c_detail = F.pixel_unshuffle(outputs[primary]-high[:, 0], 2)
                    oracle_detail = F.pixel_unshuffle(target-control_base, 2)
                    additions = (c_detail-low_pass(c_detail), amplitude*(unit-low_pass(unit)),
                                 oracle_detail-low_pass(oracle_detail))
                    outputs.update({name: F.pixel_shuffle(anchor_planes+addition, 2).clamp(0, 1)
                                    for name, addition in zip(detail_names, additions)})
            rgb_outputs = {name: demosaic(value).clamp(0, 1) for name, value in outputs.items()}
            if fsr:
                rgb_outputs["official-fsr1-rcas0"] = fsr.get(keys, device)
                outputs["official-fsr1-rcas0"] = rgb_to_raw(rgb_outputs["official-fsr1-rcas0"])
            truth_rgb = demosaic(target).clamp(0, 1)
            errors = {n: update_stats(raw_stats, n, o, target) for n, o in outputs.items()}
            rgb_errors = {n: update_stats(rgb_stats, n, o, truth_rgb) for n, o in rgb_outputs.items()}
            for name, output in outputs.items():
                diff = F.pixel_unshuffle(output-target, 2)
                smooth = low_pass(diff)
                stat = detail_stats[name]
                stat[0] += smooth.abs().sum().item()
                stat[1] += (diff-smooth).abs().sum().item()
                stat[2] += diff.numel()
                yy = torch.arange(8, device=device)[None, :, None]
                xx = torch.arange(320, device=device)[None, None, :]
                phase = phases[:, 0]
                mask = (((yy-phase[:, None, None]//4)%4 == 0) &
                        ((xx-phase[:, None, None]%4)%4 == 0))[:, None]
                known_stats[name][0] += (diff.abs()*mask).sum().item()
                known_stats[name][1] += int(mask.sum().item())*4
            for i, (frame, block) in enumerate(keys):
                rows.append({"frame": frame, "block": block, "phase": int(phases[i, 0]),
                    "raw_projection_mae": {n: e[i]*255 for n, e in errors.items()},
                    "rgb_reference_mae": {n: e[i]*255 for n, e in rgb_errors.items()}})
                methods = {"target": truth_rgb[i].cpu(), **{n: value[i].cpu() for n, value in rgb_outputs.items()}}
                if frame == args.preview_frame:
                    for name, value in methods.items():
                        preview[name][block] = value
                for name, value in outputs.items():
                    raw_pending[name][block] = (value[i]-target[i]).cpu()
                if block == 29:
                    if all(len(v) == 30 for v in raw_pending.values()):
                        for name, blocks in raw_pending.items():
                            full = torch.cat([blocks[b] for b in range(30)], dim=1)[None]
                            previous = previous_errors.get(name)
                            if previous is not None and previous[0] == frame - 1:
                                sums = temporal_stats[name]
                                sums[0] += (full-previous[1]).abs().sum().item()
                                sums[1] += full.numel()
                                sums[2] += 1
                            previous_errors[name] = frame, full
                            planes = F.pixel_unshuffle(full, 2)[0]
                            jumps = (planes[:, 1:]-planes[:, :-1]).abs()
                            boundary = torch.arange(1, planes.shape[1])%8 == 0
                            values = seam_stats[name]
                            values[0] += jumps[:, boundary].sum().item()
                            values[1] += jumps[:, boundary].numel()
                            values[2] += jumps[:, ~boundary].sum().item()
                            values[3] += jumps[:, ~boundary].numel()
                    raw_pending = defaultdict(dict)
                if video:
                    for name, value in methods.items():
                        pending[name][block] = value
                    if block == 29:
                        video.write(frame, pending)
                        single.write(pending[video_method])
                        pending = defaultdict(dict)
                        if video.frames % 30 == 0:
                            print(f"Rendered {video.frames} frames", flush=True)
        if video and pending:
            raise ValueError("incomplete video frame")
    raw_metrics, rgb_metrics = summarize(raw_stats), summarize(rgb_stats)
    paired = {}
    for name in outputs:
        errors = np.array([r["raw_projection_mae"][name] for r in rows])
        paired[name] = {"worst_10_percent_mae": float(np.sort(errors)[-int(np.ceil(len(errors)*0.1)):].mean()),
                        "primary_win_rate": float(np.mean([r["raw_projection_mae"][primary] <
                                                           r["raw_projection_mae"][name] for r in rows]))}
    regressions = sorted([{"frame": r["frame"], "block": r["block"],
                           "delta_vs_bilinear": r["raw_projection_mae"][primary]-r["raw_projection_mae"]["bilinear"]}
                          for r in rows], key=lambda r: r["delta_vs_bilinear"], reverse=True)[:10]
    report = {"samples": len(dataset), "start_frame": args.start_frame, "stop_frame": args.stop_frame,
        "first_evaluated_frame": dataset.keys[0][0], "last_evaluated_frame": dataset.keys[-1][0],
        "scope": "Same-video holdout. Main seed/gamma fixed, not a perceptual or deployment acceptance.",
        "checkpoint": {"path": str(args.checkpoint), "sha256": fingerprint(args.checkpoint), "config": config, "epoch": saved["epoch"]},
        "decoded_cache": cache.metadata, "steps": args.steps, "seed": args.seed, "noise_policy": args.noise_policy,
        "raw_projection_metrics": raw_metrics, "rgb_reference_metrics": rgb_metrics, "primary": primary,
        "paired_comparison": paired, "largest_regressions_vs_bilinear": regressions,
        "frequency_errors": {n: {"low_frequency_mae": a/c*255, "high_frequency_mae": b/c*255}
                             for n, (a, b, c) in detail_stats.items()},
        "noise_control": calibration, "sharpen_control": "fixed CFA-plane unsharp alpha0.5, Gaussian5 sigma1",
        "known_sample_mae": {n: a/c*255 for n, (a, c) in known_stats.items()},
        "seam_diagnostics": {n: {"cfa_error_jump_at_strip_boundaries": a/c*255,
                                  "cfa_error_jump_inside_strips": b/d*255}
                             for n, (a, c, b, d) in seam_stats.items() if c and d},
        "temporal_diagnostics": {n: {"raw_error_change_mae_8bit": total/pixels*255, "frame_pairs": pairs}
                                 for n, (total, pixels, pairs) in temporal_stats.items()},
        "temporal_metric_note": "Adjacent-frame change of prediction error in fixed raw coordinates, without motion compensation.",
        "phase_metrics": {str(p): {n: float(np.mean([r["raw_projection_mae"][n] for r in rows if r["phase"] == p]))
                                    for n in raw_metrics} for p in sorted({r["phase"] for r in rows})},
        "metric_note": "RGB reference is demosaiced target RAW, not original full RGB. No LPIPS or FID measured.",
        "numerical_fsr_passed": bool(raw_metrics[primary]["mae_8bit"] < raw_metrics["official-fsr1-rcas0"]["mae_8bit"] and
                                     raw_metrics[primary]["psnr"] > raw_metrics["official-fsr1-rcas0"]["psnr"]) if fsr else None}
    if fsr:
        report["fsr_reference"] = {"path": str(args.fsr_report), "sha256": fingerprint(args.fsr_report), "reused_official_outputs": True}
    if reference:
        report["v9_reference"] = {"path": str(args.reference_checkpoint), "sha256": fingerprint(args.reference_checkpoint)}
    if control:
        report["deterministic_control"] = {"path": str(args.control_checkpoint), "sha256": fingerprint(args.control_checkpoint),
                                           "config": control_saved["config"]}
    if args.detail_probe:
        report["detail_probe"] = {"generated": detail_names[0], "matched_noise": detail_names[1],
                                  "labeled_oracle": detail_names[2],
                                  "note": "The oracle uses high-pass target-minus-B only for diagnosis."}
    if args.video:
        report["video"] = {"frames": video.frames, "fps": 30, "panel_order": names,
                           "single_video": "generated.mp4", "single_method": video_method}
    (args.out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out/"blocks.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row)+"\n")
    if all(len(preview[n]) == 30 for n in names):
        sheet = Image.new("RGB", (1280, ((len(names)+1)//2)*508), "white")
        draw = ImageDraw.Draw(sheet)
        for i, name in enumerate(names):
            rgb = torch.cat([preview[name][b] for b in range(30)], dim=1).permute(1, 2, 0).numpy()
            panel = Image.fromarray((rgb*255).round().astype(np.uint8))
            panel.save(args.out/f"{name}.png")
            x, y = i%2*640, i//2*508
            sheet.paste(panel, (x, y+28))
            draw.text((x+8, y+8), name, fill="black")
        sheet.save(args.out/"comparison.png")
    print(json.dumps({"samples": len(dataset), "raw": raw_metrics, "rgb": rgb_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    run_logged("evaluation", main)
