"""Evaluate frozen v13 checkpoints and export separate full-size videos."""
import argparse
from contextlib import ExitStack, closing
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from evaluate import demosaic
from evaluate_fsr import summarize, update_stats
from evaluate_v10 import SingleVideo
from record_experiment import digest, run_logged, write_json
from v11.data import connect
from v12 import nearest_bayer_bytes
from v13.data import SPLITS, fingerprints, labels, load_inputs, raw_frame
from v13.ensemble import EnsembleGate
from v13.model import ResidualCNN, mix_residual_bands, quantize


FIXED = {"S1-dev": (736, 743, 750), "S2-dev": (736, 750, 779),
         "S2-audit": (795, 844, 874, 899)}
ROI = Path(__file__).resolve().parent / "v13" / "roi-bbb-q5.json"


def unsharp(raw):
    packed = F.pixel_unshuffle(raw, 2)
    axis = torch.arange(-2, 3, dtype=raw.dtype, device=raw.device)
    g = torch.exp(-axis.square() / 2)
    g /= g.sum()
    weights = (g[:, None] * g).expand(4, 1, 5, 5)
    smoother = F.conv2d(F.pad(packed, (2, 2, 2, 2), mode="replicate"), weights, groups=4)
    return F.pixel_shuffle(packed + 0.5 * (packed - smoother), 2)


def verify_video(path, count):
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-xerror", "-i", str(path),
                             "-map", "0:v:0", "-progress", "pipe:1", "-nostats", "-f", "null", "-"],
                            check=True, capture_output=True, text=True)
    streams = [line for line in result.stderr.splitlines() if "Video:" in line]
    if not streams or not re.search(r"\b640x480\b.*\b30 fps\b", streams[0]):
        raise ValueError(f"unexpected encoded stream: {streams}")
    frames = re.findall(r"^frame=(\d+)\s*$", result.stdout, re.MULTILINE)
    if not frames or int(frames[-1]) != count:
        raise ValueError(f"{path}: expected {count} frames, read {frames}")


def emit(out, name, strips, first, first_valid, fixed):
    full_name, valid_name = name, f"{name}-valid"
    with ExitStack() as stack:
        full = SingleVideo(out, full_name)
        stack.callback(full.close)
        valid = SingleVideo(out, valid_name)
        stack.callback(valid.close)
        for index, rgb in enumerate(strips):
            frame = first + index
            per_strip = dict(enumerate(rgb))
            full.write(per_strip)
            if frame >= first_valid:
                valid.write(per_strip)
            if frame in fixed:
                image = torch.cat(list(rgb), dim=1).permute(1, 2, 0).numpy()
                Image.fromarray(np.rint(image.clip(0, 1) * 255).astype(np.uint8)).save(out / f"{name}-frame-{frame}.png")
    verify_video(out / f"{full_name}.mp4", len(strips))
    verify_video(out / f"{valid_name}.mp4", len(strips) - (first_valid - first))


def display(strips):
    rgb = torch.cat([demosaic(batch.float() / 255) for batch in strips.split(120)])
    return rgb.reshape(-1, 30, 3, 16, 640)


def roi_scores(error, rgb, reference, regions):
    rgb = rgb.permute(0, 2, 1, 3, 4).reshape(-1, 3, 480, 640)
    reference = reference.permute(0, 2, 1, 3, 4).reshape_as(rgb)
    result = {}
    for name, (x0, y0, x1, y1) in regions.items():
        crop = error[:, :, y0:y1, x0:x1]
        color = rgb[:, :, y0:y1, x0:x1] - reference[:, :, y0:y1, x0:x1]
        cfa = F.pixel_unshuffle(error[:, :, y0 + y0 % 2:y1 - y1 % 2, x0 + x0 % 2:x1 - x1 % 2], 2)
        result[name] = {
            "raw_mae": crop.abs().mean().item(),
            "rgb_mae": color.abs().mean().item() * 255,
            "temporal": crop.diff(dim=0).abs().mean().item(),
            "gradient_mae": (cfa.diff(dim=-1).abs().mean() + cfa.diff(dim=-2).abs().mean()).item() / 2,
        }
    return result


def scores(pred, truth, rgb, reference, regions):
    raw_stats, rgb_stats = {"x": [0., 0., 0]}, {"x": [0., 0., 0]}
    per_key = []
    for output, target in zip(pred.split(120), truth.split(120)):
        per_key.extend(update_stats(raw_stats, "x", output.float() / 255, target.float() / 255))
    for output, target in zip(rgb.split(4), reference.split(4)):
        update_stats(rgb_stats, "x", output.flatten(0, 1), target.flatten(0, 1))
    error = pred.float() - truth
    mosaic = error.reshape(-1, 1, 480, 640)
    packed = F.pixel_unshuffle(mosaic, 2)
    dx, dy = packed.diff(dim=-1), packed.diff(dim=-2)
    boundary_rows = slice(7, None, 8)
    is_boundary = torch.zeros(dy.shape[-2], dtype=torch.bool)
    is_boundary[boundary_rows] = True
    low_sum = high_sum = 0
    for batch in error.split(120):
        cfa = F.pixel_unshuffle(batch, 2)
        low = F.avg_pool2d(F.pad(cfa, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
        low_sum += low.abs().sum().item()
        high_sum += (cfa - low).abs().sum().item()
    temporal_pairs = mosaic.diff(dim=0).abs().mean(dim=(1, 2, 3))
    result = {"raw": summarize(raw_stats)["x"], "rgb": summarize(rgb_stats)["x"],
              "exact_raw_fraction": (error == 0).float().mean().item(),
              "pairs": len(temporal_pairs), "temporal": temporal_pairs.mean().item(),
              "gradient_x_mae": dx.abs().mean().item(), "gradient_y_mae": dy.abs().mean().item(),
              "low_frequency_mae": low_sum / error.numel(), "high_frequency_mae": high_sum / error.numel(),
              "boundary_mae": dy[..., is_boundary, :].abs().mean().item(),
              "inside_mae": dy[..., ~is_boundary, :].abs().mean().item(),
              "roi": roi_scores(mosaic, rgb, reference, regions)}
    return result, torch.tensor(per_key, dtype=torch.float64) * 255, temporal_pairs


def load_restorer(checkpoint, variant_device, provenance):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    if config["fingerprints"] != provenance:
        raise ValueError(f"checkpoint belongs to a different input configuration: {checkpoint}")
    model = ResidualCNN(config["variant"], config.get("width", 32), config.get("context", "compact"))
    model = model.to(variant_device, memory_format=torch.channels_last)
    model.load_state_dict(saved["model"])
    model.eval()
    info = {"checkpoint": str(checkpoint.resolve()), "sha256": digest(checkpoint),
            "epoch": saved["epoch"], "variant": config["variant"]}
    return model, info


def model_predictions(checkpoint, x, variant_device, alpha, provenance, band_alpha=None):
    model, info = load_restorer(checkpoint, variant_device, provenance)
    outputs = []
    clipped = delta_square = delta_max = 0
    for cpu_batch in x.split(16):
        batch = cpu_batch.to(variant_device)
        base = batch[:, :1].float() / 255
        raw = model(batch, 1 if band_alpha else alpha)
        if band_alpha:
            raw = mix_residual_bands(base, raw, *band_alpha)
        if not torch.isfinite(raw).all():
            raise ValueError("nonfinite prediction")
        delta = (raw - base) * 255
        clipped += ((raw < 0) | (raw > 1)).sum().item()
        delta_square += delta.square().sum().item()
        delta_max = max(delta_max, delta.abs().max().item())
        outputs.append((quantize(raw) if alpha or band_alpha else batch[:, :1]).cpu())
    pixels = x[:, :1].numel()
    return torch.cat(outputs), {**info, "alpha": alpha, "band_alpha": band_alpha,
                               "clipped": clipped / pixels, "delta_rms": (delta_square / pixels) ** 0.5,
                               "delta_max": delta_max}


@torch.inference_mode()
def raw_predictions(path, x, device, provenance):
    net, info = load_restorer(path, device, provenance)
    return torch.cat([net(batch.to(device)).cpu() for batch in x.split(16)]), info


def ensemble_predictions(paths, x, device, provenance):
    restored, models = [], []
    for path in paths:
        output, info = raw_predictions(path, x, device, provenance)
        restored.append(output)
        models.append(info)
    for name, first, second in (("trend-low-spatial-high", 0, 1), ("spatial-low-trend-high", 1, 0)):
        low = mix_residual_bands(torch.zeros_like(restored[first]), restored[first], 1, 0)
        high = mix_residual_bands(torch.zeros_like(restored[second]), restored[second], 0, 1)
        result = low + high
        yield name, quantize(result), {"low_model": models[first], "high_model": models[second],
                                      "clipped": ((result < 0) | (result > 1)).float().mean().item()}
    avg = (restored[0] + restored[1]) / 2
    yield "equal-weight-model-average", quantize(avg), {"models": models, "weights": [0.5, 0.5],
                                                        "clipped": ((avg < 0) | (avg > 1)).float().mean().item()}


def gated_predictions(path, x, device, provenance):
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = state["config"]
    if config["fingerprints"] != provenance:
        raise ValueError("gate provenance mismatch")
    gate = EnsembleGate().to(device, memory_format=torch.channels_last)
    gate.load_state_dict(state["model"])
    gate.eval()
    restored = []
    for row in config["models"]:
        source = Path(row["checkpoint"])
        if digest(source) != row["sha256"]:
            raise ValueError("fixed source checkpoint changed")
        out, _ = raw_predictions(source, x, device, provenance)
        restored.append(out)
    result = torch.cat([gate(*(part.to(device) for part in batch)).cpu()
                        for batch in zip(*(part.split(16) for part in (x, *restored)))])
    return quantize(result), {"models": config["models"], "gate_sha256": digest(path), "epoch": state["epoch"],
                              "clipped": ((result < 0) | (result > 1)).float().mean().item()}


def validate_audit(args):
    path = args.audit_freeze
    if path is None:
        raise ValueError("audit requires frozen output settings")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze["roi_sha256"] != digest(ROI) or freeze["quantization"] != "round-ties-to-even-raw8":
        raise ValueError("audit configuration differs from pre-audit freeze")
    historical = freeze.get("kind") == "historical-recheck"
    mode = freeze.get("ensemble_mode")
    if args.gated_ensemble or args.band_ensemble:
        if not historical or args.checkpoint or args.band_alpha or not mode:
            raise ValueError("historical ensemble recheck requires one preselected output")
        if args.gated_ensemble:
            if (args.band_ensemble or mode != "gated-model-average"
                    or freeze["gate_sha256"] != digest(args.gated_ensemble)):
                raise ValueError("gate differs from the historical recheck freeze")
        elif (mode not in ("trend-low-spatial-high", "spatial-low-trend-high", "equal-weight-model-average")
              or freeze["checkpoint_sha256s"] != [digest(checkpoint) for checkpoint in args.band_ensemble]):
            raise ValueError("ensemble differs from the historical recheck freeze")
    else:
        if len(args.checkpoint) != 1 or freeze["checkpoint_sha256"] != digest(args.checkpoint[0]):
            raise ValueError("checkpoint differs from the audit freeze")
        if args.band_alpha:
            if not historical or len(args.band_alpha) != 1 or freeze["band_alpha"] != args.band_alpha[0]:
                raise ValueError("band strengths differ from the historical recheck freeze")
        elif len(args.alpha) != 1 or freeze["alpha"] != args.alpha[0]:
            raise ValueError("alpha differs from the audit freeze")
    return digest(path), mode, freeze.get("kind", "original-audit")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", choices=FIXED, required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="*", default=[])
    parser.add_argument("--alpha", type=float, choices=(0., 0.5, 1.), nargs="+", default=[1.])
    parser.add_argument("--band-alpha", type=float, choices=(0., 0.5, 0.75, 1.), nargs=2, action="append", default=[])
    parser.add_argument("--band-ensemble", type=Path, nargs=2, metavar=("TREND", "SPATIAL"))
    parser.add_argument("--gated-ensemble", type=Path)
    parser.add_argument("--audit-freeze", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    is_audit = args.split == "S2-audit"
    freeze_hash, frozen_mode, recheck_kind = validate_audit(args) if is_audit else (None, None, None)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    provenance = fingerprints(args.db, args.scene_plan)
    saved = torch.load(args.cache / f"{args.split}.pt", weights_only=True)
    if saved["provenance"] != provenance or saved["split"] != args.split:
        raise ValueError("incompatible evaluation cache")
    first, last, expected = SPLITS[args.split]
    manifest, x = saved["manifest"], saved["features"]
    first_valid = first + 15
    if len(x) != expected or manifest[0]["frame"] != first_valid or manifest[-1]["frame"] != last:
        raise ValueError("invalid evaluation manifest")
    with closing(connect(args.db)) as conn:
        decoded = load_inputs(conn, first, first_valid - 1)
    warm = nearest_bayer_bytes(torch.stack([row[2] for row in decoded.values()]),
                               torch.tensor([row[1] for row in decoded.values()]))
    target = labels(args.db, [{"frame": frame, "block": block}
                              for frame in range(first, last + 1) for block in range(30)])
    truth = target[len(warm):]
    reference_rgb = display(target)
    regions = json.loads(ROI.read_text(encoding="utf-8"))["regions"]
    emit(args.out, "target", reference_rgb, first, first_valid, FIXED[args.split])
    report = {"split": args.split, "source_range": [first, last], "scored_range": [first_valid, last],
              "samples": expected, "start_frame": first_valid, "scored_strips": expected,
              "warmup_strips": len(warm), "provenance": provenance, "roi_sha256": digest(ROI),
              "audit_freeze_sha256": freeze_hash, "recheck_kind": recheck_kind,
              "seam_cfa_rows": list(range(7, 232, 8)), "methods": {}}
    detail = [dict(row) for row in manifest]
    baseline_keys = baseline_pairs = None

    def evaluate(name, result, info):
        nonlocal baseline_keys, baseline_pairs
        rgb = display(torch.cat((warm, result)))
        numbers, key_scores, pairs = scores(result, truth, rgb[15:], reference_rgb[15:], regions)
        if baseline_keys is None:
            baseline_keys, baseline_pairs = key_scores, pairs
        difference = key_scores - baseline_keys
        numbers["wins_vs_v12"] = int((difference < 0).sum())
        numbers["worst_keys"] = [{"frame": manifest[i]["frame"], "block": manifest[i]["block"],
                                 "delta_mae_8bit": difference[i].item()} for i in difference.argsort(descending=True)[:8]]
        by_pair = pairs - baseline_pairs
        numbers["temporal_pairs_better_than_v12"] = int((by_pair < 0).sum())
        numbers["temporal_worst_pairs"] = [
            {"frames": [first_valid + i, first_valid + i + 1], "delta": by_pair[i].item()}
            for i in by_pair.argsort(descending=True)[:5].tolist()]
        report["methods"][name] = {**info, **numbers}
        for row, score in zip(detail, key_scores.tolist()):
            row[name] = score
        emit(args.out, name, rgb, first, first_valid, FIXED[args.split])
        print(f"{name}: RAW {numbers['raw']['mae_8bit']:.4f}, RGB {numbers['rgb']['mae_8bit']:.4f}, "
              f"temporal {numbers['temporal']:.4f}, seam {numbers['boundary_mae']:.4f}", flush=True)
        write_json(args.out / "report.json", report)

    base = x[:, :1]
    evaluate("v12", base, {})
    evaluate("unsharp", torch.cat([quantize(unsharp(batch.float() / 255)) for batch in base.split(120)]), {})
    for path in args.checkpoint:
        strengths = [] if is_audit and args.band_alpha else args.alpha
        for alpha in strengths:
            result, info = model_predictions(path, x, args.device, alpha, provenance)
            name = f"{path.parent.name}-{path.stem}-alpha{alpha:g}"
            evaluate(name, result, info)
        for low, high in args.band_alpha:
            result, info = model_predictions(path, x, args.device, None, provenance, (low, high))
            name = f"{path.parent.name}-{path.stem}-low{low:g}-high{high:g}"
            evaluate(name, result, info)
    if args.band_ensemble:
        for name, result, info in ensemble_predictions(args.band_ensemble, x, args.device, provenance):
            if not is_audit or name == frozen_mode:
                evaluate(name, result, info)
    if args.gated_ensemble:
        result, info = gated_predictions(args.gated_ensemble, x, args.device, provenance)
        evaluate("gated-model-average", result, info)
    with (args.out / "blocks.jsonl").open("w", encoding="utf-8") as stream:
        for row in detail:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


if __name__ == "__main__":
    run_logged("evaluation", main)
