"""Evaluate V11 checkpoints on common Q5 windows and render paired previews."""
import argparse
from collections import defaultdict
from contextlib import ExitStack, closing
import json
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

from baseline import DecodedCache, upscale
from evaluate import demosaic
from evaluate_fsr import RgbVideo, summarize, update_stats
from evaluate_v10 import FsrReference, SingleVideo
from fsr_reference import rgb_to_raw
from network import build_model
from record_experiment import digest, run_logged
from train_v11 import split_windows
from v11.data import BLOCKS, HEIGHT, WIDTH, connect, read_inputs
from v11.model import FusionNet


def load_models(paths, device):
    models, configs, epochs = {}, {}, {}
    for path in paths:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        config = saved["config"]
        if config.get("input_contract") != "multi_frame_phase_aware_fusion_v2":
            raise ValueError("checkpoint predates the corrected window and coordinate contract")
        anchor = config.get("anchor", "current")
        name = f"v11-t{config['frames']}-{config['alignment']}"
        if anchor != "current":
            name += f"-{anchor}"
        if name in models:
            raise ValueError(f"duplicate method {name}")
        model = FusionNet(config["frames"], config["channels"], config["blocks"],
                          alignment=config["alignment"], anchor=anchor)
        model.load_state_dict(saved["model"])
        models[name], configs[name], epochs[name] = model.to(device).eval(), config, saved["epoch"]
    return models, configs, epochs


def batch_predict(model, decoded, phases, batch_size, with_support=False):
    batches = []
    for first in range(0, len(decoded), batch_size):
        last = first + batch_size
        batches.append(model(decoded[first:last], phases[first:last], return_support=with_support))
    if with_support:
        return tuple(torch.cat(parts) for parts in zip(*batches))
    return torch.cat(batches)


def add_error(stats, name, diff, mask):
    values = diff[mask]
    stats[name][0] += values.abs().sum().item()
    stats[name][1] += values.square().sum().item()
    stats[name][2] += values.numel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fsr-report", type=Path)
    parser.add_argument("--decoded-cache", type=Path)
    parser.add_argument("--reference-checkpoint", type=Path, help="existing Q5 V9 single-frame reference")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frame-range", type=int, nargs=2)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--full-video", action="store_true", help="render one 640x480 candidate across the requested range")
    args = parser.parse_args()
    if ((args.fsr_report or args.reference_checkpoint) and not args.decoded_cache) or args.batch_size < 1:
        parser.error("external references require a decoded cache and positive batch size")
    if args.out.exists():
        raise ValueError("select a new output directory")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    scene_plan = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    models, configs, epochs = load_models(args.checkpoint, device)
    single_name = next((name for name, model in models.items() if model.frames == 1), None)
    if single_name is None:
        raise ValueError("include a T=1 checkpoint for comparison and previews")
    config = configs[single_name]
    fields = ("database_sha256", "scene_plan_sha256", "keys_sha256", "train_range", "validation_range",
              "cohort_frames", "channels", "blocks", "seed", "batch_size", "epochs", "max_steps", "lr",
              "weight_decay", "precision")
    if any(other[field] != config[field] for other in configs.values() for field in fields):
        raise ValueError("models must share source, cohort, and training budget")
    if digest(args.db) != config["database_sha256"] or digest(args.scene_plan) != config["scene_plan_sha256"]:
        raise ValueError("checkpoint does not match the data or scene plan")
    _, validation = split_windows(args.db, config["cohort_frames"], config["train_range"],
                                  config["validation_range"], config["scene_starts"])
    start, stop = args.frame_range or config["validation_range"]
    if not config["validation_range"][0] <= start <= stop <= config["validation_range"][1]:
        raise ValueError("requested frames must be within the saved validation range")
    keys_by_frame = defaultdict(dict)
    for block, frames in validation:
        if start <= frames[-1] <= stop:
            keys_by_frame[frames[-1]][block] = frames
    if not keys_by_frame:
        raise ValueError("no common windows in the requested frame range")
    fsr = None
    cache = DecodedCache(args.decoded_cache, args.db) if args.decoded_cache else None
    if args.fsr_report:
        fsr = FsrReference(args.fsr_report, cache)
    reference = reference_info = None
    if args.reference_checkpoint:
        saved = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=True)
        rc = saved["config"]
        if rc["architecture"] != "v9" or rc["history"] != 1 or rc["header_id"] != cache.metadata["header_id"]:
            raise ValueError("reference must be V9 single-frame with a corresponding header")
        reference = build_model(rc).to(device).eval()
        reference.load_state_dict(saved["model"])
        reference_info = {"path": str(args.reference_checkpoint), "best_epoch": saved["epoch"],
                          "epochs": rc["epochs"], "train_samples": rc["train_samples"],
                          "batch_size": rc["batch_size"], "learning_rate": rc["lr"]}
    args.out.mkdir(parents=True)
    previews = []
    if args.video:
        for clip in scene_plan["previews"]:
            first, last = max(start, clip["start"]), min(stop + 1, clip["stop"])
            if first < last:
                previews.append({**clip, "start": first, "stop": last})
        if not previews:
            raise ValueError("no preview frames in the requested frame range")
    names = list(models) + ["bilinear", "nearest"]
    include_mosaic = config["cohort_frames"] == 16
    mosaic_name = "direct-16-phase"
    if include_mosaic:
        names.append(mosaic_name)
    if fsr:
        names += ["fsr1-rcas0", "fsr1-easu"]
    if reference:
        names.append("v9-reference")
    raw_stats, rgb_stats = defaultdict(lambda: [0., 0., 0]), defaultdict(lambda: [0., 0., 0])
    regions = ("static", "changed", "source-repeat", "source-distinct", "measured")
    region_stats = {region: defaultdict(lambda: [0., 0., 0]) for region in regions}
    known_error = defaultdict(lambda: [0., 0., 0])
    support_stats = defaultdict(lambda: [0, 0, 0])
    seam_errors = defaultdict(lambda: [0., 0])
    temporal_errors = defaultdict(lambda: [0., 0])
    previous_errors, previous_frame = {}, None
    wins = defaultdict(int)
    samples = 0
    full_preview = None
    with ExitStack() as cleanup, torch.inference_mode():
        conn = cleanup.enter_context(closing(connect(args.db)))
        rows = cleanup.enter_context((args.out/"blocks.jsonl").open("w", encoding="utf-8"))
        videos = {}
        video_model = max(reversed(models), key=lambda name: models[name].frames)
        full_video = None
        if args.full_video:
            full_folder = args.out/"full-validation"
            full_folder.mkdir()
            full_video = SingleVideo(full_folder)
            cleanup.callback(full_video.close)
            full_preview = {"start": start, "stop": stop + 1, "method": video_model, "fallback_strips": 0}
        panel_names = ["target", *models, "fsr1-rcas0" if fsr else "bilinear"]
        if include_mosaic:
            panel_names.append(mosaic_name)
        for clip in previews:
            folder = args.out/clip["name"]
            folder.mkdir()
            panel = RgbVideo(folder, panel_names)
            cleanup.callback(panel.close)
            single = SingleVideo(folder)
            cleanup.callback(single.close)
            videos[clip["name"]] = (panel, single)
            clip["fallback_strips"] = 0
        for frame in range(start, stop + 1):
            video_clips = [clip for clip in previews if clip["start"] <= frame < clip["stop"]]
            selected = keys_by_frame.get(frame, {})
            if not selected and not (video_clips or full_video):
                continue
            scored = list(selected)
            blocks = list(range(BLOCKS)) if video_clips or full_video else scored
            primary = [read_inputs(conn, selected.get(block, [frame]), block) for block in blocks]
            decoded = torch.stack([value[0][0] for value in primary]).to(device)
            phases = torch.stack([value[1][0] for value in primary]).to(device)
            scoring = torch.tensor([blocks.index(block) for block in scored], device=device, dtype=torch.long)
            outputs = {"bilinear": upscale(decoded, phases), "nearest": upscale(decoded, phases, "nearest")}
            if reference:
                keys = [(frame, block) for block in blocks]
                cached = torch.stack([cache.get(key) for key in keys]).to(device)
                if not torch.equal(cached, decoded):
                    raise ValueError("V9 and V11 decoded inputs differ")
                dummy = torch.zeros(len(decoded), 1, 640, dtype=torch.long, device=device)
                outputs["v9-reference"] = reference(dummy, phases[:, None], dummy[..., 0], decoded[:, None])
            outputs[single_name], geometric, active = batch_predict(
                models[single_name], decoded[:, None], phases[:, None], args.batch_size, True
            )
            supports = {single_name: (geometric[scoring], active[scoring])}
            if scored:
                history = torch.stack([primary[blocks.index(block)][0] for block in scored]).to(device)
                history_phases = torch.stack([primary[blocks.index(block)][1] for block in scored]).to(device)
            if include_mosaic:
                mosaic = outputs[single_name].clone()
                if scored:
                    values, measured = models[single_name].exact_map(history, history_phases)
                    if not bool(measured.any(dim=1).all()):
                        raise ValueError("direct tiling requires a complete 16-phase set")
                    mosaic[scoring] = F.pixel_shuffle(values.sum(dim=1), 2)
                outputs[mosaic_name] = mosaic
            for name, model in models.items():
                if name == single_name:
                    continue
                output = outputs[single_name].clone()
                if scored:
                    values, geometric, active = batch_predict(
                        model, history[:, :model.frames], history_phases[:, :model.frames], args.batch_size, True
                    )
                    output[scoring] = values
                    supports[name] = (geometric, active)
                outputs[name] = output
            # Target bytes are consumed only after the checkpoint predictions are ready.
            target_rows = conn.execute(
                "SELECT block_index, raw_bayer_target, source_frame_index FROM samples "
                "WHERE frame_index=? ORDER BY block_index", (frame,)
            ).fetchall()
            target_by_block = {block: (raw, source) for block, raw, source in target_rows}
            target = torch.stack([torch.frombuffer(bytearray(target_by_block[block][0]), dtype=torch.uint8)
                                  for block in blocks]).to(device).float().div_(255).view(-1, 1, HEIGHT, WIDTH)
            rgb = {name: demosaic(output) for name, output in outputs.items()}
            if fsr:
                keys = [(frame, block) for block in blocks]
                rgb["fsr1-rcas0"] = fsr.get(keys, device)
                rgb["fsr1-easu"] = fsr.get(keys, device, variant="easu4")
                for name in ("fsr1-rcas0", "fsr1-easu"):
                    outputs[name] = rgb_to_raw(rgb[name])
            target_rgb = demosaic(target)
            if scored:
                prev = dict(conn.execute("SELECT block_index, raw_bayer_target FROM samples WHERE frame_index=?",
                                         (frame - 1,)).fetchall())
                previous = torch.stack([torch.frombuffer(bytearray(prev[block]), dtype=torch.uint8)
                                        for block in scored]).to(device).float().div_(255).view(-1, 1, HEIGHT, WIDTH)
                earlier_source = conn.execute("SELECT source_frame_index FROM samples WHERE frame_index=? LIMIT 1",
                                              (frame - 1,)).fetchone()[0]
                repeated = torch.tensor([earlier_source == target_by_block[b][1] for b in scored], device=device)
                truth = target[scoring]
                static = ((truth - previous).abs() * 255).round() < 2
                exact, exact_mask = models[single_name].exact_map(decoded[scoring, None], phases[scoring, None])
                exact = F.pixel_shuffle(exact[:, 0], 2)
                measured = F.pixel_shuffle(exact_mask[:, 0].float(), 2).bool()
                repeated_mask = repeated[:, None, None, None].expand_as(truth)
                region_masks = (static, ~static, repeated_mask, ~repeated_mask, measured)
                add_error(known_error, "decoded-to-target", exact - truth, measured)
                errors, rgb_errors = {}, {}
                complete = scored == list(range(BLOCKS))
                for name in names:
                    prediction = outputs[name][scoring]
                    residual = prediction - truth
                    errors[name] = update_stats(raw_stats, name, prediction, truth)
                    rgb_errors[name] = update_stats(rgb_stats, name, rgb[name][scoring], target_rgb[scoring])
                    for region, mask in zip(regions, region_masks):
                        add_error(region_stats[region], name, residual, mask)
                    add_error(known_error, name, prediction - exact, measured)
                    wins[name] += sum(a < b for a, b in zip(errors[name], errors[single_name]))
                    if complete:
                        # Compare equal CFA colors on either side of each strip seam.
                        seams = (residual[1:, :, :2] - residual[:-1, :, -2:]).abs()
                        seam_errors[name][0] += seams.sum().item()
                        seam_errors[name][1] += seams.numel()
                        if previous_frame == frame - 1:
                            temporal = (residual - previous_errors[name]).abs()
                            temporal_errors[name][0] += temporal.sum().item()
                            temporal_errors[name][1] += temporal.numel()
                        previous_errors[name] = residual
                if complete:
                    previous_frame = frame
                for name, (geometry, valid) in supports.items():
                    support_stats[name][0] += int(geometry.sum())
                    support_stats[name][1] += int(valid.sum())
                    support_stats[name][2] += geometry.numel()
                for i, block in enumerate(scored):
                    rows.write(json.dumps({"frame": frame, "block": block, "same_source_as_previous": bool(repeated[i]),
                                           "raw_projection_mae": {n: e[i] * 255 for n, e in errors.items()},
                                           "rgb_reference_mae": {n: e[i] * 255 for n, e in rgb_errors.items()}}) + "\n")
                samples += len(scored)
            for clip in video_clips:
                panel, single = videos[clip["name"]]
                images = {name: dict(enumerate(value.cpu())) for name, value in rgb.items() if name in panel_names}
                images["target"] = dict(enumerate(target_rgb.cpu()))
                panel.write(frame, images)
                single.write(images[video_model])
                clip["fallback_strips"] += BLOCKS - len(scored)
            if full_video:
                full_video.write(dict(enumerate(rgb[video_model].cpu())))
                full_preview["fallback_strips"] += BLOCKS - len(scored)
        for clip in previews:
            clip["frames"] = videos[clip["name"]][0].frames
            clip["generated_method"] = video_model
        if full_video:
            full_preview["frames"] = full_video.frames
    for clip in previews:
        if clip["frames"] != clip["stop"] - clip["start"]:
            raise ValueError("preview frame count does not match its source range")
        folder = args.out/clip["name"]
        middle = clip["frames"] // 2
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
                        "-i", str(folder/"comparison.mp4"), "-vf", f"select=eq(n\\,{middle})",
                        "-frames:v", "1", "-update", "1", str(folder/"preview.png")], check=True)
        if not (folder/"preview.png").is_file():
            raise RuntimeError("could not read back the comparison video")
    if full_preview:
        if full_preview["frames"] != stop - start + 1:
            raise ValueError("full video frame count does not match its source range")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror",
                        "-i", str(args.out/"full-validation"/"generated.mp4"), "-f", "null", "-"], check=True)
    report = {
        "samples": samples, "start_frame": start, "stop_frame_inclusive": stop,
        "first_evaluated_frame": min(keys_by_frame), "last_evaluated_frame": max(keys_by_frame),
        "database_sha256": config["database_sha256"], "training_key_sha256": config["keys_sha256"],
        "checkpoint_epochs": epochs, "training_budget": {field: config[field] for field in fields[3:]},
        "raw_projection_metrics": summarize(raw_stats), "rgb_reference_metrics": summarize(rgb_stats),
        "region_raw_metrics": {r: summarize({n: s for n, s in per_method.items() if s[2]})
                               for r, per_method in region_stats.items()},
        "region_raw_pixels": {r: per_method[single_name][2] for r, per_method in region_stats.items()},
        "measurement_reference_metrics": summarize(known_error),
        "strip_seam_residual_mae": {n: total / count * 255 for n, (total, count) in seam_errors.items()},
        "consecutive_frame_residual_mae": {n: total / count * 255 for n, (total, count) in temporal_errors.items()},
        "coverage": {n: {"geometry": g / count, "valid": v / count} for n, (g, v, count) in support_stats.items()},
        "win_rate_vs_t1": {n: wins[n] / samples for n in names}, "previews": previews,
        "full_video": full_preview,
        "fsr_report": str(args.fsr_report) if fsr else None,
        "v9_reference": reference_info,
        "status": "flow_check" if config["max_steps"] else "same_video_quality_comparison",
    }
    (args.out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("samples", "raw_projection_metrics", "previews")}), flush=True)


if __name__ == "__main__":
    run_logged("evaluation", main)
