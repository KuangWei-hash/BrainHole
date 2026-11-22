"""Measure direct 16-phase tiling on complete same-strip windows."""
import argparse
from collections import defaultdict
from contextlib import ExitStack, closing
import json
from pathlib import Path

import torch

from baseline import upscale
from evaluate import demosaic
from evaluate_fsr import RgbVideo, summarize, update_stats
from record_experiment import digest, run_logged
from train_v11 import split_windows
from v11.data import HEIGHT, LR_HEIGHT, LR_WIDTH, WIDTH, connect
from v11.phase import PHASES, scatter_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("select a new output directory")
    torch.set_num_threads(3)
    scene_plan = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    if digest(args.db) != scene_plan["database_sha256"]:
        raise ValueError("scene plan belongs to a different database")
    _, validation = split_windows(args.db, PHASES, (0, 719), (721, 899), scene_plan["scene_starts"])
    groups = defaultdict(dict)
    for block, frames in validation:
        groups[frames[-1]][block] = frames
    args.out.mkdir(parents=True)
    methods = ("bilinear", "phase-mosaic-t16")
    stats = {"all": defaultdict(lambda: [0., 0., 0]),
             "unchanged_since_sample": defaultdict(lambda: [0., 0., 0]),
             "changed_since_sample": defaultdict(lambda: [0., 0., 0])}
    source_noise = defaultdict(lambda: [0., 0., 0])
    phase_indices = [scatter_indices(phase) for phase in range(PHASES)]
    phases, lows, targets = {}, {}, {}
    with closing(connect(args.db)) as conn:
        for frame, block, phase, low, raw in conn.execute(
            "SELECT frame_index, block_index, phase_id, decoded, raw_bayer_target FROM samples "
            "WHERE frame_index BETWEEN 721 AND 899"
        ):
            key = (frame, block)
            phases[key] = phase - 1
            lows[key] = torch.frombuffer(bytearray(low), dtype=torch.uint8).view(LR_HEIGHT, LR_WIDTH)
            targets[key] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(HEIGHT, WIDTH)
    previews = [clip.copy() for clip in scene_plan["previews"] if args.video]
    with ExitStack() as cleanup, torch.inference_mode():
        rows = cleanup.enter_context((args.out/"blocks.jsonl").open("w", encoding="utf-8"))
        videos = {}
        for clip in previews:
            folder = args.out/clip["name"]
            folder.mkdir()
            videos[clip["name"]] = RgbVideo(folder, ["target", *methods])
            cleanup.callback(videos[clip["name"]].close)
        for frame, blocks in groups.items():
            mosaics, sampled_targets = [], []
            for block, window in blocks.items():
                keys = [(source, block) for source in window]
                if {phases[key] for key in keys} != set(range(PHASES)):
                    raise ValueError("diagnostic requires all sixteen distinct phases per strip")
                mosaic = torch.empty((HEIGHT, WIDTH), dtype=torch.uint8)
                for key in keys:
                    y, x = phase_indices[phases[key]]
                    mosaic[y[:, None], x[None, :]] = lows[key]
                mosaics.append(mosaic)
                sampled_target = torch.empty_like(mosaic)
                for key in keys:
                    y, x = phase_indices[phases[key]]
                    sampled_target[y[:, None], x[None, :]] = targets[key][y[:, None], x[None, :]]
                sampled_targets.append(sampled_target)
            keys = [(frame, block) for block in blocks]
            truth = torch.stack([targets[key] for key in keys])[:, None].float()
            sampled_truth = torch.stack(sampled_targets)[:, None].float()
            low = torch.stack([lows[key] for key in keys])[:, None].float() / 255
            outputs = {
                "bilinear": upscale(low, torch.tensor([phases[key] for key in keys])),
                "phase-mosaic-t16": torch.stack(mosaics)[:, None].float() / 255,
            }
            stable = (truth - sampled_truth).abs() < 2
            masks = {"unchanged_since_sample": stable, "changed_since_sample": ~stable}
            errors = {}
            for name, output in outputs.items():
                errors[name] = update_stats(stats["all"], name, output, truth / 255)
                difference = output - truth / 255
                for region, mask in masks.items():
                    values = difference[mask]
                    counts = stats[region][name]
                    counts[0] += values.abs().sum().item()
                    counts[1] += values.square().sum().item()
                    counts[2] += values.numel()
            update_stats(source_noise, "decoded_vs_own_source", outputs["phase-mosaic-t16"], sampled_truth / 255)
            stable_per_strip = stable.float().mean((1, 2, 3)).tolist()
            for i, block in enumerate(blocks):
                rows.write(json.dumps({"frame": frame, "block": block, "geometry_coverage": 1.0,
                                       "unchanged_fraction": stable_per_strip[i],
                                       "raw_projection_mae": {n: e[i] * 255 for n, e in errors.items()}}) + "\n")
            for clip in previews:
                if clip["start"] <= frame < clip["stop"]:
                    images = {name: dict(zip(blocks, demosaic(raw))) for name, raw in
                              {"target": truth / 255, **outputs}.items()}
                    videos[clip["name"]].write(frame, images)
        for clip in previews:
            clip["frames"] = videos[clip["name"]].frames
    report = {
        "samples": len(validation), "start_frame": 721, "first_evaluated_frame": min(groups),
        "last_evaluated_frame": max(groups), "geometric_coverage": 1.0,
        "training_steps": 0, "raw_projection_metrics": summarize(stats["all"]),
        "region_raw_metrics": {name: summarize({n: s for n, s in values.items() if s[2]})
                               for name, values in stats.items() if name != "all"},
        "region_raw_pixels": {name: values["bilinear"][2] for name, values in stats.items()},
        "codec_measurement_error": summarize(source_noise), "previews": previews,
    }
    (args.out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    run_logged("evaluation", main)
