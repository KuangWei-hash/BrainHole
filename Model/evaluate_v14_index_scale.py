"""Vary horizontal lookup width while holding the stored codebook and blend fixed."""
import argparse
import json
from pathlib import Path

import torch

from evaluate_v13 import ROI, display, scores
from evaluate_v14 import export, load_split
from evaluate_v14_v12_index import compare
from record_experiment import digest, run_logged, write_json
from v11.phase import scatter_indices
from v13.data import fingerprints, labels


WIDTHS = (640, 160, 80, 40)


def tiles(raw, width):
    return raw.reshape(-1, 30, 16, 640 // width, width).permute(1, 3, 0, 2, 4).reshape(30 * (640 // width), -1, 16 * width)


def match_and_mix(base, phases, keys, values, width, limit, strength):
    query = tiles(base, width).float()
    distance, indices = torch.cdist(query, keys.float(), p=1).min(dim=-1)
    found = values[torch.arange(len(indices))[:, None], indices]
    selected = found.reshape(30, 640 // width, -1, 16, width).permute(2, 0, 3, 1, 4).reshape_as(base)
    weight = (1 - distance / (16 * width * limit)).clamp_min(0) * strength
    coverage = (weight > 0).float().mean().item()
    weight = weight.T.reshape(-1, 30, 1, 640 // width).repeat_interleave(width, dim=-1)
    delta = (selected.short() - base.short()).reshape(-1, 30, 16, 640) * weight
    for frame, phase in enumerate(phases):
        rows, cols = scatter_indices(phase)
        delta[frame, :, rows[:, None], cols[None, :]] = 0
    return (base + delta.reshape_as(base)).round().byte(), coverage


def nearest_possible(truth, values, width):
    return (torch.cdist(tiles(truth, width).float(), values.float(), p=1).amin(dim=-1) / (16 * width)).mean().item()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--audit-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)
    provenance = fingerprints(args.db, args.scene_plan)
    previous = json.loads(args.checkpoint.with_name("freeze.json").read_text(encoding="utf-8"))
    if previous["provenance"] != provenance or previous["codebook_sha256"] != digest(args.checkpoint):
        raise ValueError("existing dictionary does not match this database and scene plan")
    book = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    limit, strength = (previous["chosen"][name] for name in ("limit", "strength"))
    if limit <= 0 or strength <= 0:
        raise ValueError("expected an active blend in the reference run")
    regions = json.loads(ROI.read_text(encoding="utf-8"))["regions"]

    phases, _, base, dev = load_split(args.db, args.cache, provenance, "S2-dev")
    trim = (previous["development_range"][0] - dev[0]["frame"]) * 30
    phases, base, dev = phases[trim // 30:], base[trim:], dev[trim:]
    truth = labels(args.db, dev)
    baseline = compare(base, truth, regions)
    development = []
    for width in WIDTHS:
        keys, values = tiles(book["v12_keys"], width), tiles(book["raw"], width)
        pred, coverage = match_and_mix(base, phases, keys, values, width, limit, strength)
        development.append({"width": width, "eligible_tile_fraction": coverage, **compare(pred, truth, regions)})
    eligible = [row for row in development if row["temporal"] <= baseline["temporal"] + 0.05
                and row["title_temporal"] <= baseline["title_temporal"] + 0.05]
    chosen = min(eligible, key=lambda row: row["raw_mae"])["width"] if eligible else None
    frozen = {"widths": WIDTHS, "patch_height": 16, "reference_range": book["reference_range"],
              "development_range": [dev[0]["frame"], dev[-1]["frame"]], "blend_limit": limit, "blend_strength": strength,
              "development_baseline": baseline, "development": development, "selected_on_development": chosen,
              "codebook_sha256": previous["codebook_sha256"], "provenance": provenance,
              "dev_cache_sha256": digest(args.cache / "S2-dev.pt"), "roi_sha256": digest(ROI)}
    write_json(args.out / "freeze.json", frozen)
    print(f"development chose width {chosen}", flush=True)

    phases, decoded, base, audit = load_split(args.db, args.audit_cache, provenance, "S2-audit")
    truth = labels(args.db, audit)
    first, last = audit[0]["frame"], audit[-1]["frame"]
    reference = display(truth)
    export(args.out, "target", reference, first)
    base_rgb = display(base)
    numbers, errors = {}, {}
    numbers["v12"], errors["v12"], _ = scores(base, truth, base_rgb, reference, regions)
    export(args.out, "v12", base_rgb, first)
    for width in WIDTHS:
        keys, values = tiles(book["v12_keys"], width), tiles(book["raw"], width)
        pred, coverage = match_and_mix(base, phases, keys, values, width, limit, strength)
        for phase, frame, measured in zip(phases, pred.reshape(-1, 30, 16, 640), decoded):
            rows, cols = scatter_indices(phase)
            if not torch.equal(frame[:, rows[:, None], cols[None, :]], measured):
                raise ValueError("lookup modified the current phase samples")
        name = f"w{width}"
        rgb = display(pred)
        numbers[name], errors[name], _ = scores(pred, truth, rgb, reference, regions)
        numbers[name].update({"eligible_tile_fraction": coverage, "oracle_raw_mae": nearest_possible(truth, values, width)})
        export(args.out, name, rgb, first)
        print(f"{width}x16: RAW {numbers[name]['raw']['mae_8bit']:.4f}, "
              f"high frequency {numbers[name]['high_frequency_mae']:.4f}", flush=True)
    report = {**frozen, "freeze_sha256": digest(args.out / "freeze.json"), "samples": len(audit), "start_frame": first,
              "first_evaluated_frame": first, "last_evaluated_frame": last, "methods": numbers,
              "audit_cache_sha256": digest(args.audit_cache / "S2-audit.pt")}
    write_json(args.out / "report.json", report)
    with (args.out / "blocks.jsonl").open("w", encoding="utf-8") as stream:
        for i, row in enumerate(audit):
            stream.write(json.dumps({**row, **{name: value[i].item() for name, value in errors.items()}}) + "\n")


if __name__ == "__main__":
    run_logged("evaluation", main)
