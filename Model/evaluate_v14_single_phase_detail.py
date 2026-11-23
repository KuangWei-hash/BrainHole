"""Test spotwise detail transfer from one whole-strip reference, using only the current phase."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from baseline import upscale
from evaluate_v13 import ROI, display, scores
from evaluate_v14 import export, high_pass
from evaluate_v14_current_phase import AUDIT, REFERENCE, evaluate_strips, phase_ids, sample
from evaluate_v14_strips import manifest
from evaluate_v14_v12_index import compare
from record_experiment import digest, run_logged, write_json
from v11.phase import scatter_indices
from v12 import nearest_bayer_bytes
from v13.data import fingerprints, labels


DEV = (750, 779)
WIDTHS = (640, 160, 80, 40)
GRID = [(0, 0.)] + [(limit, strength) for limit in (8, 12, 16, 20, 24) for strength in (0.5, 1.)]


def inputs(db, bounds, references, phase_keys):
    rows = manifest(*bounds)
    truth = labels(db, rows)
    phases = phase_ids(db, *bounds)
    reference, _, _, queries = evaluate_strips(truth.reshape(-1, 30, 16, 640), references, phase_keys, phases)
    differences = torch.stack([(sample(frame, phase).short() - current.short()).abs()
                               for frame, phase, current in zip(reference, phases, queries)])
    phase_vector = torch.tensor(phases).repeat_interleave(30)
    decoded = queries.reshape(-1, 1, 4, 160)
    nearest = nearest_bayer_bytes(decoded, phase_vector)
    bilinear = upscale(decoded.float(), phase_vector).round().clamp(0, 255).byte()
    assert_samples(bilinear, phases, queries)
    return truth, phases, queries, reference.reshape_as(truth), differences, {"nearest": nearest, "bilinear": bilinear}, rows


def assert_samples(pred, phases, queries):
    for phase, frame, current in zip(phases, pred.reshape(-1, 30, 16, 640), queries):
        rows, cols = scatter_indices(phase)
        if not torch.equal(frame[:, rows[:, None], cols[None, :]], current):
            raise ValueError("single-phase output changed a measured sample")


def correction(base, reference, phases):
    high = torch.cat([high_pass(raw) for raw in reference.split(120)])
    high -= torch.cat([high_pass(raw) for raw in base.split(120)])
    delta = high.reshape(-1, 30, 16, 640)
    for phase, frame in zip(phases, delta):
        rows, cols = scatter_indices(phase)
        frame[:, rows[:, None], cols[None, :]] = 0
    return high


def local_distance(differences, width):
    tiles = 640 // width
    return differences.reshape(-1, 4, tiles, width // 4).float().mean(dim=(1, 3))


def transfer(base, delta, distances, limit, strength):
    if not strength:
        return base, 0.
    weight = (1 - distances / limit).clamp_min(0) * strength
    ramps = F.interpolate(weight[:, None], size=640, mode="linear", align_corners=False)
    result = (base + delta * ramps[:, :, None, :]).round().clamp(0, 255).byte()
    return result, (weight > 0).float().mean().item()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)
    provenance = fingerprints(args.db, args.scene_plan)
    original = json.loads(args.checkpoint.with_name("freeze.json").read_text(encoding="utf-8"))
    if original["provenance"] != provenance or original["codebook_sha256"] != digest(args.checkpoint):
        raise ValueError("codebook does not match the selected database and scene")
    book = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if tuple(book["reference_range"]) != REFERENCE or not REFERENCE[1] < DEV[0] <= DEV[1] < AUDIT[0]:
        raise ValueError("expected separate reference, development and evaluation ranges")
    references = book["raw"].transpose(0, 1).contiguous()
    phase_keys = [sample(references, phase).short() for phase in range(16)]
    regions = json.loads(ROI.read_text(encoding="utf-8"))["regions"]

    truth, phases, _, reference, differences, bases, _ = inputs(args.db, DEV, references, phase_keys)
    baseline = {name: compare(pred, truth, regions) for name, pred in bases.items()}
    base_name = min(baseline, key=lambda name: baseline[name]["raw_mae"])
    base = bases[base_name]
    delta = correction(base, reference, phases)
    development, chosen = {}, {}
    for width in WIDTHS:
        distances = local_distance(differences, width)
        candidates = []
        for limit, strength in GRID:
            pred, coverage = transfer(base, delta, distances, limit, strength)
            candidates.append({"limit": limit, "strength": strength, "eligible_tile_fraction": coverage,
                               **compare(pred, truth, regions)})
        development[str(width)] = candidates
        target = baseline[base_name]
        eligible = [row for row in candidates if row["temporal"] <= target["temporal"] + 0.05 and
                    row["title_temporal"] <= target["title_temporal"] + 0.05]
        chosen[str(width)] = min(eligible, key=lambda row: row["raw_mae"])
    selected_width = min(WIDTHS, key=lambda width: chosen[str(width)]["raw_mae"])
    frozen = {"reference_range": REFERENCE, "development_range": DEV, "widths": WIDTHS,
              "codebook_sha256": original["codebook_sha256"], "provenance": provenance, "roi_sha256": digest(ROI),
              "development_baselines": baseline, "development_candidates": development,
              "selected_base": base_name, "selected_width": selected_width, "selected_by_width": chosen}
    write_json(args.out / "freeze.json", frozen)
    print(f"development chose {base_name}, width {selected_width}: {chosen[str(selected_width)]}", flush=True)

    truth, phases, queries, reference, differences, bases, rows = inputs(args.db, AUDIT, references, phase_keys)
    base = bases[base_name]
    delta = correction(base, reference, phases)
    target_rgb = display(truth)
    export(args.out, "target", target_rgb, AUDIT[0])
    predictions = {name: (pred, None) for name, pred in bases.items()}
    for width in WIDTHS:
        options = chosen[str(width)]
        predictions[f"detail-w{width}"] = transfer(base, delta, local_distance(differences, width),
                                                     options["limit"], options["strength"])
    methods, per_strip = {}, {}
    for name, (pred, coverage) in predictions.items():
        assert_samples(pred, phases, queries)
        rgb = display(pred)
        methods[name], per_strip[name], _ = scores(pred, truth, rgb, target_rgb, regions)
        if coverage is not None:
            methods[name]["eligible_tile_fraction"] = coverage
        export(args.out, name, rgb, AUDIT[0])
        print(f"{name}: RAW {methods[name]['raw']['mae_8bit']:.4f}, temporal {methods[name]['temporal']:.4f}", flush=True)
    report = {**frozen, "freeze_sha256": digest(args.out / "freeze.json"), "samples": len(rows),
              "start_frame": AUDIT[0], "first_evaluated_frame": AUDIT[0], "last_evaluated_frame": AUDIT[1],
              "methods": methods}
    write_json(args.out / "report.json", report)
    with (args.out / "blocks.jsonl").open("w", encoding="utf-8") as stream:
        for i, row in enumerate(rows):
            stream.write(json.dumps({**row, **{name: error[i].item() for name, error in per_strip.items()}}) + "\n")


if __name__ == "__main__":
    run_logged("evaluation", main)
