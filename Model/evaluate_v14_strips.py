"""Measure whole-strip retrieval with complete 640x16 RAW8 keys and values."""
import argparse
import json
from pathlib import Path

import torch

from evaluate_v13 import ROI, display, scores
from evaluate_v14 import REFERENCE, export
from record_experiment import digest, run_logged, write_json
from v13.data import SPLITS, fingerprints, labels


def manifest(first, last):
    return [{"frame": frame, "block": block}
            for frame in range(first, last + 1) for block in range(30)]


def nearest_strips(raw, keys, query):
    """Find the closest complete RAW strip at each of the 30 vertical positions."""
    distance = (keys - query.short()).abs().sum(dim=(-2, -1), dtype=torch.int32)
    index = distance.argmin(dim=0)
    return raw[index, torch.arange(raw.shape[1])], index


def roundtrip(raw, keys, frame_indices):
    exact = absolute_error = 0
    for index in frame_indices:
        result, _ = nearest_strips(raw, keys, raw[index])
        difference = (result.short() - keys[index]).abs()
        exact += int((difference == 0).flatten(1).all(dim=1).sum())
        absolute_error += int(difference.sum())
    strips = len(frame_indices) * raw.shape[1]
    return {"strips": strips, "exact_strip_count": exact, "bytes_checked": strips * raw.shape[-2] * raw.shape[-1],
            "absolute_error": absolute_error}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exact-only", action="store_true", help="check every reference strip against the codebook")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)
    provenance = fingerprints(args.db, args.scene_plan)
    scene = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    if scene["database_sha256"] != provenance["db"] or [f for f in scene["scene_starts"] if f >= REFERENCE[0]] != [REFERENCE[0]]:
        raise ValueError("expected the BBB scene plan paired with the input database")

    refs = manifest(*REFERENCE)
    raw = labels(args.db, refs).reshape(-1, 30, 16, 640)
    keys = raw.short()
    book = args.out / "codebook.pt"
    torch.save({"frame_range": REFERENCE, "raw": raw}, book)
    if args.exact_only:
        first, last = REFERENCE
    else:
        audit_start, last, _ = SPLITS["S2-audit"]
        first = audit_start + 15
        if first <= REFERENCE[1]:
            raise ValueError("reference and evaluation frames must be disjoint")
    freeze = {"reference_range": REFERENCE, "reference_strips": len(refs),
              "strip_dimensions": [640, 16], "candidates_per_position": len(raw),
              "data_column": "raw_bayer_target", "distance": "full-strip sum of absolute differences",
              "tie_breaker": "earliest reference frame", "codebook_sha256": digest(book),
              "score_range": [first, last], "provenance": provenance, "roi_sha256": digest(ROI)}
    write_json(args.out / "freeze.json", freeze)
    report = {**freeze, "freeze_sha256": digest(args.out / "freeze.json"), "start_frame": first,
              "first_evaluated_frame": first, "last_evaluated_frame": last}

    control_indices = range(len(raw)) if args.exact_only else (0, len(raw) // 2, len(raw) - 1)
    control = roundtrip(raw, keys, control_indices)
    if args.exact_only:
        mae = control["absolute_error"] / control["bytes_checked"]
        write_json(args.out / "report.json", {**report, "samples": len(refs), "scope": "in-codebook roundtrip",
                                             "raw_mae": mae, **control})
        print(f"RAW MAE {mae:.4f}, exact {control['exact_strip_count']}/{len(refs)} strips", flush=True)
    if control["absolute_error"]:
        raise ValueError("a reference strip could not be retrieved byte for byte")
    if args.exact_only:
        return

    expected = SPLITS["S2-audit"][2]
    review = manifest(first, last)
    if len(review) != expected:
        raise ValueError("unexpected evaluation size")
    truth = labels(args.db, review)
    selected = [nearest_strips(raw, keys, frame) for frame in truth.reshape(-1, 30, 16, 640)]
    pred = torch.cat([strips for strips, _ in selected]).unsqueeze(1)
    ref_frames = torch.cat([indices for _, indices in selected]).add(REFERENCE[0]).tolist()
    ref_rgb, rgb = display(truth), display(pred)
    regions = json.loads(ROI.read_text(encoding="utf-8"))["regions"]
    method, per_strip, _ = scores(pred, truth, rgb, ref_rgb, regions)
    exact = int((pred == truth).flatten(1).all(dim=1).sum())
    method.update({"exact_strip_count": exact, "exact_strip_fraction": exact / expected,
                   "strip_mae_min": per_strip.min().item(), "strip_mae_median": per_strip.median().item(),
                   "strip_mae_p90": per_strip.quantile(0.9).item()})
    export(args.out, "target", ref_rgb, first)
    export(args.out, "raw-strip-lookup", rgb, first)
    report.update({"samples": expected,
                   "exact_roundtrip_controls": {"frames": [REFERENCE[0] + i for i in control_indices], **control},
                   "methods": {"raw-strip-lookup": method}})
    write_json(args.out / "report.json", report)
    with (args.out / "blocks.jsonl").open("w", encoding="utf-8") as stream:
        for row, ref, mae in zip(review, ref_frames, per_strip.tolist()):
            stream.write(json.dumps({**row, "reference_frame": ref, "raw_mae": mae}) + "\n")
    print(f"full RAW strip lookup: RAW MAE {method['raw']['mae_8bit']:.4f}, exact {exact}/{expected}", flush=True)


if __name__ == "__main__":
    run_logged("evaluation", main)
