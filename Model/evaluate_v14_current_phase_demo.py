"""demo.mp4 version of evaluate_v14_current_phase.py: codebook keyed by the exact
causal signal (phase id + its 160x4 RAW8 sample), values are the untouched
original full RAW strip. Proves zero-error retrieval for in-table queries, then
measures generalization to a genuinely later, unseen frame range.
"""
import argparse
from contextlib import closing
import json
from pathlib import Path

import torch

from evaluate_v13 import ROI, display, scores
from evaluate_v14 import export
from evaluate_v14_raw import sample
from evaluate_v14_strips import manifest
from record_experiment import digest, run_logged, write_json
from v11.data import connect
from v12 import nearest_bayer_bytes
from v13.data import atomic_save, fingerprints, labels


def phase_ids(db, first, last):
    with closing(connect(db)) as conn:
        rows = conn.execute("SELECT frame_index, min(phase_id), max(phase_id), count(*) FROM samples "
                            "WHERE frame_index BETWEEN ? AND ? GROUP BY frame_index ORDER BY frame_index",
                            (first, last)).fetchall()
    if [row[0] for row in rows] != list(range(first, last + 1)) or any(
            low != high or not 1 <= low <= 16 or count != 30 for _, low, high, count in rows):
        raise ValueError("expected one valid phase and 30 strips per frame")
    return [row[1] - 1 for row in rows]


def lookup_strip(references, keys, current):
    distances = (keys - current.short().unsqueeze(-3)).abs().sum(dim=(-2, -1), dtype=torch.int32)
    minimum, indices = distances.min(dim=-1)
    gather = indices[..., None, None, None].expand(*indices.shape, 1, 16, 640)
    output = references.gather(-3, gather).squeeze(-3)
    return output, minimum, indices


def evaluate_strips(raw, references, phase_keys, phases):
    samples = [sample(frame, phase) for frame, phase in zip(raw, phases)]
    results = [lookup_strip(references, phase_keys[phase], current) for phase, current in zip(phases, samples)]
    return (torch.stack([row[0] for row in results]), torch.stack([row[1] for row in results]),
            torch.stack([row[2] for row in results]), torch.stack(samples))


def counts(pred, raw, distance):
    exact = (pred == raw).flatten(-2).all(dim=-1)
    return {"complete_exact_strips": int(exact.sum()), "exact_phase_keys": int((distance == 0).sum()),
            "exact_key_but_different_raw_strips": int(((distance == 0) & ~exact).sum()),
            "strips": exact.numel()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-range", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--audit-range", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    reference, audit = tuple(args.reference_range), tuple(args.audit_range)
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)
    provenance = fingerprints(args.db, args.scene_plan)
    scene = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    if scene["database_sha256"] != provenance["db"] or any(reference[0] < f <= audit[1] for f in scene["scene_starts"]):
        raise ValueError("reference and audit must be within the same recorded scene")

    raw = labels(args.db, manifest(*reference)).reshape(-1, 30, 16, 640)
    references = raw.transpose(0, 1).contiguous()
    phase_keys = [sample(references, phase).short() for phase in range(16)]
    book = args.out / "codebook.pt"
    atomic_save({"raw": raw, "reference_range": reference}, book)
    freeze = {"reference_range": reference, "reference_strips": raw.shape[0] * raw.shape[1],
              "query": "one current uncompressed 160x4 RAW8 phase sample and phase id",
              "returned_strip": "one complete 640x16 reference RAW8 strip at the same position",
              "tie_breaker": "earliest reference frame", "distance": "L1 on the 160x4 sample",
              "codebook_sha256": digest(book), "provenance": provenance, "roi_sha256": digest(ROI)}
    write_json(args.out / "freeze.json", freeze)

    reference_phases = phase_ids(args.db, *reference)
    in_book, in_distance, _, _ = evaluate_strips(raw, references, phase_keys, reference_phases)
    control = counts(in_book, raw, in_distance)
    if control["exact_phase_keys"] != control["strips"]:
        raise ValueError("an existing reference did not have an exact phase key")
    if control["complete_exact_strips"] == control["strips"]:
        print(f"ZERO-ERROR CHECK PASSED: {control['complete_exact_strips']}/{control['strips']} "
              f"in-table strips retrieved byte-for-byte", flush=True)
    else:
        print(f"ZERO-ERROR CHECK FAILED: only {control['complete_exact_strips']}/{control['strips']} "
              f"in-table strips retrieved byte-for-byte ({control['exact_key_but_different_raw_strips']} "
              f"strips had an exact phase-key collision with a different reference's true RAW)", flush=True)

    review = manifest(*audit)
    truth = labels(args.db, review)
    truth_frames = truth.reshape(-1, 30, 16, 640)
    phases = phase_ids(args.db, *audit)
    output, distance, indices, queries = evaluate_strips(truth_frames, references, phase_keys, phases)
    positions = truth_frames.transpose(0, 1).flatten(-2).float()
    bounds = torch.cdist(positions, references.flatten(-2).float(), p=1).amin(dim=-1)
    oracle = bounds.double().mean().item() / (16 * 640)
    simple = nearest_bayer_bytes(queries.reshape(-1, 1, 4, 160), torch.tensor(phases).repeat_interleave(30))
    ref_rgb = display(truth)
    regions = json.loads(ROI.read_text(encoding="utf-8"))["regions"]
    if args.video:
        export(args.out, "target", ref_rgb, audit[0])

    methods, errors = {}, {}
    for name, pred in (("single-phase-nearest", simple), ("single-phase-lookup", output.reshape_as(truth))):
        rgb = display(pred)
        methods[name], errors[name], _ = scores(pred, truth, rgb, ref_rgb, regions)
        if args.video:
            export(args.out, name, rgb, audit[0])
        print(f"{name}: RAW MAE {methods[name]['raw']['mae_8bit']:.4f}", flush=True)
    report = {**freeze, "freeze_sha256": digest(args.out / "freeze.json"), "start_frame": audit[0],
              "first_evaluated_frame": audit[0], "last_evaluated_frame": audit[1], "samples": len(review),
              "in_book_control": control, "audit_exact_counts": counts(output, truth_frames, distance),
              "oracle_raw_mae": oracle, "methods": methods}
    write_json(args.out / "report.json", report)
    print(f"oracle (best phase-matched in-table strip) RAW MAE: {oracle:.4f}", flush=True)
    print(f"audit exact byte-for-byte strips: {report['audit_exact_counts']['complete_exact_strips']}"
          f"/{report['audit_exact_counts']['strips']}", flush=True)


if __name__ == "__main__":
    run_logged("evaluation", main)
