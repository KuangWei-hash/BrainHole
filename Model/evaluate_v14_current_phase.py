"""Retrieve a complete RAW8 strip using only one current 160x4 phase sample."""
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


REFERENCE = (706, 749)
AUDIT = (795, 899)


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
    """Compare one 160x4 sample to phase-matched keys and return one complete RAW strip.

    Leading dimensions can be used to batch independent positions. The caller
    provides keys sampled from references at the single current phase.
    """
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
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)
    provenance = fingerprints(args.db, args.scene_plan)
    scene = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    if scene["database_sha256"] != provenance["db"] or any(REFERENCE[0] < f <= AUDIT[1] for f in scene["scene_starts"]):
        raise ValueError("reference and audit must be within the same recorded scene")

    raw = labels(args.db, manifest(*REFERENCE)).reshape(-1, 30, 16, 640)
    references = raw.transpose(0, 1).contiguous()
    phase_keys = [sample(references, phase).short() for phase in range(16)]
    book = args.out / "codebook.pt"
    atomic_save({"raw": raw, "reference_range": REFERENCE}, book)
    freeze = {"reference_range": REFERENCE, "reference_strips": raw.shape[0] * raw.shape[1],
              "query": "one current uncompressed 160x4 RAW8 phase sample and phase id",
              "returned_strip": "one complete 640x16 reference RAW8 strip at the same position",
              "tie_breaker": "earliest reference frame", "distance": "L1 on the 160x4 sample",
              "codebook_sha256": digest(book), "provenance": provenance, "roi_sha256": digest(ROI)}
    write_json(args.out / "freeze.json", freeze)

    reference_phases = phase_ids(args.db, *REFERENCE)
    in_book, in_distance, _, _ = evaluate_strips(raw, references, phase_keys, reference_phases)
    control = counts(in_book, raw, in_distance)
    if control["exact_phase_keys"] != control["strips"]:
        raise ValueError("an existing reference did not have an exact phase key")

    review = manifest(*AUDIT)
    truth = labels(args.db, review)
    truth_frames = truth.reshape(-1, 30, 16, 640)
    phases = phase_ids(args.db, *AUDIT)
    output, distance, indices, queries = evaluate_strips(truth_frames, references, phase_keys, phases)
    positions = truth_frames.transpose(0, 1).flatten(-2).float()
    bounds = torch.cdist(positions, references.flatten(-2).float(), p=1).amin(dim=-1)
    oracle = bounds.double().mean().item() / (16 * 640)
    simple = nearest_bayer_bytes(queries.reshape(-1, 1, 4, 160), torch.tensor(phases).repeat_interleave(30))
    ref_rgb = display(truth)
    regions = json.loads(ROI.read_text(encoding="utf-8"))["regions"]
    export(args.out, "target", ref_rgb, AUDIT[0])

    methods, errors = {}, {}
    for name, pred in (("single-phase-nearest", simple), ("single-phase-lookup", output.reshape_as(truth))):
        rgb = display(pred)
        methods[name], errors[name], _ = scores(pred, truth, rgb, ref_rgb, regions)
        export(args.out, name, rgb, AUDIT[0])
        print(f"{name}: RAW MAE {methods[name]['raw']['mae_8bit']:.4f}", flush=True)
    report = {**freeze, "freeze_sha256": digest(args.out / "freeze.json"), "start_frame": AUDIT[0],
              "first_evaluated_frame": AUDIT[0], "last_evaluated_frame": AUDIT[1], "samples": len(review),
              "in_book_control": control, "audit_exact_counts": counts(output, truth_frames, distance),
              "oracle_raw_mae": oracle, "methods": methods}
    write_json(args.out / "report.json", report)
    matched_indices = indices.flatten().tolist()
    low_mae = distance.flatten().float().div(640).tolist()
    with (args.out / "blocks.jsonl").open("w", encoding="utf-8") as stream:
        for i, row in enumerate(review):
            stream.write(json.dumps({**row, "phase_id": phases[i // 30] + 1, "low_raw_mae": low_mae[i],
                                     "reference_frame": matched_indices[i] + REFERENCE[0],
                                     **{name: values[i].item() for name, values in errors.items()}}) + "\n")


if __name__ == "__main__":
    run_logged("evaluation", main)
