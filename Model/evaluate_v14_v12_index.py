"""Evaluate whole-strip lookup keyed by causal v12 reconstructions."""
import argparse
import json
from pathlib import Path

import torch

from evaluate_v13 import ROI, display, scores
from evaluate_v14 import dev_scores, export, load_split, read_window
from record_experiment import digest, run_logged, write_json
from v11.phase import scatter_indices
from v12 import HISTORY, nearest_bayer_bytes, phase_protected_blend
from v13.data import atomic_save, fingerprints, labels


SCENE_FIRST = 691
REFERENCE = (SCENE_FIRST + HISTORY - 1, 749)
GRID = [(0, 0.)] + [(limit, strength) for limit in (8, 12, 16, 20, 24) for strength in (0.5, 1.)]


def build_keys(db):
    phases, decoded, manifest = read_window(db, SCENE_FIRST, REFERENCE[1])
    decoded = decoded.unsqueeze(2)
    nearest = nearest_bayer_bytes(decoded.flatten(0, 1), torch.tensor(phases).repeat_interleave(30))
    nearest = nearest.reshape(len(phases), 30, 1, 16, 640)
    keys = []
    for frame in range(HISTORY - 1, len(phases)):
        history = list(range(frame, frame - HISTORY, -1))
        window_phases = [phases[i] for i in history]
        if len(set(window_phases)) != HISTORY:
            raise ValueError("expected all 16 phases in the reference history")
        for block in range(30):
            keys.append(phase_protected_blend(nearest[history, block], decoded[history, block], window_phases))
    return torch.stack(keys), manifest[(HISTORY - 1) * 30:]


def as_positions(raw):
    return raw.reshape(-1, 30, 16 * 640).transpose(0, 1).contiguous().float()


def match(query, keys, raw_refs):
    distance, indices = torch.cdist(as_positions(query), keys, p=1).min(dim=-1)
    selected = raw_refs[indices.T, torch.arange(30)]
    return selected.reshape_as(query), distance.T.contiguous().reshape(-1) / (16 * 640), indices.T.reshape(-1)


def gated_blend(base, selected, distance, phases, limit, strength):
    if strength == 0:
        return base, 0.
    weight = (1 - distance / limit).clamp_min(0) * strength
    delta = ((selected.short() - base.short()) * weight[:, None, None, None]).reshape(-1, 30, 16, 640)
    for frame, phase in enumerate(phases):
        rows, cols = scatter_indices(phase)
        delta[frame, :, rows[:, None], cols[None, :]] = 0
    result = (base + delta.reshape_as(base)).round().byte()
    return result, (weight > 0).float().mean().item()


def compare(pred, truth, regions):
    residual = (pred.short() - truth.short()).reshape(-1, 480, 640)
    return {**dev_scores(pred, truth, regions), "temporal": residual.diff(dim=0).abs().float().mean().item()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--audit-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)
    provenance = fingerprints(args.db, args.scene_plan)
    scene = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    if scene["database_sha256"] != provenance["db"] or [f for f in scene["scene_starts"] if f >= SCENE_FIRST] != [SCENE_FIRST]:
        raise ValueError("expected the BBB scene plan paired with the input database")
    regions = json.loads(ROI.read_text(encoding="utf-8"))["regions"]

    keys, ref_manifest = build_keys(args.db)
    refs = labels(args.db, ref_manifest).reshape(-1, 30, 16, 640)
    book = args.out / "codebook.pt"
    atomic_save({"v12_keys": keys.reshape_as(refs), "raw": refs, "reference_range": REFERENCE}, book)
    index = as_positions(keys)
    phases, _, base, dev = load_split(args.db, args.cache, provenance, "S2-dev")
    offset = (dev[0]["frame"] - REFERENCE[0]) * 30
    overlap = len(keys) - offset
    if not torch.equal(keys[offset:], base[:overlap]):
        raise ValueError("generated codebook keys disagree with the existing v12 cache")
    dev, phases, base = dev[overlap:], phases[overlap // 30:], base[overlap:]
    truth = labels(args.db, dev)
    direct, distance, _ = match(base, index, refs)
    candidates = []
    for limit, strength in GRID:
        pred, coverage = gated_blend(base, direct, distance, phases, limit, strength)
        candidates.append({"limit": limit, "strength": strength, "accepted_strip_fraction": coverage,
                           **compare(pred, truth, regions)})
    baseline = candidates[0]
    chosen = min((row for row in candidates if row["temporal"] <= baseline["temporal"] + 0.05
                  and row["title_temporal"] <= baseline["title_temporal"] + 0.05), key=lambda row: row["raw_mae"])
    frozen = {"reference_range": REFERENCE, "reference_strips": len(keys), "key_cache_overlap_strips": overlap,
              "development_range": [dev[0]["frame"], dev[-1]["frame"]], "development_candidates": candidates,
              "development_direct": compare(direct, truth, regions), "chosen": chosen,
              "provenance": provenance, "codebook_sha256": digest(book), "roi_sha256": digest(ROI),
              "dev_cache_sha256": digest(args.cache / "S2-dev.pt")}
    write_json(args.out / "freeze.json", frozen)
    print(f"selected on development: {chosen}", flush=True)

    phases, decoded, base, audit = load_split(args.db, args.audit_cache, provenance, "S2-audit")
    direct, distance, selected_frames = match(base, index, refs)
    gated, coverage = gated_blend(base, direct, distance, phases, chosen["limit"], chosen["strength"])
    for phase, frame, measured in zip(phases, gated.reshape(-1, 30, 16, 640), decoded):
        rows, cols = scatter_indices(phase)
        if not torch.equal(frame[:, rows[:, None], cols[None, :]], measured):
            raise ValueError("gated lookup changed current decoded measurements")
    truth = labels(args.db, audit)
    oracle, oracle_mae, _ = match(truth, as_positions(refs), refs)
    if ((oracle.short() - truth.short()).abs().sum(dim=(-2, -1)) >
            (direct.short() - truth.short()).abs().sum(dim=(-2, -1))).any():
        raise ValueError("lookup exceeded the paired bank's nearest-RAW bound")
    first, last = audit[0]["frame"], audit[-1]["frame"]
    reference = display(truth)
    export(args.out, "target", reference, first)
    methods, by_strip = {}, {}
    for name, pred in (("v12", base), ("v14-direct", direct), ("v14-gated", gated)):
        rgb = display(pred)
        methods[name], by_strip[name], _ = scores(pred, truth, rgb, reference, regions)
        export(args.out, name, rgb, first)
        print(f"{name}: RAW {methods[name]['raw']['mae_8bit']:.4f}, temporal {methods[name]['temporal']:.4f}", flush=True)
    report = {**frozen, "freeze_sha256": digest(args.out / "freeze.json"), "samples": len(audit),
              "start_frame": first, "first_evaluated_frame": first, "last_evaluated_frame": last,
              "oracle_raw_mae": oracle_mae.mean().item(), "accepted_strip_fraction": coverage, "methods": methods,
              "audit_cache_sha256": digest(args.audit_cache / "S2-audit.pt")}
    write_json(args.out / "report.json", report)
    with (args.out / "blocks.jsonl").open("w", encoding="utf-8") as stream:
        for i, row in enumerate(audit):
            stream.write(json.dumps({**row, "reference_frame": int(selected_frames[i]) + REFERENCE[0],
                                     "v12_key_mae": distance[i].item(),
                                     **{name: values[i].item() for name, values in by_strip.items()}}) + "\n")


if __name__ == "__main__":
    run_logged("evaluation", main)
