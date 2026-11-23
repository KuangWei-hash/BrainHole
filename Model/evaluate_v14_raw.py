"""Measure exact RAW patch retrieval with a codebook built without JPEG XS."""
import argparse
import json
from pathlib import Path

import torch

from evaluate_v13 import ROI, display, scores
from evaluate_v14 import PATCH_WIDTH, REFERENCE, export, load_split
from record_experiment import digest, run_logged, write_json
from v11.phase import scatter_indices
from v13.data import fingerprints, labels


def sample(raw, phase):
    rows, cols = scatter_indices(phase)
    return raw[..., rows[:, None], cols[None, :]]


def patches(strips, width):
    return strips.unfold(-1, width, width).transpose(-3, -2)


def retrieve(raw, keys, query):
    """Pick the lowest-MAE reference per location, breaking ties by source order."""
    distance = (keys - query).abs().sum(dim=(-2, -1), dtype=torch.int32)
    best = distance.argmin(dim=0, keepdim=True)
    indices = best[..., None, None].expand(1, *raw.shape[1:])
    selected = raw.gather(0, indices).squeeze(0)
    return selected.transpose(-3, -2).reshape(30, 1, 16, 640)


def predict(raw, lookup_keys, phases, queries):
    outputs = []
    for phase, query in zip(phases, queries):
        outputs.append(retrieve(raw, lookup_keys[phase], patches(query, PATCH_WIDTH // 4).short()))
    return torch.cat(outputs)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)
    provenance = fingerprints(args.db, args.scene_plan)
    scene = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    if scene["database_sha256"] != provenance["db"] or [f for f in scene["scene_starts"] if f >= REFERENCE[0]] != [REFERENCE[0]]:
        raise ValueError("expected the BBB Q=5 scene plan")
    references = [{"frame": frame, "block": block}
                  for frame in range(REFERENCE[0], REFERENCE[1] + 1) for block in range(30)]
    reference_raw = labels(args.db, references).reshape(-1, 30, 16, 640)
    clean_keys = torch.stack([sample(reference_raw, phase) for phase in range(16)])
    book = args.out / "codebook.pt"
    torch.save({"range": REFERENCE, "patch_width": PATCH_WIDTH, "raw": reference_raw,
                "clean_keys": clean_keys}, book)
    freeze = {"reference_range": REFERENCE, "reference_strips": len(references), "patch_width": PATCH_WIDTH,
              "candidate_count_per_location": len(reference_raw), "key": "phase-sampled RAW8 without compression",
              "value": "full RAW8 patch", "distance": "sum of absolute differences", "provenance": provenance,
              "codebook_sha256": digest(book), "audit_cache_sha256": digest(args.cache / "S2-audit.pt"),
              "roi_sha256": digest(ROI)}
    write_json(args.out / "freeze.json", freeze)

    phases, decoded, base, manifest = load_split(args.db, args.cache, provenance, "S2-audit")
    raw = patches(reference_raw, PATCH_WIDTH).contiguous()
    keys = [patches(value, PATCH_WIDTH // 4).short() for value in clean_keys]
    decoded_lookup = predict(raw, keys, phases, decoded)

    truth = labels(args.db, manifest)
    truth_frames = truth.reshape(-1, 30, 16, 640)
    clean_lookup = predict(raw, keys, phases, (sample(frame, phase) for frame, phase in zip(truth_frames, phases)))
    oracle_keys = raw.short()
    oracle = torch.cat([retrieve(raw, oracle_keys, patches(frame, PATCH_WIDTH).short()) for frame in truth_frames])
    del oracle_keys
    first, last = manifest[0]["frame"], manifest[-1]["frame"]
    regions = json.loads(ROI.read_text(encoding="utf-8"))["regions"]
    reference_rgb = display(truth)
    export(args.out, "target", reference_rgb, first)
    methods, per_strip = {}, {}
    outputs = {"v12": base, "raw-lookup-decoded-query": decoded_lookup,
               "raw-lookup-clean-query": clean_lookup, "raw-oracle": oracle}
    for name, pred in outputs.items():
        rgb = display(pred)
        result, per_strip[name], _ = scores(pred, truth, rgb, reference_rgb, regions)
        exact = patches(pred == truth, PATCH_WIDTH).all(dim=(-2, -1)).sum().item()
        result["exact_patch_count"] = exact
        result["exact_patch_fraction"] = exact / (len(manifest) * (640 // PATCH_WIDTH))
        methods[name] = result
        export(args.out, name, rgb, first)
        print(f"{name}: RAW MAE {result['raw']['mae_8bit']:.4f}, exact patches {exact}", flush=True)
    if (per_strip["raw-oracle"] > per_strip["raw-lookup-clean-query"] + 1e-4).any() or (
            per_strip["raw-oracle"] > per_strip["raw-lookup-decoded-query"] + 1e-4).any():
        raise ValueError("oracle must have the lowest RAW MAE among patch lookups")
    report = {**freeze, "freeze_sha256": digest(args.out / "freeze.json"), "samples": len(manifest),
              "start_frame": first, "first_evaluated_frame": first, "last_evaluated_frame": last,
              "patches_scored": len(manifest) * (640 // PATCH_WIDTH), "methods": methods}
    write_json(args.out / "report.json", report)
    with (args.out / "blocks.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(manifest):
            stream.write(json.dumps({**row, **{name: value[index].item() for name, value in per_strip.items()}}) + "\n")


if __name__ == "__main__":
    run_logged("evaluation", main)
