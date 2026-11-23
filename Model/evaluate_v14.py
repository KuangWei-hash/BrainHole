"""Test a same-position, same-phase RAW lookup table on the BBB Q=5 scene."""
import argparse
from contextlib import closing
import json
from pathlib import Path

from PIL import Image
import torch
from torch.nn import functional as F

from evaluate_v10 import SingleVideo
from evaluate_v13 import FIXED, ROI, display, scores, verify_video
from record_experiment import digest, run_logged, write_json
from v11.data import connect
from v11.phase import scatter_indices
from v13.data import SPLITS, fingerprints, labels, load_inputs


REFERENCE = (691, 720)
PATCH_WIDTH = 32
STRIDE = PATCH_WIDTH // 2
PATCHES = (640 - PATCH_WIDTH) // STRIDE + 1
GRID = [(0, 0.)] + [(limit, strength) for limit in (8, 16, 24) for strength in (0.5, 1.)]


def high_pass(raw):
    packed = F.pixel_unshuffle(raw.float(), 2)
    smooth = F.avg_pool2d(F.pad(packed, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
    return F.pixel_shuffle(packed - smooth, 2)


def read_window(db, first, last):
    manifest = [{"frame": frame, "block": block}
                for frame in range(first, last + 1) for block in range(30)]
    with closing(connect(db)) as conn:
        inputs = load_inputs(conn, first, last)
    headers, phases, decoded = zip(*(inputs[row["frame"], row["block"]] for row in manifest))
    if len(set(headers)) != 1 or any(phase != phases[i // 30 * 30] for i, phase in enumerate(phases)):
        raise ValueError("expected a single header and one phase per frame")
    return list(phases[::30]), torch.stack(decoded).reshape(-1, 30, 4, 160), manifest


def load_split(db, cache, provenance, split):
    first, last, expected = SPLITS[split]
    saved = torch.load(cache / f"{split}.pt", weights_only=True)
    phases, decoded, manifest = read_window(db, first + 15, last)
    keys = [(row["frame"], row["block"]) for row in saved["manifest"]]
    if (saved["provenance"] != provenance or saved["split"] != split or
            keys != [(row["frame"], row["block"]) for row in manifest] or len(keys) != expected):
        raise ValueError(f"invalid v12 cache for {split}")
    return phases, decoded, saved["features"][:, :1].contiguous(), manifest


def match_references(ref_phases, keys, values, phases, decoded):
    keys = keys.float().unfold(-1, PATCH_WIDTH // 4, STRIDE // 4)
    values = values.unfold(-1, PATCH_WIDTH, STRIDE).permute(0, 1, 3, 2, 4)
    offsets = torch.arange(STRIDE, dtype=torch.float32) + 0.5
    ramp = torch.cat((offsets, STRIDE - offsets)) / STRIDE
    ramps = ramp.repeat(PATCHES, 1)
    ramps[0, :STRIDE] = ramps[-1, STRIDE:] = 1
    block_ids = torch.arange(30)[:, None]
    patch_ids = torch.arange(PATCHES)[None, :]
    for phase, strip in zip(phases, decoded):
        candidates = [i for i, key_phase in enumerate(ref_phases) if key_phase == phase]
        current = strip.float().unfold(-1, PATCH_WIDTH // 4, STRIDE // 4)
        error = (keys[candidates] - current).abs()
        distance, selected = (0.75 * error.mean(dim=(2, 4)) + 0.25 * error.amax(dim=(2, 4))).min(dim=0)
        detail = values[torch.tensor(candidates)[selected], block_ids, patch_ids]
        yield phase, distance, detail, ramps


def reconstruct(base, matches, limit, strength):
    if not strength:
        return base, 0.
    current_detail = high_pass(base).reshape(-1, 30, 16, 640)
    correction = torch.zeros_like(current_detail)
    accepted = 0
    for index, (phase, distance, detail, ramps) in enumerate(matches):
        weight = (1 - distance / limit).clamp_min(0) * strength
        accepted += int((weight > 0).sum())
        for patch in range(PATCHES):
            start = patch * STRIDE
            span = slice(start, start + PATCH_WIDTH)
            delta = detail[:, patch] - current_detail[index, :, :, span]
            correction[index, :, :, span] += weight[:, patch, None, None] * ramps[patch] * delta
        rows, cols = scatter_indices(phase)
        correction[index, :, rows[:, None], cols[None, :]] = 0
    result = (base + correction.reshape_as(base)).round().clamp(0, 255).byte()
    return result, accepted / (len(matches) * 30 * PATCHES)


def dev_scores(pred, truth, regions):
    error = (pred.float() - truth).reshape(-1, 480, 640)
    x0, y0, x1, y1 = regions["title-when-visible"]
    title = error[:, y0:y1, x0:x1]
    return {"raw_mae": error.abs().mean().item(), "title_raw_mae": title.abs().mean().item(),
            "title_temporal": title.diff(dim=0).abs().mean().item()}


def export(out, name, rgb, first):
    video = SingleVideo(out, name)
    try:
        for frame, strips in enumerate(rgb, first):
            video.write(strips)
            if frame in FIXED["S2-audit"]:
                image = torch.cat(list(strips), dim=1).permute(1, 2, 0)
                Image.fromarray((image.clamp(0, 1) * 255).round().byte().numpy()).save(out / f"{name}-{frame}.png")
    finally:
        video.close()
    verify_video(out / f"{name}.mp4", len(rgb))


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
    if scene["database_sha256"] != provenance["db"] or [f for f in scene["scene_starts"] if f >= REFERENCE[0]] != [REFERENCE[0]]:
        raise ValueError("expected the BBB Q=5 scene plan")
    regions = json.loads(ROI.read_text(encoding="utf-8"))["regions"]
    ref_phases, keys, ref_manifest = read_window(args.db, *REFERENCE)
    if set(ref_phases) != set(range(16)):
        raise ValueError("the lookup table must cover all 16 phases")
    values = high_pass(labels(args.db, ref_manifest)).reshape(-1, 30, 16, 640)
    torch.save({"reference_range": REFERENCE, "phases": ref_phases, "keys": keys, "detail": values},
               args.out / "codebook.pt")

    phases, decoded, base, manifest = load_split(args.db, args.cache, provenance, "S2-dev")
    truth = labels(args.db, manifest)
    matches = list(match_references(ref_phases, keys, values, phases, decoded))
    development = []
    for limit, strength in GRID:
        pred, coverage = reconstruct(base, matches, limit, strength)
        development.append({"limit": limit, "strength": strength, "accepted_patch_fraction": coverage,
                            **dev_scores(pred, truth, regions)})
    baseline = development[0]
    tested = [row for row in development[1:] if row["accepted_patch_fraction"] > 0]
    if not tested:
        raise ValueError("no active lookup candidate on the development window")
    eligible = [row for row in tested if row["title_raw_mae"] <= baseline["title_raw_mae"] + 0.05
                and row["title_temporal"] <= baseline["title_temporal"] + 0.05]
    chosen = min(eligible or tested, key=lambda row: row["raw_mae"])
    frozen = {"chosen": chosen, "title_guard_passed": bool(eligible), "reference_range": REFERENCE,
              "development_range": [manifest[0]["frame"], manifest[-1]["frame"]],
              "development": development, "codebook_sha256": digest(args.out / "codebook.pt"),
              "patch_width": PATCH_WIDTH, "stride": STRIDE, "provenance": provenance,
              "roi_sha256": digest(ROI), "dev_cache_sha256": digest(args.cache / "S2-dev.pt")}
    write_json(args.out / "freeze.json", frozen)
    print(f"development frozen: {chosen}; title guard passed: {bool(eligible)}", flush=True)
    del matches, base, truth, pred

    phases, decoded, base, manifest = load_split(args.db, args.audit_cache, provenance, "S2-audit")
    matches = list(match_references(ref_phases, keys, values, phases, decoded))
    lookup, coverage = reconstruct(base, matches, chosen["limit"], chosen["strength"])
    del matches
    for phase, pred, strip in zip(phases, lookup.reshape(-1, 30, 16, 640), decoded):
        rows, cols = scatter_indices(phase)
        if not torch.equal(pred[:, rows[:, None], cols[None, :]], strip):
            raise ValueError("lookup changed the current phase's decoded samples")
    truth = labels(args.db, manifest)
    first, last = manifest[0]["frame"], manifest[-1]["frame"]
    reference = display(truth)
    export(args.out, "target", reference, first)
    methods, per_strip = {}, {}
    for name, pred in (("v12", base), ("v14-lookup", lookup)):
        rgb = display(pred)
        methods[name], per_strip[name], _ = scores(pred, truth, rgb, reference, regions)
        export(args.out, name, rgb, first)
        print(f"{name}: RAW MAE {methods[name]['raw']['mae_8bit']:.4f}", flush=True)
    difference = per_strip["v14-lookup"] - per_strip["v12"]
    report = {**frozen, "freeze_sha256": digest(args.out / "freeze.json"), "scored_range": [first, last],
              "scored_strips": len(manifest), "accepted_patch_fraction": coverage, "methods": methods,
              "audit_cache_sha256": digest(args.audit_cache / "S2-audit.pt"),
              "strips_better_than_v12": int((difference < 0).sum()),
              "strips_worse_than_v12": int((difference > 0).sum())}
    write_json(args.out / "report.json", report)
    with (args.out / "blocks.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(manifest):
            stream.write(json.dumps({**row, **{name: values[index].item() for name, values in per_strip.items()}}) + "\n")


if __name__ == "__main__":
    run_logged("evaluation", main)
