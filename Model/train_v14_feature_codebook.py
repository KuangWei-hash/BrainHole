"""Train, evaluate, or query v14's whole-strip feature codebook."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from baseline import upscale
from evaluate_v13 import ROI, display, scores
from evaluate_v14 import export
from evaluate_v14_current_phase import AUDIT, REFERENCE, lookup_strip, phase_ids, sample
from evaluate_v14_strips import manifest
from record_experiment import digest, run_logged, update_active_record, write_json
from train_v14_lossy import measurements
from v13.data import atomic_save, fingerprints, labels


DEV = (750, 779)
CAPACITIES = (4, 8, 16)
ROUNDS = 6
FORMAT = "v14-cfa-strip-features-v1"


def describe(sampling):
    """Position-sensitive intensity, signed gradients and texture per CFA plane."""
    planes = F.pixel_unshuffle(sampling.reshape(-1, 1, 4, 160).float(), 2)
    dx = F.pad(planes.diff(dim=-1), (1, 0, 0, 0), mode="replicate")
    dy = planes.diff(dim=-2)
    intensity = F.avg_pool2d(planes, (2, 5))
    horizontal = F.avg_pool2d(dx, (2, 5))
    vertical = F.avg_pool2d(dy, (1, 5))
    texture = F.avg_pool2d(dx.abs(), (2, 5))
    descriptor = torch.cat((intensity, 0.75 * horizontal, 0.5 * vertical, 0.5 * texture), dim=1)
    return descriptor.reshape(*sampling.shape[:-2], -1)


def nearest_feature(query, keys):
    return (keys.float() - query.unsqueeze(-2)).abs().sum(dim=-1).argmin(dim=-1)


def predict_strip(book, current, position, phase):
    """Select a dictionary entry using one current 4x160 RAW8 phase sample."""
    if current.shape != (4, 160) or current.dtype != torch.uint8 or not 0 <= position < 30 or not 0 <= phase < 16:
        raise ValueError("expected one 160x4 RAW8 sample, position in 0..29 and phase in 0..15")
    keys = book["keys"][position, :, phase]
    index = nearest_feature(describe(current), keys)
    return book["raw"][position, index], int(index)


def predict_frames(book, queries, phases):
    keys = book["keys"].permute(2, 0, 1, 3)[phases]
    indices = nearest_feature(queries, keys)
    selected = book["raw"][torch.arange(30), indices]
    return selected.reshape(-1, 1, 16, 640), indices


def load_book(path):
    book = torch.load(path, map_location="cpu", weights_only=True)
    raw, keys = book["raw"], book["keys"]
    if (book["config"]["architecture"] != FORMAT or raw.ndim != 4 or raw.shape[0] != 30 or
            raw.shape[2:] != (16, 640) or raw.dtype != torch.uint8 or keys.shape != (*raw.shape[:2], 16, 256) or
            keys.dtype != torch.float16 or not torch.isfinite(keys).all()):
        raise ValueError("invalid v14 feature codebook")
    return book


def observations(db, frame_range):
    truth = labels(db, manifest(*frame_range))
    phases = torch.tensor(phase_ids(db, *frame_range))
    current = torch.stack([sample(frame, phase) for frame, phase in zip(truth.reshape(-1, 30, 16, 640), phases)])
    return truth, current, phases


def initial_indices(embedding):
    distances = torch.cdist(embedding, embedding, p=1)
    count, samples, _ = distances.shape
    nearest = torch.full((count, samples), float("inf"))
    used = torch.zeros_like(nearest, dtype=torch.bool)
    positions = torch.arange(count)
    ranking = []
    for _ in range(max(CAPACITIES)):
        cost = torch.minimum(distances, nearest[:, None, :]).sum(dim=-1)
        selected = cost.masked_fill(used, float("inf")).argmin(dim=-1)
        used[positions, selected] = True
        nearest = torch.minimum(nearest, distances[positions, selected])
        ranking.append(selected)
    return torch.stack(ranking, dim=1)


def update_dictionary(features, raw, keys, entries):
    groups = torch.cdist(features.flatten(-2), keys.flatten(-2), p=1).argmin(dim=-1)
    updated_keys, updated_entries = keys.clone(), entries.clone()
    sizes = torch.zeros(keys.shape[:2], dtype=torch.int64)
    for position in range(len(keys)):
        for i in range(keys.shape[1]):
            members = groups[position] == i
            sizes[position, i] = members.sum()
            if members.any():
                updated_keys[position, i] = features[position, members].median(dim=0).values
                updated_entries[position, i] = raw[position, members].median(dim=0).values
    return updated_keys, updated_entries, sizes


def raw_metrics(pred, target):
    distance = (pred.short() - target.short()).abs().sum(dim=(-2, -1))
    return measurements(distance, 16 * 640)


def train(args):
    provenance = fingerprints(args.db, args.scene_plan)
    source = json.loads(args.checkpoint.with_name("freeze.json").read_text(encoding="utf-8"))
    bank_sha = digest(args.checkpoint)
    if source["provenance"] != provenance or source["codebook_sha256"] != bank_sha:
        raise ValueError("the source RAW entries do not match this database and scene plan")
    bank = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if tuple(bank["reference_range"]) != REFERENCE or not REFERENCE[1] < DEV[0] <= DEV[1] < AUDIT[0]:
        raise ValueError("expected disjoint reference, development and evaluation windows")
    raw = bank["raw"].transpose(0, 1).contiguous()
    features = torch.stack([describe(sample(raw, phase)) for phase in range(16)], dim=2)
    ranking = initial_indices(features.flatten(-2))
    phases = torch.tensor(phase_ids(args.db, *REFERENCE))
    train_obs = features.permute(1, 2, 0, 3)[torch.arange(len(phases)), phases]
    train_raw = bank["raw"].reshape(-1, 1, 16, 640)
    dev_raw, dev_obs, dev_phases = observations(args.db, DEV)
    dev_features = describe(dev_obs)
    config = {"architecture": FORMAT, "reference_range": REFERENCE, "development_range": DEV,
              "training_samples": len(train_raw), "validation_samples": len(dev_raw), "feature_dimensions_per_phase": 256,
              "features": ["CFA mean", "signed horizontal gradient", "signed vertical gradient", "horizontal texture energy"],
              "phase_feature_weights": [1, 0.75, 0.5, 0.5], "entries_per_position_options": CAPACITIES,
              "rounds_per_capacity": ROUNDS, "epochs": len(CAPACITIES) * ROUNDS,
              "provenance": provenance, "raw_bank_sha256": bank_sha}
    args.out.mkdir(parents=True)
    write_json(args.out / "config.json", config)
    positions = torch.arange(len(raw))[:, None]
    best_mae, epoch = float("inf"), 0
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as metrics:
        for count in CAPACITIES:
            ids = ranking[:, :count]
            keys, entries = features[positions, ids], raw[positions, ids]
            for iteration in range(1, ROUNDS + 1):
                keys, entries, sizes = update_dictionary(features, raw, keys, entries)
                book = {"keys": keys.half(), "raw": entries, "cluster_sizes": sizes,
                        "config": {**config, "entries_per_position": count}}
                fitted, _ = predict_frames(book, train_obs, phases)
                dev_pred, _ = predict_frames(book, dev_features, dev_phases)
                epoch += 1
                row = {"epoch": epoch, "iteration": iteration, "entries_per_position": count,
                       "training": raw_metrics(fitted, train_raw), "validation": raw_metrics(dev_pred, dev_raw),
                       "members_assigned": int(sizes.sum()), "active_clusters": int((sizes > 0).sum())}
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                update_active_record()
                if row["validation"]["mae_8bit"] < best_mae:
                    best_mae = row["validation"]["mae_8bit"]
                    atomic_save({**book, "selected_metrics": row}, args.out / "best-mae.pt")
            print(f"K={count}: development RAW MAE {row['validation']['mae_8bit']:.4f}, "
                  f"active clusters {row['active_clusters']}/{30 * count}", flush=True)
    atomic_save({**book, "selected_metrics": row}, args.out / "last.pt")


def evaluate(args):
    book = load_book(args.checkpoint)
    provenance = fingerprints(args.db, args.scene_plan)
    if book["config"]["provenance"] != provenance:
        raise ValueError("feature codebook does not match this database and scene plan")
    args.out.mkdir(parents=True)
    first, last = AUDIT
    truth, current, phases = observations(args.db, AUDIT)
    prediction, selected = predict_frames(book, describe(current), phases)
    values = book["raw"]
    phase_keys = [sample(values, phase).short() for phase in range(16)]
    raw_lookup = torch.stack([lookup_strip(values, phase_keys[phase], strips)[0]
                              for strips, phase in zip(current, phases)]).reshape_as(truth)
    base = upscale(current.reshape(-1, 1, 4, 160).float(), phases.repeat_interleave(30)).round().byte()
    regions = json.loads(ROI.read_text(encoding="utf-8"))["regions"]
    reference = display(truth)
    export(args.out, "target", reference, first)
    methods, errors = {}, {}
    for name, result in (("feature-lookup", prediction), ("raw-pixel-lookup-same-dictionary", raw_lookup), ("bilinear", base)):
        rgb = display(result)
        methods[name], errors[name], _ = scores(result, truth, rgb, reference, regions)
        export(args.out, name, rgb, first)
        print(f"{name}: RAW MAE {methods[name]['raw']['mae_8bit']:.4f}", flush=True)
    report = {"start_frame": first, "first_evaluated_frame": first, "last_evaluated_frame": last,
              "samples": len(truth), "codebook_sha256": digest(args.checkpoint), "codebook_bytes": args.checkpoint.stat().st_size,
              "feature_bytes": book["keys"].numel() * book["keys"].element_size(), "raw_bytes": values.numel(),
              "training_selection": book["selected_metrics"], "provenance": provenance, "methods": methods}
    write_json(args.out / "report.json", report)
    with (args.out / "blocks.jsonl").open("w", encoding="utf-8") as stream:
        for i, row in enumerate(manifest(first, last)):
            stream.write(json.dumps({**row, "entry": int(selected.flatten()[i]),
                                     **{name: value[i].item() for name, value in errors.items()}}) + "\n")


def predict(args):
    payload = bytearray(args.input.read_bytes())
    if len(payload) != 640:
        raise ValueError("input RAW8 file must contain exactly 160x4 bytes")
    book = load_book(args.checkpoint)
    result, index = predict_strip(book, torch.frombuffer(payload, dtype=torch.uint8).reshape(4, 160),
                                  args.position, args.phase - 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        stream.write(result.numpy().tobytes())
    print(f"entry {index}, {result.numel()} RAW8 bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate", "predict"):
        options = commands.add_parser(command)
        options.add_argument("--checkpoint", type=Path, required=True)
        options.add_argument("--out", type=Path, required=True)
        if command != "predict":
            options.add_argument("--db", type=Path, required=True)
            options.add_argument("--scene-plan", type=Path, required=True)
    commands.choices["predict"].add_argument("--input", type=Path, required=True)
    commands.choices["predict"].add_argument("--position", type=int, choices=range(30), required=True)
    commands.choices["predict"].add_argument("--phase", type=int, choices=range(1, 17), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    {"train": train, "evaluate": evaluate, "predict": predict}[args.command](args)


if __name__ == "__main__":
    kind = {"train": "training", "evaluate": "evaluation", "predict": "inference"}.get(next(iter(sys.argv[1:]), ""), "training")
    with torch.inference_mode():
        run_logged(kind, main)
