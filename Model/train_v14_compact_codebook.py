"""Learn and query patch features and compact DCT residual codes from one RAW phase."""
import argparse
from contextlib import ExitStack, closing
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from baseline import upscale
from evaluate_v10 import SingleVideo
from evaluate_v13 import ROI, display, scores, verify_video
from evaluate_v14 import export
from evaluate_v14_current_phase import AUDIT, REFERENCE, phase_ids, sample
from evaluate_v14_single_phase_detail import assert_samples
from evaluate_v14_strips import manifest
from evaluate_v14_v12_index import compare
from record_experiment import digest, run_logged, update_active_record, write_json
from train_v14_feature_codebook import DEV, observations
from v11.data import connect
from v11.phase import scatter_indices
from v13.data import atomic_save, fingerprints, labels


DEFAULT_WIDTH = 40
RANK = 8
CAPACITIES = (1, 4)
ROUNDS = 5
FORMAT = "v14-patch-cfa-dct-v1"
GATES = [(0, 0.)] + [(limit, strength) for limit in (4, 8, 12, 16, 24) for strength in (0.5, 1.)]


def architecture(width):
    return FORMAT if width == DEFAULT_WIDTH else f"{FORMAT}-w{width}"


def tiles_for(width):
    return 640 // width


def check_width(width):
    if width % 8 or width < 16 or 640 % width:
        raise ValueError("patch width must divide 640, be a multiple of 8 and at least 16")
    return width


def split_strips(tensor, width, tiles):
    return tensor.reshape(*tensor.shape[:-2], tensor.shape[-2], tiles, width).transpose(-3, -2)


def describe(samples, width, tiles):
    patches = split_strips(samples, width // 4, tiles)
    planes = F.pixel_unshuffle(patches.reshape(-1, 1, 4, width // 4).float(), 2)
    dx = planes.diff(dim=-1)
    dy = planes.diff(dim=-2)
    features = torch.cat((planes.mean(dim=(-2, -1)) - 128, dx.mean(dim=(-2, -1)) * 0.5,
                          dy.mean(dim=(-2, -1)) * 0.5, dx.abs().mean(dim=(-2, -1)) * 0.5), dim=-1)
    return features.reshape(*patches.shape[:-2], 16)


def dct_axis(length):
    freq = torch.arange(length, dtype=torch.float32)[:, None]
    coordinate = torch.arange(length, dtype=torch.float32) + 0.5
    basis = (freq * coordinate * (math.pi / length)).cos() * math.sqrt(2 / length)
    basis[0] /= math.sqrt(2)
    return basis


def dct_basis(width):
    rows, cols = dct_axis(8), dct_axis(width // 2)
    modes = sorted(((y + x, y, x) for y in range(len(rows)) for x in range(len(cols))))[:RANK]
    return torch.stack([(rows[y, :, None] * cols[x, None, :]).flatten() for _, y, x in modes])


def bilinear(samples, phases):
    return upscale(samples[:, None].float(), phases).round().byte()


def phase_data(raw, phase, basis, width, tiles):
    current = sample(raw, phase)
    low = current.reshape(-1, 4, 160)
    base = bilinear(low, torch.full((len(low),), phase)).reshape_as(raw)
    residual = split_strips(raw.float() - base.float(), width, tiles)
    planes = F.pixel_unshuffle(residual.reshape(-1, 1, 16, width), 2).flatten(-2)
    features = describe(current, width, tiles).reshape(len(raw), 30 * tiles, 16).transpose(0, 1)
    projected = (planes @ basis.T).reshape(len(raw), 30 * tiles, 4 * RANK).transpose(0, 1)
    return features, projected


def fit_round(features, residuals, keys, codes):
    nearest = torch.cdist(features, keys, compute_mode="donot_use_mm_for_euclid_dist").argmin(dim=-1)
    counts = torch.zeros(keys.shape[:2], dtype=torch.int32)
    counts.scatter_add_(1, nearest, torch.ones_like(nearest, dtype=torch.int32))
    return (average_groups(features, nearest, counts[..., None], keys),
            average_groups(residuals, nearest, counts[..., None], codes), counts)


def average_groups(data, groups, counts, fallback):
    index = groups[..., None].expand(-1, -1, data.shape[-1])
    totals = torch.zeros_like(fallback).scatter_add_(1, index, data)
    return torch.where(counts > 0, totals / counts.clamp_min(1), fallback)


def pack(phases, config):
    scale = torch.stack([codes.abs().amax(dim=(0, 1)) for _, codes in phases]).amax(dim=0).clamp_min(1e-6) / 127
    tiles = phases[0][0].shape[0] // 30
    keys = torch.cat([keys for keys, _ in phases]).reshape(16, 30, tiles, -1, 16)
    codes = torch.cat([codes for _, codes in phases]).reshape(16, 30, tiles, -1, 4 * RANK)
    return {"descriptor_keys": keys.round().clamp(-128, 127).char(),
            "residual_codes": (codes / scale).round().clamp(-127, 127).char(),
            "scales": scale, "config": config}


def retrieve(book, samples, positions, phases, basis):
    width = book["config"].get("patch_width", DEFAULT_WIDTH)
    tiles = tiles_for(width)
    query = describe(samples, width, tiles)
    keys = book["descriptor_keys"][phases, positions].float()
    values = book["residual_codes"][phases, positions]
    distance, index = (keys - query[:, :, None]).square().mean(dim=-1).sqrt().min(dim=-1)
    selected = values.gather(2, index[..., None, None].expand(-1, -1, 1, 4 * RANK)).squeeze(2)
    decoded = (selected.float() * book["scales"]).reshape(-1, 4, RANK) @ basis
    patches = F.pixel_shuffle(decoded.reshape(-1, 4, 8, width // 2), 2).reshape(-1, tiles, 16, width)
    return patches, distance


def reconstruct(base, patches, distance, phases, limit, strength):
    if strength == 0:
        return base
    weight = (1 - distance / limit).clamp_min(0) * strength
    delta = (patches * weight[..., None, None]).transpose(1, 2).reshape_as(base)
    for phase in phases.unique().tolist():
        rows, cols = scatter_indices(phase)
        indices = (phases == phase).nonzero().flatten()
        delta[indices[:, None, None], 0, rows[:, None], cols[None, :]] = 0
    return (base + delta).round().clamp(0, 255).byte()


def evaluation_inputs(db, frame_range):
    truth, current, frame_phases = observations(db, frame_range)
    phases = frame_phases.repeat_interleave(30)
    samples = current.flatten(0, 1)
    positions = torch.arange(30).repeat(len(frame_phases))
    return truth, samples, phases, positions, current, frame_phases


def calibrate(book, base, observations, basis, regions, baseline):
    truth, samples, phases, positions, _, _ = observations
    patches, distance = retrieve(book, samples, positions, phases, basis)
    options = []
    for limit, strength in GATES:
        pred = reconstruct(base, patches, distance, phases, limit, strength)
        options.append({"limit": limit, "strength": strength, **compare(pred, truth, regions)})
    eligible = [row for row in options if row["temporal"] <= baseline["temporal"] + 0.05 and
                row["title_temporal"] <= baseline["title_temporal"] + 0.05]
    return min(eligible, key=lambda row: row["raw_mae"])


def train(args):
    provenance = fingerprints(args.db, args.scene_plan)
    previous = json.loads(args.checkpoint.with_name("freeze.json").read_text(encoding="utf-8"))
    bank_sha = digest(args.checkpoint)
    if previous["provenance"] != provenance or previous["codebook_sha256"] != bank_sha:
        raise ValueError("raw bank does not match this database and scene plan")
    bank = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if tuple(bank["reference_range"]) != REFERENCE or not REFERENCE[1] < DEV[0] <= DEV[1] < AUDIT[0]:
        raise ValueError("expected separate reference, development and held-out windows")
    width, tiles = check_width(args.width), tiles_for(args.width)
    args.out.mkdir(parents=True)
    basis = dct_basis(width)
    dev = evaluation_inputs(args.db, DEV)
    reference = bank["raw"]
    config = {"architecture": architecture(width), "patch_width": width,
              "reference_range": REFERENCE, "development_range": DEV,
              "patch_dimensions": [width, 16], "descriptor_dimensions": 16,
              "residual_coefficients_per_patch": 4 * RANK,
              "entries_per_tile_options": CAPACITIES, "rounds_per_capacity": ROUNDS,
              "epochs": len(CAPACITIES) * ROUNDS,
              "training_samples": len(reference) * 30, "validation_samples": len(dev[0]),
              "provenance": provenance, "raw_bank_sha256": bank_sha}
    write_json(args.out / "config.json", config)
    base = bilinear(dev[1], dev[2])
    roi = json.loads(ROI.read_text(encoding="utf-8"))["regions"]
    baseline = compare(base, dev[0], roi)
    best_global, epoch = float("inf"), 0
    book = None
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for k in CAPACITIES:
            start = torch.linspace(0, len(reference) - 1, k).round().long()
            keys = codes = None
            best = float("inf")
            for iteration in range(1, ROUNDS + 1):
                # Each phase is fitted on its own rows, so they are processed one at a
                # time and rebuilt per round to keep the working set inside memory.
                phase_keys, phase_codes, phase_counts = [], [], []
                for phase in range(16):
                    features, residuals = phase_data(reference, phase, basis, width, tiles)
                    if iteration == 1:
                        seed_keys, seed_codes = features[:, start], residuals[:, start]
                    else:
                        seed_keys, seed_codes = keys[phase], codes[phase]
                    fitted_keys, fitted_codes, counts = fit_round(features, residuals, seed_keys, seed_codes)
                    phase_keys.append(fitted_keys)
                    phase_codes.append(fitted_codes)
                    phase_counts.append(counts)
                    del features, residuals
                keys, codes = phase_keys, phase_codes
                book = pack(list(zip(keys, codes)), {**config, "entries_per_tile": k})
                gate = calibrate(book, base, dev, basis, roi, baseline)
                epoch += 1
                row = {"epoch": epoch, "iteration": iteration, "entries_per_tile": k,
                       "validation": {"mae_8bit": gate["raw_mae"], **gate},
                       "patches_clustered": int(sum(int(count.sum()) for count in phase_counts)),
                       "active_clusters": int(sum(int((count > 0).sum()) for count in phase_counts))}
                book.update({"gate": {key: gate[key] for key in ("limit", "strength")}, "selected_metrics": row})
                log.write(json.dumps(row) + "\n")
                log.flush()
                update_active_record()
                if gate["raw_mae"] < best:
                    best = gate["raw_mae"]
                    atomic_save(book, args.out / f"k{k:02d}.pt")
                if gate["raw_mae"] < best_global:
                    best_global = gate["raw_mae"]
                    atomic_save(book, args.out / "best-mae.pt")
            print(f"K={k}: development best RAW MAE {best:.4f}", flush=True)
    atomic_save(book, args.out / "last.pt")


def load_book(path):
    book = torch.load(path, map_location="cpu", weights_only=True)
    config = book["config"]
    width = config.get("patch_width", DEFAULT_WIDTH)
    tiles = tiles_for(width)
    shape = (16, 30, tiles, config["entries_per_tile"])
    if (config["architecture"] != architecture(width) or book["descriptor_keys"].shape != (*shape, 16) or
            book["residual_codes"].shape != (*shape, 4 * RANK) or book["scales"].shape != (4 * RANK,) or
            book["descriptor_keys"].dtype != torch.int8 or book["residual_codes"].dtype != torch.int8):
        raise ValueError("invalid compact v14 codebook")
    return book


def observe_batch(raw, frame_phases):
    current = torch.stack([sample(frame, phase) for frame, phase in zip(raw, frame_phases)])
    phases = frame_phases.repeat_interleave(30)
    positions = torch.arange(30).repeat(len(frame_phases))
    return current, phases, positions


def whole_predictions(raw, frame_phases, book, basis):
    current, phases, positions = observe_batch(raw, frame_phases)
    samples = current.flatten(0, 1)
    base = bilinear(samples, phases)
    patches, distance = retrieve(book, samples, positions, phases, basis)
    pred = reconstruct(base, patches, distance, phases, **book["gate"])
    return base, pred, current


def whole_outputs(out, raw, frame_phases, book, basis):
    names = ("target", "bilinear", "compact")
    absolute_error = dict.fromkeys(names[1:], 0)
    exact_strips = dict.fromkeys(names[1:], 0)
    exact_frames = dict.fromkeys(names[1:], 0)
    with ExitStack() as stack:
        videos = {}
        for name in names:
            video = SingleVideo(out, name)
            stack.callback(video.close)
            videos[name] = video
        for batch, phases in zip(raw.split(10), frame_phases.split(10)):
            base, pred, current = whole_predictions(batch, phases, book, basis)
            assert_samples(pred, phases, current)
            truth = batch.reshape(-1, 1, 16, 640)
            for name, result in zip(names, (truth, base, pred)):
                if name != "target":
                    error = (result.short() - truth.short()).abs().reshape(-1, 30, 16 * 640)
                    absolute_error[name] += int(error.sum())
                    exact = (error == 0).all(dim=-1)
                    exact_strips[name] += int(exact.sum())
                    exact_frames[name] += int(exact.all(dim=-1).sum())
                for frame in display(result):
                    videos[name].write(frame)
    for name in names:
        verify_video(out / f"{name}.mp4", len(raw))
    return {name: {"raw_mae": absolute_error[name] / raw.numel(), "exact_strips": exact_strips[name],
                   "exact_frames": exact_frames[name]} for name in names[1:]}


def whole_video(args):
    provenance = fingerprints(args.db, args.scene_plan)
    pilot = load_book(args.checkpoint)
    if pilot["config"]["provenance"] != provenance:
        raise ValueError("pilot and full-clip training must use the same input dataset")
    width = pilot["config"].get("patch_width", DEFAULT_WIDTH)
    if width != check_width(args.width):
        raise ValueError("whole-video width must match the pilot codebook")
    tiles = tiles_for(width)
    k = pilot["config"]["entries_per_tile"]
    args.out.mkdir(parents=True)
    with closing(connect(args.db)) as conn:
        first, last = conn.execute("SELECT min(frame_index), max(frame_index) FROM samples").fetchone()
    if first is None or last is None:
        raise ValueError("input contains no frames")
    phases = torch.tensor(phase_ids(args.db, first, last))
    raw = labels(args.db, manifest(first, last)).reshape(-1, 30, 16, 640)
    check, check_phase = raw[::15], phases[::15]
    basis = dct_basis(width)
    features, residuals, centers, values = [], [], [], []
    for phase in range(16):
        present = raw[phases == phase]
        if len(present) < k:
            raise ValueError("full clip lacks sufficient examples at one phase")
        key, code = phase_data(present, phase, basis, width, tiles)
        features.append(key)
        residuals.append(code)
        start = torch.linspace(0, len(present) - 1, k).round().long()
        centers.append(key[:, start])
        values.append(code[:, start])
    config = {"architecture": architecture(width), "patch_width": width, "reference_range": [first, last],
              "entries_per_tile": k, "training_samples": len(raw) * 30, "validation_samples": len(check) * 30,
              "validation_scope": "in-corpus checkpoints on every fifteenth source frame",
              "training_phases": "each frame uses its recorded phase", "epochs": ROUNDS, "provenance": provenance,
              "pilot_sha256": digest(args.checkpoint)}
    write_json(args.out / "config.json", config)
    target = check.reshape(-1, 1, 16, 640)
    best = float("inf")
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, ROUNDS + 1):
            updates = [fit_round(*row) for row in zip(features, residuals, centers, values)]
            centers, values, counts = zip(*updates)
            book = pack(list(zip(centers, values)), config)
            book["gate"] = pilot["gate"]
            _, pred, _ = whole_predictions(check, check_phase, book, basis)
            mae = (pred.short() - target.short()).abs().double().mean().item()
            row = {"epoch": epoch, "validation": {"mae_8bit": mae},
                   "patches_clustered": sum(int(count.sum()) for count in counts)}
            book["selected_metrics"] = row
            log.write(json.dumps(row) + "\n")
            log.flush()
            update_active_record()
            if mae < best:
                best = mae
                atomic_save(book, args.out / "best-mae.pt")
    atomic_save(book, args.out / "last.pt")
    best_path = args.out / "best-mae.pt"
    best_book = load_book(best_path)
    print(f"whole video: trained {len(raw)} frames, exporting original and both reconstructions", flush=True)
    result = whole_outputs(args.out, raw, phases, best_book, basis)
    report = {"frame_range": [first, last], "frames": len(raw), "samples": len(raw) * 30,
              "scope": "in-corpus whole clip", "codebook_bytes": best_path.stat().st_size,
              "codebook_sha256": digest(best_path), "gate": best_book["gate"], "methods": result}
    write_json(args.out / "report.json", report)
    print(result, flush=True)


def evaluate(args):
    provenance = fingerprints(args.db, args.scene_plan)
    args.out.mkdir(parents=True)
    truth, samples, phases, positions, current, frame_phases = evaluation_inputs(args.db, AUDIT)
    base = bilinear(samples, phases)
    predictions = {"bilinear": base}
    books = {}
    basis = dct_basis(check_width(args.width))
    for path in args.checkpoint:
        book = load_book(path)
        if book["config"]["provenance"] != provenance:
            raise ValueError("compact codebook does not match the selected database")
        if book["config"].get("patch_width", DEFAULT_WIDTH) != args.width:
            raise ValueError("compact codebook patch width does not match the requested width")
        name = f"compact-k{book['config']['entries_per_tile']:02d}"
        patches, distance = retrieve(book, samples, positions, phases, basis)
        predictions[name] = reconstruct(base, patches, distance, phases, **book["gate"])
        books[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path),
                       "gate": book["gate"], "selected_metrics": book["selected_metrics"]}
    roi = json.loads(ROI.read_text(encoding="utf-8"))["regions"]
    rgb = display(truth)
    export(args.out, "target", rgb, AUDIT[0])
    methods, errors = {}, {}
    for name, pred in predictions.items():
        assert_samples(pred, frame_phases, current)
        result_rgb = display(pred)
        methods[name], errors[name], _ = scores(pred, truth, result_rgb, rgb, roi)
        export(args.out, name, result_rgb, AUDIT[0])
        print(f"{name}: RAW MAE {methods[name]['raw']['mae_8bit']:.4f}", flush=True)
    write_json(args.out / "report.json", {"start_frame": AUDIT[0], "first_evaluated_frame": AUDIT[0],
                                        "last_evaluated_frame": AUDIT[1], "samples": len(truth),
                                        "provenance": provenance, "books": books, "methods": methods})
    with (args.out / "blocks.jsonl").open("w", encoding="utf-8") as log:
        for i, row in enumerate(manifest(*AUDIT)):
            log.write(json.dumps({**row, **{name: error[i].item() for name, error in errors.items()}}) + "\n")


def predict(args):
    data = bytearray(args.input.read_bytes())
    if len(data) != 640:
        raise ValueError("expected a 160x4 RAW8 sample (640 bytes)")
    book = load_book(args.checkpoint)
    width = book["config"].get("patch_width", DEFAULT_WIDTH)
    samples = torch.frombuffer(data, dtype=torch.uint8).reshape(1, 4, 160)
    phases, positions = torch.tensor([args.phase - 1]), torch.tensor([args.position])
    base = bilinear(samples, phases)
    patches, distance = retrieve(book, samples, positions, phases, dct_basis(width))
    output = reconstruct(base, patches, distance, phases, **book["gate"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        stream.write(output.numpy().tobytes())
    print(f"{output.numel()} RAW8 bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "predict", "whole-video"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--checkpoint", type=Path, required=True, nargs="+" if name == "evaluate" else None)
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                         help="patch width in full resolution pixels")
        if name != "predict":
            cmd.add_argument("--db", type=Path, required=True)
            cmd.add_argument("--scene-plan", type=Path, required=True)
    commands.choices["predict"].add_argument("--input", type=Path, required=True)
    commands.choices["predict"].add_argument("--position", type=int, choices=range(30), required=True)
    commands.choices["predict"].add_argument("--phase", type=int, choices=range(1, 17), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    {"train": train, "evaluate": evaluate, "predict": predict, "whole-video": whole_video}[args.command](args)


if __name__ == "__main__":
    kind = {"evaluate": "evaluation", "predict": "inference"}.get(next(iter(sys.argv[1:]), ""), "training")
    with torch.inference_mode():
        run_logged(kind, main)
