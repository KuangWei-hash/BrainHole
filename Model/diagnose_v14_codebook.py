"""Unified v14 compact codebook diagnosis.

Answers, on one codebook and one evaluation window, both questions that the
two earlier scripts asked separately:

1. Where does the RAW MAE gain come from -- genuine high-frequency texture
   recovery, or just low-frequency smoothing? (was probe_v14_compact_frequency.py)
2. What is the error made of -- descriptor/matching error, table coverage, or
   representation capacity of the truncated DCT residuals?
   (was diagnose_v14_matching_limit.py)

Both share the same machinery: the compact codebook stores, per patch, a
16-dim descriptor (lookup key) plus 4*RANK int8-quantised low-frequency DCT
residual coefficients (stored value) against a phase-aware bilinear base.
The two questions are two different decompositions of the same number, so
they are answered together from one pass over the data. The superseded
scripts were removed after a 29-point cross-check confirmed every number
they had recorded is reproduced exactly.

The `oracle-*` variants use ground-truth information and are upper-bound
measurements only; they are not deployable methods.

Signal source is explicit: --signal ideal uses the clean ground-truth-derived
160x4 sample (what the BBB-era scripts did, an input a real receiver never
has), while --signal real reads the receiver-visible `decoded` column (the
actual JPEG XS Q=5 round-trip).
"""
import argparse
import json
from contextlib import closing
from pathlib import Path

import torch
from torch.nn import functional as F

from evaluate_v13 import ROI, display, scores
from evaluate_v14 import export
from evaluate_v14_current_phase import phase_ids
from evaluate_v14_raw import sample
from evaluate_v14_strips import manifest
from evaluate_v14_v12_index import compare
from record_experiment import digest, run_logged, write_json
import v14
from v11.data import connect
from v13.data import fingerprints, labels


CHUNK = 450


def load_range(db, first, last, signal):
    """Return (raw_strips, low_samples, phases) for a frame range.

    low_samples is the (N, 4, 160) per-strip input a receiver would hold:
    either the clean ground-truth-derived sample (ideal) or the real decoded
    column (real).
    """
    phases = phase_ids(db, first, last)
    raw = labels(db, manifest(first, last)).reshape(-1, 30, 16, 640)
    frames = raw.shape[0]
    if signal == "ideal":
        low = torch.stack([sample(frame, phase)
                           for frame, phase in zip(raw, phases)]).reshape(-1, 4, 160)
        return raw, low, torch.tensor(phases)
    with closing(connect(db)) as conn:
        rows = conn.execute("SELECT frame_index, block_index, decoded FROM samples "
                            "WHERE frame_index BETWEEN ? AND ?", (first, last)).fetchall()
    low = torch.zeros((frames, 30, 4, 160), dtype=torch.uint8)
    for frame, block, decoded in rows:
        low[frame - first, block] = torch.frombuffer(bytearray(decoded), dtype=torch.uint8).reshape(4, 160)
    return raw, low.reshape(-1, 4, 160), torch.tensor(phases)


def entry_patches(book, phases, positions, basis, width, tiles, rank):
    """Dequantised residual patch of every stored entry: (M, tiles, k, 16, width)."""
    codes = book["residual_codes"][phases, positions].float() * book["scales"]
    decoded = codes.reshape(-1, 4, rank) @ basis
    shuffled = F.pixel_shuffle(decoded.reshape(-1, 4, 8, width // 2), 2)
    return shuffled.reshape(len(phases), tiles, codes.shape[2], 16, width)


def dct_residual(patches, book, basis, width, rank):
    """Project true residual patches onto the stored basis and quantise like the codebook."""
    scale = book["scales"].reshape(4, rank)
    planes = F.pixel_unshuffle(patches.reshape(-1, 1, 16, width), 2).flatten(-2)
    codes = ((planes @ basis.T) / scale).round().clamp(-127, 127) * scale
    decoded = codes.reshape(-1, 4, rank) @ basis
    shuffled = F.pixel_shuffle(decoded.reshape(-1, 4, 8, width // 2), 2)
    return shuffled.reshape(*patches.shape)


def dct_residual_float(patches, basis, width, rank):
    """Same projection without the int8 quantisation, to separate truncation from rounding."""
    planes = F.pixel_unshuffle(patches.reshape(-1, 1, 16, width), 2).flatten(-2)
    decoded = (planes @ basis.T).reshape(-1, 4, rank) @ basis
    shuffled = F.pixel_shuffle(decoded.reshape(-1, 4, 8, width // 2), 2)
    return shuffled.reshape(*patches.shape)


def gather(patches, index):
    shape = (-1, -1, 1, 16, patches.shape[-1])
    return patches.gather(2, index[:, :, None, None, None].expand(*shape)).squeeze(2)


def frequency_report(predictions, truth, phases, base):
    """Low/high frequency split of each prediction against the truth labels."""
    roi = json.loads(ROI.read_text(encoding="utf-8"))["regions"]
    reference = display(truth)
    rows = {}
    for name, value in predictions.items():
        rgb = display(value)
        result, _, _ = scores(value, truth, rgb, reference, roi)
        rows[name] = {
            "raw_mae": result["raw"]["mae_8bit"],
            "low_frequency_mae": result["low_frequency_mae"],
            "high_frequency_mae": result["high_frequency_mae"],
            "gradient_x_mae": result["gradient_x_mae"],
            "gradient_y_mae": result["gradient_y_mae"],
            "exact_raw_fraction": result["exact_raw_fraction"],
        }
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--range", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"),
                        help="evaluation frame window; defaults to the codebook's test range")
    parser.add_argument("--signal", choices=("ideal", "real"), default="ideal",
                        help="input signal: ideal (ground-truth-derived) or real (decoded column)")
    parser.add_argument("--no-video", action="store_true", help="skip video export")
    args = parser.parse_args()
    torch.set_num_threads(4)
    provenance = fingerprints(args.db, args.scene_plan)
    book = v14.load_book(args.checkpoint)
    config = book["config"]
    # Only BBB-era codebooks record a provenance fingerprint; the demo scripts'
    # codebooks do not, so the check is enforced only when one is present.
    stored = config.get("provenance")
    if stored is not None and stored != provenance:
        raise ValueError("compact codebook does not match the selected database")
    width = config.get("patch_width", v14.DEFAULT_WIDTH)
    tiles = v14.tiles_for(width)
    # The codebook's own rank must drive the DCT math; the module default (8)
    # only applies to codebooks built with it.
    rank = config.get("dct_rank", v14.RANK)
    basis = v14.dct_basis(width, rank)
    gate = book["gate"]
    first, last = args.range if args.range else config["test_range"]

    raw, low, phases = load_range(args.db, first, last, args.signal)
    samples = low
    positions = torch.arange(30).repeat(len(phases))
    flat_phases = phases.repeat_interleave(30)
    base = v14.bilinear(samples, flat_phases)
    truth = raw.reshape(-1, 1, 16, 640)
    residual = (truth.float() - base.float()).squeeze(1)

    predictions = {"bilinear": base}
    totals = {"amplitude": residual.abs().sum().item(), "amplitude_count": residual.numel()}
    for start in range(0, len(samples), CHUNK):
        stop = min(start + CHUNK, len(samples))
        chunk_phases, chunk_positions = flat_phases[start:stop], positions[start:stop]
        chunk_samples, chunk_base = samples[start:stop], base[start:stop]
        true_tiles = v14.split_strips(residual[start:stop], width, tiles)
        patches = entry_patches(book, chunk_phases, chunk_positions, basis, width, tiles, rank)
        descriptors = book["descriptor_keys"][chunk_phases, chunk_positions].float()
        query = v14.describe(chunk_samples, width, tiles)
        distance = (descriptors - query[:, :, None]).square().mean(dim=-1).sqrt()
        actual_index = distance.argmin(dim=-1)
        oracle_index = (patches - true_tiles[:, :, None]).abs().mean(dim=(-1, -2)).argmin(dim=-1)
        actual_patches = gather(patches, actual_index)
        oracle_patches = gather(patches, oracle_index)
        actual_distance = distance.gather(2, actual_index[:, :, None]).squeeze(2)
        oracle_distance = distance.gather(2, oracle_index[:, :, None]).squeeze(2)
        # Gate coverage: fraction of patches whose lookup distance falls inside
        # the gate limit, i.e. that actually receive any codebook content.
        inside = (actual_distance < gate["limit"]).sum().item()
        totals["gate_covered"] = totals.get("gate_covered", 0) + inside
        totals["gate_total"] = totals.get("gate_total", 0) + actual_distance.numel()
        dct_patches = dct_residual(true_tiles, book, basis, width, rank)
        float_patches = dct_residual_float(true_tiles, basis, width, rank)
        totals["energy_full"] = totals.get("energy_full", 0.) + (true_tiles.double() ** 2).sum().item()
        totals["energy_kept"] = totals.get("energy_kept", 0.) + (float_patches.double() ** 2).sum().item()
        totals["energy_kept_int8"] = totals.get("energy_kept_int8", 0.) + (dct_patches.double() ** 2).sum().item()
        for name, value in (("actual", actual_patches), ("oracle", oracle_patches),
                            ("dct", dct_patches), ("dct_float", float_patches)):
            totals[f"{name}_sum"] = totals.get(f"{name}_sum", 0.) + (value - true_tiles).abs().sum().item()
            totals[f"{name}_count"] = totals.get(f"{name}_count", 0) + value.numel()
        for name, value, dist, limit, strength in (
                ("actual", actual_patches, actual_distance, gate["limit"], gate["strength"]),
                ("oracle-entry", oracle_patches, oracle_distance, gate["limit"], gate["strength"]),
                ("oracle-entry-full", oracle_patches, oracle_distance, 1e9, 1.),
                ("oracle-dct", dct_patches, torch.zeros_like(oracle_distance), 1e9, 1.),
                ("oracle-dct-gated", dct_patches, torch.zeros_like(oracle_distance),
                 gate["limit"], gate["strength"]),
                ("oracle-exact", true_tiles, torch.zeros_like(oracle_distance), 1e9, 1.)):
            output = v14.reconstruct(chunk_base, value, dist, chunk_phases, limit, strength)
            predictions.setdefault(name, []).append(output)

    predictions = {name: (value if isinstance(value, torch.Tensor) else torch.cat(value))
                   for name, value in predictions.items()}

    roi = json.loads(ROI.read_text(encoding="utf-8"))["regions"]
    reference = display(truth)
    methods, errors = {}, {}
    if not args.no_video:
        args.out.mkdir(parents=True, exist_ok=True)
        export(args.out, "target", reference, first)
    for name, pred in predictions.items():
        result_rgb = display(pred)
        methods[name], errors[name], _ = scores(pred, truth, result_rgb, reference, roi)
        methods[name].update(compare(pred, truth, roi))
        if not args.no_video:
            export(args.out, name, result_rgb, first)
        print(f"{name}: RAW MAE {methods[name]['raw']['mae_8bit']:.4f}", flush=True)

    frequency = frequency_report(predictions, truth, phases, base)
    residual_error = {"amplitude": totals["amplitude"] / totals["amplitude_count"]}
    for name in ("actual", "oracle", "dct", "dct_float"):
        residual_error[f"{name}_residual_mae"] = totals[f"{name}_sum"] / totals[f"{name}_count"]
    residual_error["energy_kept_ratio"] = totals["energy_kept"] / totals["energy_full"]
    residual_error["energy_kept_int8_ratio"] = totals["energy_kept_int8"] / totals["energy_full"]
    residual_error["gate_coverage"] = totals["gate_covered"] / totals["gate_total"]

    # Decomposition of the actual path's error, in RAW MAE units.
    actual_mae = methods["actual"]["raw"]["mae_8bit"]
    matching_mae = actual_mae - methods["oracle-entry"]["raw"]["mae_8bit"]
    coverage_mae = methods["oracle-entry"]["raw"]["mae_8bit"] - methods["oracle-dct"]["raw"]["mae_8bit"]
    representation_mae = methods["oracle-dct"]["raw"]["mae_8bit"]
    decomposition = {"matching": matching_mae, "coverage": coverage_mae,
                     "representation": representation_mae, "total": actual_mae}

    # Gate coverage: fraction of patches whose lookup distance is within the
    # gate limit, i.e. that actually receive any codebook content.
    report = {"start_frame": first, "last_evaluated_frame": last, "samples": len(truth),
              "provenance": provenance, "patch_width": width, "dct_rank": rank,
              "entries_per_tile": config["entries_per_tile"], "gate": gate,
              "signal": args.signal,
              "codebook_sha256": digest(args.checkpoint),
              "methods": methods, "residual": residual_error,
              "frequency": frequency, "decomposition": decomposition}
    write_json(args.out / "report.json", report)
    if not args.no_video:
        with (args.out / "blocks.jsonl").open("w", encoding="utf-8") as log:
            for i, row in enumerate(manifest(*(first, last))):
                log.write(json.dumps({**row, **{name: error[i].item() for name, error in errors.items()}}) + "\n")
    print(json.dumps(residual_error, indent=1), flush=True)
    print(json.dumps(decomposition, indent=1), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        run_logged("evaluation", main)
