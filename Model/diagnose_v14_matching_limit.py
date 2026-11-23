"""Separate v14 compact codebook error into descriptor matching, table coverage and representation loss."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from evaluate_v13 import ROI, display, scores
from evaluate_v14 import export
from evaluate_v14_current_phase import AUDIT
from evaluate_v14_single_phase_detail import assert_samples
from evaluate_v14_strips import manifest
from evaluate_v14_v12_index import compare
from record_experiment import digest, run_logged, update_active_record, write_json
from train_v14_compact_codebook import (DEFAULT_WIDTH, RANK, bilinear, describe, dct_basis, load_book,
                                         reconstruct, split_strips, tiles_for)
from train_v14_feature_codebook import observations
from v13.data import fingerprints


CHUNK = 450


def entry_patches(book, phases, positions, basis, width, tiles):
    """Dequantised residual patch of every stored entry: (M, tiles, k, 16, width)."""
    codes = book["residual_codes"][phases, positions].float() * book["scales"]
    decoded = codes.reshape(-1, 4, RANK) @ basis
    shuffled = F.pixel_shuffle(decoded.reshape(-1, 4, 8, width // 2), 2)
    return shuffled.reshape(len(phases), tiles, codes.shape[2], 16, width)


def dct_residual(patches, book, basis, width):
    """Project true residual patches onto the stored basis and quantise them like the codebook."""
    scale = book["scales"].reshape(4, RANK)
    planes = F.pixel_unshuffle(patches.reshape(-1, 1, 16, width), 2).flatten(-2)
    codes = ((planes @ basis.T) / scale).round().clamp(-127, 127) * scale
    decoded = codes.reshape(-1, 4, RANK) @ basis
    shuffled = F.pixel_shuffle(decoded.reshape(-1, 4, 8, width // 2), 2)
    return shuffled.reshape(*patches.shape)


def dct_residual_float(patches, basis, width):
    """Same projection without the int8 quantisation, to separate truncation from rounding."""
    planes = F.pixel_unshuffle(patches.reshape(-1, 1, 16, width), 2).flatten(-2)
    decoded = (planes @ basis.T).reshape(-1, 4, RANK) @ basis
    shuffled = F.pixel_shuffle(decoded.reshape(-1, 4, 8, width // 2), 2)
    return shuffled.reshape(*patches.shape)


def gather(patches, index):
    shape = (-1, -1, 1, 16, patches.shape[-1])
    return patches.gather(2, index[:, :, None, None, None].expand(*shape)).squeeze(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    provenance = fingerprints(args.db, args.scene_plan)
    book = load_book(args.checkpoint)
    if book["config"]["provenance"] != provenance:
        raise ValueError("compact codebook does not match the selected database")
    width = book["config"].get("patch_width", DEFAULT_WIDTH)
    tiles = tiles_for(width)
    basis = dct_basis(width)
    gate = book["gate"]

    truth, current, frame_phases = observations(args.db, AUDIT)
    samples = current.flatten(0, 1)
    positions = torch.arange(30).repeat(len(frame_phases))
    phases = frame_phases.repeat_interleave(30)
    base = bilinear(samples, phases)
    residual = (truth.float() - base.float()).squeeze(1)

    predictions = {"bilinear": base}
    totals = {"amplitude": residual.abs().sum().item(), "amplitude_count": residual.numel()}
    for start in range(0, len(samples), CHUNK):
        stop = min(start + CHUNK, len(samples))
        chunk_phases, chunk_positions = phases[start:stop], positions[start:stop]
        chunk_samples, chunk_base = samples[start:stop], base[start:stop]
        true_tiles = split_strips(residual[start:stop], width, tiles)
        patches = entry_patches(book, chunk_phases, chunk_positions, basis, width, tiles)
        descriptors = book["descriptor_keys"][chunk_phases, chunk_positions].float()
        query = describe(chunk_samples, width, tiles)
        distance = (descriptors - query[:, :, None]).square().mean(dim=-1).sqrt()
        actual_index = distance.argmin(dim=-1)
        oracle_index = (patches - true_tiles[:, :, None]).abs().mean(dim=(-1, -2)).argmin(dim=-1)
        actual_patches = gather(patches, actual_index)
        oracle_patches = gather(patches, oracle_index)
        actual_distance = distance.gather(2, actual_index[:, :, None]).squeeze(2)
        oracle_distance = distance.gather(2, oracle_index[:, :, None]).squeeze(2)
        dct_patches = dct_residual(true_tiles, book, basis, width)
        float_patches = dct_residual_float(true_tiles, basis, width)
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
            output = reconstruct(chunk_base, value, dist, chunk_phases, limit, strength)
            predictions.setdefault(name, []).append(output)

    predictions = {name: (value if isinstance(value, torch.Tensor) else torch.cat(value))
                   for name, value in predictions.items()}
    for name, value in predictions.items():
        assert_samples(value, frame_phases, current)

    roi = json.loads(ROI.read_text(encoding="utf-8"))["regions"]
    reference = display(truth)
    methods, errors = {}, {}
    args.out.mkdir(parents=True, exist_ok=True)
    export(args.out, "target", reference, AUDIT[0])
    for name, pred in predictions.items():
        result_rgb = display(pred)
        methods[name], errors[name], _ = scores(pred, truth, result_rgb, reference, roi)
        methods[name].update(compare(pred, truth, roi))
        export(args.out, name, result_rgb, AUDIT[0])
        print(f"{name}: RAW MAE {methods[name]['raw']['mae_8bit']:.4f}", flush=True)
    residual_error = {"amplitude": totals["amplitude"] / totals["amplitude_count"]}
    for name in ("actual", "oracle", "dct", "dct_float"):
        residual_error[f"{name}_residual_mae"] = totals[f"{name}_sum"] / totals[f"{name}_count"]
    residual_error["energy_kept_ratio"] = totals["energy_kept"] / totals["energy_full"]
    residual_error["energy_kept_int8_ratio"] = totals["energy_kept_int8"] / totals["energy_full"]
    report = {"start_frame": AUDIT[0], "last_evaluated_frame": AUDIT[1], "samples": len(truth),
              "provenance": provenance, "patch_width": width,
              "entries_per_tile": book["config"]["entries_per_tile"], "gate": gate,
              "codebook_sha256": digest(args.checkpoint), "methods": methods, "residual": residual_error}
    write_json(args.out / "report.json", report)
    with (args.out / "blocks.jsonl").open("w", encoding="utf-8") as log:
        for i, row in enumerate(manifest(*AUDIT)):
            log.write(json.dumps({**row, **{name: error[i].item() for name, error in errors.items()}}) + "\n")
    print(json.dumps(residual_error, indent=1), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        run_logged("evaluation", main)
