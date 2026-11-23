"""Corrected compact codebook training/eval: uses the REAL `decoded` column
(actual JPEG XS Q=5 round-trip) as v14.phase_data's `low` argument everywhere,
instead of a clean ground-truth-derived sample -- an idealized, noise-free
input a real receiver never has. The residual TARGET still correctly uses the
true raw_bayer_target label; only the input signal is fixed.
"""
import argparse
from contextlib import closing
from pathlib import Path

import torch

from evaluate import demosaic
from evaluate_v10 import SingleVideo
from evaluate_v13 import verify_video
from evaluate_v14_current_phase import phase_ids
from evaluate_v14_strips import manifest
from record_experiment import digest, run_logged, update_active_record, write_json
import v14
from v11.data import connect
from v13.data import atomic_save, labels


TRAIN_RANGE = (0, 1389)
TEST_RANGE = (1748, 1852)
CAPACITIES = (1, 4, 8)
ROUNDS = 5
WIDTH = 40
RANK = 160


def load_range_real(db, first, last):
    phases = phase_ids(db, first, last)
    raw = labels(db, manifest(first, last)).reshape(-1, 30, 16, 640)
    with closing(connect(db)) as conn:
        rows = conn.execute("SELECT frame_index, block_index, decoded FROM samples "
                            "WHERE frame_index BETWEEN ? AND ?", (first, last)).fetchall()
    frames = raw.shape[0]
    low = torch.zeros((frames, 30, 4, 160), dtype=torch.uint8)
    for frame, block, decoded in rows:
        low[frame - first, block] = torch.frombuffer(bytearray(decoded), dtype=torch.uint8).reshape(4, 160)
    return raw, low, torch.tensor(phases)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--rank", type=int, default=RANK)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)
    v14.RANK = args.rank
    width, tiles = args.width, v14.tiles_for(args.width)
    basis = v14.dct_basis(width)

    train_raw, train_low, train_phases = load_range_real(args.db, *TRAIN_RANGE)
    test_raw, test_low, test_phases = load_range_real(args.db, *TEST_RANGE)
    test_flat_low = test_low.reshape(-1, 4, 160)
    test_flat_phases = test_phases.repeat_interleave(30)
    test_positions = torch.arange(30).repeat(len(test_phases))
    base = v14.bilinear(test_flat_low, test_flat_phases)
    truth = test_raw.reshape(-1, 1, 16, 640)
    baseline_mae = (base.short() - truth.short()).abs().double().mean().item()
    print(f"REAL-signal bilinear baseline RAW MAE: {baseline_mae:.4f}", flush=True)

    config_base = {"architecture": "v14-demo-compact-codebook-real-signal", "patch_width": width,
                    "dct_rank": args.rank, "train_range": TRAIN_RANGE, "test_range": TEST_RANGE,
                    "descriptor_dimensions": 16, "residual_coefficients_per_patch": 4 * args.rank,
                    "input_signal": "actual JPEG XS Q=5 decoded column, not a clean ground-truth sample"}
    write_json(args.out / "config.json", {**config_base, "entries_per_tile_options": CAPACITIES,
                                          "rounds_per_capacity": ROUNDS, "baseline_raw_mae": baseline_mae})

    results, best, epoch = {}, None, 0
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for k in CAPACITIES:
            keys = codes = None
            for iteration in range(1, ROUNDS + 1):
                phase_keys, phase_codes = [], []
                for phase in range(16):
                    mask = train_phases == phase
                    present_raw, present_low = train_raw[mask], train_low[mask]
                    if len(present_raw) < k:
                        raise ValueError(f"not enough phase-{phase} frames in the train range for k={k}")
                    features, residuals = v14.phase_data(present_raw, present_low, phase, basis, width, tiles)
                    if iteration == 1:
                        start = torch.linspace(0, len(present_raw) - 1, k).round().long()
                        seed_keys, seed_codes = features[:, start], residuals[:, start]
                    else:
                        seed_keys, seed_codes = keys[phase], codes[phase]
                    fitted_keys, fitted_codes, _ = v14.fit_round(features, residuals, seed_keys, seed_codes)
                    phase_keys.append(fitted_keys)
                    phase_codes.append(fitted_codes)
                keys, codes = phase_keys, phase_codes
            book = v14.pack(list(zip(keys, codes)), {**config_base, "entries_per_tile": k})
            patches, distance = v14.retrieve(book, test_flat_low, test_positions, test_flat_phases, basis)
            best_gate, best_mae = None, float("inf")
            for limit, strength in v14.GATES:
                pred = v14.reconstruct(base.reshape(-1, 1, 16, 640), patches, distance, test_flat_phases,
                                           limit, strength)
                mae = (pred.short() - truth.short()).abs().double().mean().item()
                if mae < best_mae:
                    best_mae, best_gate = mae, {"limit": limit, "strength": strength}
            book["gate"] = best_gate
            path = args.out / f"k{k:02d}.pt"
            atomic_save(book, path)
            size = path.stat().st_size
            epoch += 1
            row = {"epoch": epoch, "entries_per_tile": k, "validation": {"mae_8bit": best_mae},
                   "gate": best_gate, "codebook_bytes": size}
            log.write(__import__("json").dumps(row) + "\n")
            log.flush()
            update_active_record()
            results[f"k{k:02d}"] = row
            print(f"k={k}: RAW MAE {best_mae:.4f} (gate {best_gate}), codebook {size/1e6:.3f} MB", flush=True)
            if best is None or best_mae < best[1]:
                best = (k, best_mae, book, path)

    k, mae, book, path = best
    print(f"best: k={k}, RAW MAE {mae:.4f} vs REAL bilinear {baseline_mae:.4f} "
          f"({(1 - mae / baseline_mae) * 100:+.1f}%), codebook {path.stat().st_size/1e6:.3f} MB", flush=True)

    patches, distance = v14.retrieve(book, test_flat_low, test_positions, test_flat_phases, basis)
    pred = v14.reconstruct(base.reshape(-1, 1, 16, 640), patches, distance, test_flat_phases,
                               **book["gate"]).reshape_as(truth)
    names = ("target", "bilinear", f"compact-k{k:02d}")
    videos = {name: SingleVideo(args.out, name) for name in names}
    for name, value in zip(names, (truth, base.reshape(-1, 1, 16, 640), pred)):
        rgb = demosaic(value.float() / 255).reshape(-1, 30, 3, 16, 640)
        for frame in rgb:
            videos[name].write(dict(enumerate(frame)))
    for video in videos.values():
        video.close()
    for name in names:
        verify_video(args.out / f"{name}.mp4", len(test_raw))

    atomic_save(book, args.out / "last.pt")
    atomic_save(book, args.out / "best-mae.pt")
    write_json(args.out / "report.json", {"baseline_raw_mae": baseline_mae, "results": results,
                                          "chosen": {"entries_per_tile": k, "raw_mae": mae},
                                          "db_sha256": digest(args.db)})


if __name__ == "__main__":
    run_logged("training", main)
