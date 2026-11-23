"""Quantify where the compact codebook's RAW MAE gain actually comes from:
low-frequency correction vs genuine high-frequency texture recovery."""
from pathlib import Path

import torch

from evaluate_v13 import ROI, display, scores
from evaluate_v14_current_phase import phase_ids, sample
from evaluate_v14_strips import manifest
import v14 as compact
from v13.data import labels

DB = Path("../training-data-preparation/拆好的訓練資料/demo-q5-native.sqlite")
TEST_RANGE = (1748, 1852)
BOOK = Path("runs/v14-demo-compact-codebook-w16-rank64/best-mae.pt")


def main():
    torch.set_num_threads(4)
    phases = torch.tensor(phase_ids(DB, *TEST_RANGE))
    raw = labels(DB, manifest(*TEST_RANGE)).reshape(-1, 30, 16, 640)
    samples = torch.stack([sample(frame, int(p)) for frame, p in zip(raw, phases)])
    flat_samples = samples.flatten(0, 1)
    flat_phases = phases.repeat_interleave(30)
    positions = torch.arange(30).repeat(len(phases))
    truth = raw.reshape(-1, 1, 16, 640)

    book = torch.load(BOOK, map_location="cpu", weights_only=True)
    compact.RANK = book["config"].get("dct_rank", compact.RANK)
    base = compact.bilinear(flat_samples, flat_phases)
    width = book["config"].get("patch_width", 40)
    basis = compact.dct_basis(width)
    patches, distance = compact.retrieve(book, flat_samples, positions, flat_phases, basis)
    pred = compact.reconstruct(base, patches, distance, flat_phases, **book["gate"])
    limit, strength = book["gate"]["limit"], book["gate"]["strength"]
    weight = (1 - distance / limit).clamp_min(0) * strength
    coverage = (weight > 0).float().mean().item()
    print(f"gate: {book['gate']}; tile coverage (weight>0): {coverage*100:.2f}%", flush=True)

    regions = __import__("json").loads(ROI.read_text(encoding="utf-8"))["regions"]
    ref_rgb = display(truth)
    for name, value in (("bilinear", base), ("compact-k08", pred)):
        rgb = display(value)
        result, _, _ = scores(value, truth, rgb, ref_rgb, regions)
        print(f"{name}: raw_mae={result['raw']['mae_8bit']:.4f} "
              f"low_freq_mae={result['low_frequency_mae']:.4f} high_freq_mae={result['high_frequency_mae']:.4f} "
              f"gradient_x_mae={result['gradient_x_mae']:.4f} gradient_y_mae={result['gradient_y_mae']:.4f} "
              f"exact_raw_fraction={result['exact_raw_fraction']:.4f}", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
