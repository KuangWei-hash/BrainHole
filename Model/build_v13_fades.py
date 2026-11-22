"""Prepare training-only synthetic fades from existing phase-matched BBB data."""
import argparse
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import torch

from record_experiment import digest, run_logged, write_json
from v11.data import connect
from v11.phase import scatter_indices
from v13.base import features
from v13.data import atomic_save, fingerprints, labels, load_inputs


def build(db, provenance):
    with closing(connect(db)) as conn:
        inputs = load_inputs(conn, 0, 719)
    rows = [{"frame": frame, "block": block} for frame in range(720) for block in range(30)]
    targets = labels(db, rows).reshape(720, 30, 1, 16, 640)
    keys = [(frame, block, scene) for start, stop, scene in ((0, 355, 0), (356, 471, 1),
                                                               (472, 690, 2), (691, 719, 3))
            for frame in range(start + 15, stop + 1) for block in range(30)]
    groups = defaultdict(list)
    for frame, block, scene in keys:
        groups[(inputs[frame, block][1], block)].append((frame, scene))
    order = torch.Generator().manual_seed(42)
    replacement = torch.randperm(len(keys), generator=order)[:len(keys) // 5]
    x = torch.empty((len(replacement), 4, 16, 640), dtype=torch.uint8)
    y = torch.empty((len(replacement), 1, 16, 640), dtype=torch.uint8)
    coordinate = torch.arange(640)
    grids = [scatter_indices(phase) for phase in range(16)]
    manifest = []
    for i, index in enumerate(replacement.tolist()):
        frame, block, scene = keys[index]
        phase = inputs[frame, block][1]
        partners = [other for other, other_scene in groups[phase, block] if other_scene != scene]
        other = partners[int(torch.randint(len(partners), (), generator=order))]
        params = torch.rand(4, generator=order).tolist()
        center = 80 + params[0] * 480
        width = 90 + params[1] * 150
        local = ((width - (coordinate - center).abs()) / 24).clamp(0, 1)
        level = 0.2 + params[2] * 0.6
        slope = (0.02 + params[3] * 0.03) * (1 if i % 2 else -1)
        phases, mixed = [], []
        for age in range(16):
            _, p, a = inputs[frame - age, block]
            _, q, b = inputs[other - age, block]
            if p != q:
                raise ValueError("phase mismatch in fade parents")
            _, columns = grids[p]
            weight = local[columns] * max(0., min(1., level - age * slope))
            mixed.append(torch.lerp(a.float(), b.float(), weight).round().byte())
            phases.append(p)
        y[i] = torch.lerp(targets[frame, block].float(), targets[other, block].float(), level * local).round().byte()
        x[i] = features(torch.stack(mixed), tuple(phases))
        manifest.append({"frame": frame, "other_frame": other, "block": block,
                         "center": center, "width": width, "level": level, "slope": slope})
    return {"provenance": provenance, "indices": replacement, "features": x, "targets": y,
            "manifest": manifest, "train_count": len(keys)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("db", "scene-plan", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    torch.set_num_threads(3)
    provenance = fingerprints(args.db, args.scene_plan)
    result = build(args.db, provenance)
    path = args.out / "fades.pt"
    atomic_save(result, path)
    write_json(args.out / "manifest.json", {"sources": [0, 719], "seed": 42,
               "source_split_size": result["train_count"], "samples": len(result["manifest"]),
               "fade_cache_sha256": digest(path), "manifest": result["manifest"]})
    print(f"Built {len(result['manifest'])} synthetic training strips", flush=True)


if __name__ == "__main__":
    run_logged("checks", main)
