"""Build v13 S2-train / S2-dev caches for demo.mp4 (counterpart to check_v13.py for BBB)."""
import argparse
from pathlib import Path

import torch

from record_experiment import digest, run_logged, write_json
from v13.data import atomic_save, fingerprints
from v13.demo_data import DEMO_SPLITS, make_demo_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=DEMO_SPLITS, nargs="+", default=["S2-train", "S2-dev"])
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    provenance = fingerprints(args.db, args.scene_plan)
    counts = {}
    for name in args.split:
        cache = make_demo_split(args.db, args.scene_plan, name, provenance)
        path = args.out / f"{name}.pt"
        atomic_save(cache, path)
        write_json(args.out / f"{name}.json", {k: v for k, v in cache.items() if k != "features"})
        counts[name] = {"samples": len(cache["manifest"]), "sha256": digest(path)}
    write_json(args.out / "report.json", {"counts": counts, "provenance": provenance})
    print(counts, flush=True)


if __name__ == "__main__":
    run_logged("checks", main)
