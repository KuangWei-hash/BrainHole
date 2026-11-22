"""Build isolated feature caches and check the v13 S0 contracts."""
import argparse
from contextlib import closing
from pathlib import Path
import sqlite3

import torch
from torch.nn import functional as F

from record_experiment import digest, run_logged, write_json
from v11.data import connect
from v11.phase import scatter_indices
from v12 import nearest_bayer_bytes, phase_protected_blend
from v13.base import features
from v13.data import SPLITS, atomic_save, fingerprints, load_inputs, make_split
from v13.model import ResidualCNN


def check(db, dev, device):
    with closing(connect(db)) as conn:
        def authorizer(action, table, column, *_):
            if action == sqlite3.SQLITE_READ and column == "raw_bayer_target":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        conn.set_authorizer(authorizer)
        decoded = load_inputs(conn, 721, 750)
        for phase_start in range(16):
            # Use all phase orientations, including orientations preceding the first dev key.
            phases = tuple((phase_start - i) % 16 for i in range(16))
            strips = torch.stack([decoded[736 - i, phase_start][2] for i in range(16)])
            feature = features(strips, phases)
            reference = phase_protected_blend(nearest_bayer_bytes(strips, torch.tensor(phases)), strips, phases, 2)
            torch.testing.assert_close(feature[:1], reference, rtol=0, atol=0)
            for age, phase in enumerate(phases):
                rows, cols = scatter_indices(phase)
                grid = (rows[:, None], cols[None, :])
                assert torch.equal(feature[1][grid], strips[age, 0])
                assert torch.all(feature[2][grid] == age)
                assert torch.all(feature[3][grid] == 1)
            assert not torch.equal(feature, features(strips.flip(0), phases[::-1]))
        first_phases = tuple(decoded[t, 0][1] for t in (722, 721))
        first_strips = torch.stack([decoded[t, 0][2] for t in (722, 721)])
        warmup = features(first_strips, first_phases)
        assert torch.equal(warmup[:1], nearest_bayer_bytes(first_strips[:1], torch.tensor(first_phases[:1]))[0])
        assert int(warmup[3].sum()) == 1280
        repeated = features(strips, (0,) * 16)
        assert int(repeated[3].sum()) == 640
        assert torch.count_nonzero(repeated[2][repeated[3] == 1]) == 0
    assert dev["manifest"][0]["first"] == 721 and dev["manifest"][-1]["frame"] == 750
    example = dev["features"][:2].to(device)
    assert torch.equal(F.pixel_shuffle(F.pixel_unshuffle(example[:, :1].float(), 2), 2), example[:, :1])
    for channel in (2, 3):
        cells = F.pixel_unshuffle(example[:, channel:channel + 1].float(), 2)
        assert torch.equal(cells, cells[:, :1].expand_as(cells))
    with torch.inference_mode():
        for variant in ("R0", "R1"):
            model = ResidualCNN(variant).to(device)
            assert torch.equal(model.predict(example), example[:, :1])
            assert torch.equal(model.predict(example, 0), example[:, :1])
            assert all(torch.equal(model.predict(example[i:i + 1]), example[i:i + 1, :1]) for i in range(2))
    return {"byte_exact_phases": 16, "target_column_denied": True, "zero_init_and_alpha0": True,
            "cuda": str(device), "parameters_r1": sum(x.numel() for x in ResidualCNN("R1").parameters())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, nargs="+", default=["S1-train", "S1-dev"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    provenance = fingerprints(args.db, args.scene_plan)
    counts = {}
    for name in args.split:
        cache = make_split(args.db, args.scene_plan, name, provenance)
        path = args.out / f"{name}.pt"
        atomic_save(cache, path)
        write_json(args.out / f"{name}.json", {k: v for k, v in cache.items() if k != "features"})
        counts[name] = {"samples": len(cache["manifest"]), "sha256": digest(path)}
        if name == "S1-dev":
            checks = check(args.db, cache, torch.device(args.device))
            write_json(args.out / "checks.json", checks)
    write_json(args.out / "report.json", {"counts": counts, "provenance": provenance})
    print(counts, flush=True)


if __name__ == "__main__":
    run_logged("checks", main)
