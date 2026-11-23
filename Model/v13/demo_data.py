"""Isolated offline windows for v13 on the real demo.mp4 recording.

Mirrors v13/data.py's make_split, but against demo.mp4's own frame ranges
instead of the frozen BBB windows in SPLITS. Reuses v13.data.fingerprints
unchanged (it only hashes the db, scene plan and the four BBB-agnostic
source files), so caches built here stay compatible with the unmodified
train_v13.py / evaluate_v13.py loaders.
"""
from contextlib import closing
import json

import torch

from v11.data import build_windows, connect
from v13.base import features
from v13.data import load_inputs


DEMO_SPLITS = {
    "S2-train": (0, 1389, 41250),
    "S2-dev": (1748, 1852, 2700),
}


def make_demo_split(db, scene_plan, name, provenance):
    first, last, expected = DEMO_SPLITS[name]
    plan = json.loads(scene_plan.read_text(encoding="utf-8"))
    if plan["database_sha256"] != provenance["db"]:
        raise ValueError("scene plan belongs to a different database")
    cuts = sorted(set(plan["scene_starts"]))
    windows = sorted(build_windows(db, 16, first, last, scene_starts=cuts), key=lambda item: (item[1][-1], item[0]))
    if len(windows) != expected:
        raise ValueError(f"{name}: expected {expected} windows, found {len(windows)}")
    data = torch.empty((expected, 4, 16, 640), dtype=torch.uint8)
    with closing(connect(db)) as conn:
        inputs = load_inputs(conn, first, last)
    manifest = []
    for i, (block, oldest_first) in enumerate(windows):
        frame = oldest_first[-1]
        records = [inputs[t, block] for t in reversed(oldest_first)]
        header, phases, decoded = zip(*records)
        if len(set(header)) != 1 or len(set(phases)) != 16:
            raise ValueError(f"{name}: unsupported training window at {(frame, block)}")
        data[i] = features(torch.stack(decoded), phases)
        manifest.append({"frame": frame, "block": block, "first": oldest_first[0], "header": header[0]})
    return {"split": name, "source": [first, last], "provenance": provenance,
            "manifest": manifest, "features": data}
