"""Isolated offline windows and byte caches for the BBB experiment."""
from contextlib import closing
from pathlib import Path
import json
import uuid

import torch

from record_experiment import atomic_replace, digest
from v11.data import BLOCKS, build_windows, connect
from v13.base import features


SPLITS = {
    "S1-train": (691, 719, 420), "S1-dev": (721, 750, 450),
    "S2-train": (0, 719, 19800), "S2-dev": (721, 779, 1320), "S2-audit": (780, 899, 3150),
}


def atomic_save(value, path):
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        atomic_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprints(db, scene_plan):
    model = Path(__file__).resolve().parents[1]
    return {"db": digest(db), "scene": digest(scene_plan),
            **{name: digest(model / name) for name in
               ("v13/base.py", "v13/data.py", "v12/__init__.py", "v11/phase.py")}}


def load_inputs(conn, first, last):
    rows = conn.execute(
        "SELECT frame_index, block_index, header_id, phase_id, decoded FROM samples "
        "WHERE frame_index BETWEEN ? AND ? ORDER BY frame_index, block_index", (first, last))
    result = {}
    for frame, block, header, phase, decoded in rows:
        if (frame, block) in result or len(decoded) != 640 or not 1 <= phase <= 16:
            raise ValueError(f"invalid decoded strip at {(frame, block)}")
        result[frame, block] = header, phase - 1, torch.frombuffer(bytearray(decoded), dtype=torch.uint8).reshape(1, 4, 160)
    return result


def make_split(db, scene_plan, name, provenance):
    first, last, expected = SPLITS[name]
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


def labels(db, manifest):
    first, last = manifest[0]["frame"], manifest[-1]["frame"]
    with closing(connect(db)) as conn:
        rows = conn.execute(
            "SELECT frame_index, block_index, raw_bayer_target FROM samples WHERE frame_index BETWEEN ? AND ?",
            (first, last))
        targets = {(frame, block): raw for frame, block, raw in rows}
    raw = bytearray()
    for row in manifest:
        target = targets[row["frame"], row["block"]]
        if len(target) != 10240:
            raise ValueError(f"invalid target size at {row}")
        raw.extend(target)
    return torch.frombuffer(raw, dtype=torch.uint8).reshape(-1, 1, 16, 640)


def raw_frame(strips):
    return strips.reshape(1, 1, BLOCKS * 16, 640)
