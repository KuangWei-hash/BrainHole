"""Measure decoded compression and historical changes on matched CFA samples."""
import argparse
from contextlib import closing
import json
from pathlib import Path

import torch

from record_experiment import digest, run_logged, write_json
from v11.data import connect
from v11.phase import scatter_indices
from v13.data import labels, load_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("db", "scene-plan", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    scenes = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    if scenes["database_sha256"] != digest(args.db) or any(721 < cut <= 779 for cut in scenes["scene_starts"]):
        raise ValueError("unexpected source database or scene cut")
    roi = Path(__file__).with_name("v13") / "roi-bbb-q5.json"
    regions = {"full": [0, 0, 640, 480],
               **json.loads(roi.read_text(encoding="utf-8"))["regions"]}
    manifest = [{"frame": frame, "block": block} for frame in range(721, 780) for block in range(30)]
    raw = labels(args.db, manifest).reshape(59, 480, 640).short()
    with closing(connect(args.db)) as conn:
        inputs = load_inputs(conn, 721, 779)
    decoded = torch.stack([inputs[row["frame"], row["block"]][2] for row in manifest]).reshape(59, 120, 160).short()
    phases = []
    for frame in range(721, 780):
        frame_phases = {inputs[frame, block][1] for block in range(30)}
        if len(frame_phases) != 1:
            raise ValueError("expected one phase per frame")
        phases.append(frame_phases.pop())
    grids = [scatter_indices(phase) for phase in range(16)]
    totals = {group: {name: torch.zeros(16, 4, dtype=torch.int64) for name in regions}
              for group in ("736-749", "750-779")}
    for frame in range(736, 780):
        group = "736-749" if frame < 750 else "750-779"
        for age in range(16):
            history = frame - age - 721
            strip_rows, cols = grids[phases[history]]
            rows = (torch.arange(30)[:, None] * 16 + strip_rows).flatten()
            old, current = raw[history][rows[:, None], cols], raw[frame - 721][rows[:, None], cols]
            measured = decoded[history]
            errors = torch.stack(((measured - old).abs(), (old - current).abs(), (measured - current).abs()))
            for name, (x0, y0, x1, y1) in regions.items():
                crop = errors[:, (rows >= y0) & (rows < y1)][:, :, (cols >= x0) & (cols < x1)]
                totals[group][name][age, :3] += crop.sum(dim=(1, 2))
                totals[group][name][age, 3] += crop[0].numel()
    report = {}
    for group, areas in totals.items():
        report[group] = {}
        for area, values in areas.items():
            report[group][area] = [
                {"age": age, "pixels": n, "compression_mae": c / n,
                 "history_change_mae": h / n, "combined_mae": b / n}
                for age, (c, h, b, n) in enumerate(values.tolist())]
        for age in (0, 7, 15):
            print(group, "full", report[group]["full"][age], flush=True)
    write_json(args.out / "report.json", {"source": [721, 779], "scored": [736, 779],
               "strips": 1320, "roi_sha256": digest(roi), "groups": report})


if __name__ == "__main__":
    run_logged("checks", main)
