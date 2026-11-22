"""End-to-end consistency check for a schema v4 dataset.

Recomputes the low-resolution RGGB input from the stored full-resolution
RAW8 target and compares it with the stored decoded mosaic. The two must
not be identical (JPEG XS is lossy at Q=4) but must stay close, which proves
the decoded column belongs to the same sample as the target.
"""
import json
import sqlite3
import sys
from pathlib import Path

WIDTH, BLOCK_HEIGHT, INPUT_WIDTH, INPUT_HEIGHT = 640, 16, 160, 4


def downsample(raw, phase):
    out = bytearray(INPUT_WIDTH * INPUT_HEIGHT)
    px, py = ((phase - 1) % 4) * 2, ((phase - 1) // 4) * 2
    for j in range(2):
        for i in range(80):
            for dy in range(2):
                for dx in range(2):
                    out[(2 * j + dy) * INPUT_WIDTH + 2 * i + dx] = raw[(8 * j + py + dy) * WIDTH + 8 * i + px + dx]
    return out


def main():
    database = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    db = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    total = db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    step = max(1, total // limit)
    rows = db.execute("SELECT frame_index,block_index,phase_id,raw_bayer_target,decoded FROM samples "
                       "ORDER BY frame_index,block_index").fetchall()[::step]
    diffs = []
    for _, _, phase, target, decoded in rows:
        reference = downsample(target, phase)
        diffs.extend(abs(a - b) for a, b in zip(reference, decoded))
    diffs.sort()
    report = {
        "database": str(database),
        "samples_in_database": total,
        "sampled_samples": len(rows),
        "compared_values": len(diffs),
        "mean_absolute_error": sum(diffs) / len(diffs),
        "median_absolute_error": diffs[len(diffs) // 2],
        "p99_absolute_error": diffs[int(len(diffs) * 0.99)],
        "max_absolute_error": diffs[-1],
        "identical_fraction": sum(1 for d in diffs if d == 0) / len(diffs),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
