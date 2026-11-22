"""v11 量測（第三版）：擴大水平搜尋範圍並統計飽和率。

目的：第二版的 oracle 位移搜尋範圍（dy ±4、dx ±12）可能飽和，讓「對齊後」數字
偏保守。本版把 dx 擴到 ±24，並統計最佳位移落在搜尋邊界的比例（飽和率），
以區分「對齊方法不足」與「搜尋範圍不足」。

輸出 Model/v11/runs-align-measure-wide.json（UTF-8, no BOM）。
"""
import json
import os
import sqlite3
import sys

import numpy as np

DB = r"C:\Users\dinos\BrainHole\training-data-preparation\拆好的訓練資料\big-buck-bunny-30s-q4-shared-v4-raw8-fixed.sqlite"
BLOCKS = [0, 7, 15, 22, 29]
WIDTH, HEIGHT = 640, 16
LR_W, LR_H = 160, 4
DY_RANGE = (-4, -2, 0, 2, 4)
DX_RANGE = tuple(range(-24, 25, 2))


def phase_offsets(phase):
    p = phase - 1
    return (p % 4) * 2, (p // 4) * 2


def load_block(conn, bi):
    return conn.execute(
        "SELECT frame_index, source_frame_index, phase_id, raw_bayer_target, decoded "
        "FROM samples WHERE block_index=? ORDER BY frame_index", (bi,)).fetchall()


def best_even_shift(a, b):
    H, W = a.shape
    best = (0, 0, float("inf"))
    for dy in DY_RANGE:
        for dx in DX_RANGE:
            y0 = 4 + max(0, -dy)
            y1 = H - 4 - max(0, dy)
            x0 = 16 + max(0, -dx)
            x1 = W - 16 - max(0, dx)
            if y1 - y0 < 4 or x1 - x0 < 32:
                continue
            ac = a[y0:y1, x0:x1].astype(np.float32)
            bc = b[y0 + dy:y1 + dy, x0 + dx:x1 + dx].astype(np.float32)
            mae = float(np.abs(ac - bc).mean())
            if mae < best[2]:
                best = (dy, dx, mae)
    return best


def place(dec, ph, hr, sy, sx):
    px, py = phase_offsets(ph)
    d = np.frombuffer(dec, dtype=np.uint8).reshape(LR_H, LR_W)
    for j in range(2):
        for i in range(80):
            for dy in range(2):
                for dx in range(2):
                    y = 8 * j + py + dy - sy
                    x = 8 * i + px + dx - sx
                    if 0 <= y < HEIGHT and 0 <= x < WIDTH:
                        hr[y, x] = d[2 * j + dy, 2 * i + dx]


def main():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows_all = {}
    for bi in BLOCKS:
        rows = load_block(conn, bi)
        rows_all[bi] = {
            "phase": [r[2] for r in rows],
            "src": [r[1] for r in rows],
            "target": [np.frombuffer(r[3], dtype=np.uint8).reshape(HEIGHT, WIDTH) for r in rows],
            "decoded": [r[4] for r in rows],
        }
    starts = [0, 37, 120, 300, 600]
    out = {}
    for T in (2, 4, 8, 16):
        naive, aligned, single = [], [], []
        sat = 0
        total_shift = 0
        for bi, d in rows_all.items():
            n = len(d["phase"])
            for s in starts:
                if s + T > n:
                    continue
                tgt0 = d["target"][s].astype(np.float32)
                hr_n = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
                hr_a = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
                hr_s = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
                for k in range(T):
                    ph, dec = d["phase"][s + k], d["decoded"][s + k]
                    place(dec, ph, hr_n, 0, 0)
                    place(dec, ph, hr_s, 0, 0)
                    sy, sx = 0, 0
                    if k > 0:
                        sy, sx, _ = best_even_shift(tgt0, d["target"][s + k].astype(np.float32))
                        total_shift += 1
                        if sx in (DX_RANGE[0], DX_RANGE[-1]) or sy in (DY_RANGE[0], DY_RANGE[-1]):
                            sat += 1
                    place(dec, ph, hr_a, sy, sx)
                for hr, sink in ((hr_n, naive), (hr_a, aligned), (hr_s, single)):
                    m = ~np.isnan(hr)
                    if m.any():
                        sink.append(float(np.abs(hr[m] - tgt0[m]).mean()))
        out[f"T={T}"] = {
            "single_frame_mae": float(np.mean(single)) if single else None,
            "naive_accum_mae": float(np.mean(naive)) if naive else None,
            "aligned_accum_mae": float(np.mean(aligned)) if aligned else None,
            "shift_search_saturation_rate": sat / total_shift if total_shift else None,
            "n_shift_estimates": total_shift,
            "n_windows": len(single),
        }
    out["search_range"] = {"dy": list(DY_RANGE), "dx": list(DX_RANGE)}
    out["note"] = ("dy 受條帶僅 16 列限制（±4 已用掉一半重疊視窗）；dx 擴到 ±24。"
                   "飽和率 = 最佳位移落在搜尋邊界的比例。")
    conn.close()
    payload = json.dumps(out, ensure_ascii=False, indent=2)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs-align-measure-wide.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(payload)
    print(payload)


if __name__ == "__main__":
    sys.exit(main())
