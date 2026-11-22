"""v11 量測（第二版）：對齊後的多幀相位累積精度。

比較三種情況在「被量測到的高解析像素」上的誤差：
  1. single      : 只用當前幀（單幀相位覆蓋 1/16）
  2. naive_accum : 不對齊，直接把 T 幀按相位放回高解析網格
  3. aligned_accum: 先用整數偶數位移對齊內容，再按相位放回

對齊位移以「oracle」方式從 raw_bayer_target 求得（上限參考），非模型實測。
垂直位移受條帶高度限制（16 列），另外量測剩餘誤差。

輸出 JSON 到 Model/v11/runs-align-measure-tos.json（UTF-8, no BOM）。
"""
import json
import os
import sqlite3
import sys

import numpy as np

DB = r"C:\Users\dinos\BrainHole\training-data-preparation\拆好的訓練資料\tears-of-steel-q4-shared-v4-raw8.sqlite"
BLOCKS = [0, 15, 29]
WIDTH, HEIGHT = 640, 16
LR_W, LR_H = 160, 4
DY_RANGE = (-4, -2, 0, 2, 4)
DX_RANGE = tuple(range(-12, 13, 2))


def phase_offsets(phase):
    p = phase - 1
    return (p % 4) * 2, (p // 4) * 2


def load_block(conn, bi):
    return conn.execute(
        "SELECT frame_index, source_frame_index, phase_id, raw_bayer_target, decoded "
        "FROM samples WHERE block_index=? ORDER BY frame_index", (bi,)).fetchall()


def best_even_shift(a, b):
    """回傳 (dy, dx, mae)：使 a[y,x] ≈ b[y+dy, x+dx] 的最佳偶數位移與誤差。"""
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


def place(dec, ph, hr, shift_y, shift_x):
    """把某幀的 decoded 依其相位放到高解析網格 hr（NaN=未量測），shift 為內容位移。"""
    px, py = phase_offsets(ph)
    d = np.frombuffer(dec, dtype=np.uint8).reshape(LR_H, LR_W)
    for j in range(2):
        for i in range(80):
            for dy in range(2):
                for dx in range(2):
                    y = 8 * j + py + dy - shift_y
                    x = 8 * i + px + dx - shift_x
                    if 0 <= y < HEIGHT and 0 <= x < WIDTH:
                        hr[y, x] = d[2 * j + dy, 2 * i + dx]


def evaluate(conn):
    rows_all = {}
    for bi in BLOCKS:
        rows = load_block(conn, bi)
        rows_all[bi] = {
            "frame_index": [r[0] for r in rows],
            "source_frame_index": [r[1] for r in rows],
            "phase": [r[2] for r in rows],
            "target": [np.frombuffer(r[3], dtype=np.uint8).reshape(HEIGHT, WIDTH) for r in rows],
            "decoded": [r[4] for r in rows],
        }
    starts = [0, 5000, 12000, 18000]
    res = {}
    for T in (1, 2, 4, 8, 16):
        naive, aligned, single, cov = [], [], [], []
        naive_dup, aligned_dup, cov_dup = [], [], []
        shift_used = []
        for bi, d in rows_all.items():
            n = len(d["frame_index"])
            for s in starts:
                if s + T > n:
                    continue
                tgt0 = d["target"][s].astype(np.float32)
                hr_n = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
                hr_a = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
                hr_s = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
                same_content = all(d["source_frame_index"][s + k] == d["source_frame_index"][s] for k in range(T))
                for k in range(T):
                    ph = d["phase"][s + k]
                    dec = d["decoded"][s + k]
                    place(dec, ph, hr_n, 0, 0)
                    place(dec, ph, hr_s, 0, 0)
                    sy, sx = 0, 0
                    if k > 0:
                        sy, sx, _ = best_even_shift(tgt0, d["target"][s + k].astype(np.float32))
                        shift_used.append((sy, sx))
                    place(dec, ph, hr_a, sy, sx)
                for hr, sink in ((hr_n, naive), (hr_a, aligned), (hr_s, single)):
                    m = ~np.isnan(hr)
                    if not m.any():
                        continue
                    err = float(np.abs(hr[m] - tgt0[m]).mean())
                    sink.append(err)
                    cov.append(float(m.mean()))
                    if same_content:
                        if sink is naive:
                            naive_dup.append(err)
                        elif sink is aligned:
                            aligned_dup.append(err)
                        else:
                            cov_dup.append(err)
        key = f"T={T}"
        res[key] = {
            "single_frame_mae": float(np.mean(single)) if single else None,
            "naive_accum_mae": float(np.mean(naive)) if naive else None,
            "aligned_accum_mae": float(np.mean(aligned)) if aligned else None,
            "coverage_ratio": float(np.mean(cov)) if cov else None,
            "single_frame_mae_same_content": float(np.mean(cov_dup)) if cov_dup else None,
            "naive_accum_mae_same_content": float(np.mean(naive_dup)) if naive_dup else None,
            "aligned_accum_mae_same_content": float(np.mean(aligned_dup)) if aligned_dup else None,
            "n_windows": len(single),
        }
    hist = {}
    for sy, sx in shift_used:
        hist[f"dy={sy},dx={sx}"] = hist.get(f"dy={sy},dx={sx}", 0) + 1
    res["oracle_shift_histogram"] = dict(sorted(hist.items(), key=lambda kv: -kv[1])[:12])
    res["oracle_shift_note"] = "以 raw_bayer_target 求得的整數偶數位移（oracle 上限），dy 限 ±4（條帶僅 16 列）"
    return res


def main():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    out = {"db": os.path.basename(DB), "results": evaluate(conn)}
    conn.close()
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs-align-measure-tos.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
