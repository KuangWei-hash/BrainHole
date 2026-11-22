"""v11 量測（第六版）：移動區域的運動方向分解。

目的：v11 需要逐像素運動補償，但條帶只有 16 列高，縱向可估計的自由度極少。
本量測對每個「移動像素」求出使局部 SSD 最小的整數偶數位移 (dy, dx)，
統計 dy 與 dx 的分布，回答：運動主要在橫向還是縱向？

位移約定：b(y+dy, x+dx) ≈ a(y, x)，即內容自 a 往 b 移動了 (dy, dx) 高解析像素。

輸出 Model/v11/runs-motion-direction.json（UTF-8, no BOM）。
"""
import json
import os
import sqlite3
import sys

import numpy as np

DB = r"C:\Users\dinos\BrainHole\training-data-preparation\拆好的訓練資料\big-buck-bunny-30s-q4-shared-v4-raw8-fixed.sqlite"
WIDTH, HEIGHT = 640, 16
BLOCKS = [0, 7, 15, 22, 29]
MAX_PAIRS_PER_BLOCK = 200
SHIFT_RANGE = 8
MOVING_THRESHOLD = 6.0
PATCH_H, PATCH_W = 3, 7


def box_filter(image, h, w):
    """Sum over a (2h+1, 2w+1) window, edge-replicated."""
    pad = np.pad(image, ((h, h), (w, w)), mode="edge")
    cs = pad.cumsum(0).cumsum(1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    out = (cs[2*h+1:, 2*w+1:] - cs[:-2*h-1, 2*w+1:]
           - cs[2*h+1:, :-2*w-1] + cs[:-2*h-1, :-2*w-1])
    return out


def local_ssd(a, b, dy, dx):
    """Per-pixel SSD between a and b shifted by (dy, dx).

    Only pixels whose whole patch lies inside the valid overlap are scored; the
    out-of-range region must not be counted as zero error, otherwise large shifts
    look artificially good.
    """
    H, W = a.shape
    ys0, ys1 = max(0, -dy), H - max(0, dy)
    xs0, xs1 = max(0, -dx), W - max(0, dx)
    diff = np.zeros_like(a)
    valid = np.zeros_like(a)
    diff[ys0:ys1, xs0:xs1] = (a[ys0:ys1, xs0:xs1] - b[ys0+dy:ys1+dy, xs0+dx:xs1+dx]) ** 2
    valid[ys0:ys1, xs0:xs1] = 1.0
    num = box_filter(diff * valid, PATCH_H//2, PATCH_W//2)
    cnt = box_filter(valid, PATCH_H//2, PATCH_W//2)
    ssd = num / np.maximum(cnt, 1.0)
    ssd[cnt < PATCH_H * PATCH_W] = np.inf
    return ssd


def best_shift_map(a, b):
    """Per-pixel best (dy, dx) minimizing local SSD."""
    best_ssd = None
    best_dy = np.zeros_like(a, dtype=np.int16)
    best_dx = np.zeros_like(a, dtype=np.int16)
    for dy in range(-SHIFT_RANGE, SHIFT_RANGE+1, 2):
        for dx in range(-SHIFT_RANGE, SHIFT_RANGE+1, 2):
            ssd = local_ssd(a, b, dy, dx)
            if best_ssd is None:
                best_ssd, best_dy, best_dx = ssd, np.full_like(a, dy), np.full_like(a, dx)
            else:
                better = ssd < best_ssd
                best_ssd = np.where(better, ssd, best_ssd)
                best_dy = np.where(better, dy, best_dy)
                best_dx = np.where(better, dx, best_dx)
    return best_dy, best_dx


def main():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    dy_hist, dx_hist = {}, {}
    total_moving = 0
    total_pixels = 0
    pairs_used = 0
    per_block = {}
    threshold_counts = {4.0: 0, 6.0: 0, 10.0: 0}
    for bi in BLOCKS:
        rows = conn.execute(
            "SELECT frame_index, raw_bayer_target FROM samples WHERE block_index=? ORDER BY frame_index",
            (bi,)).fetchall()
        targets = [np.frombuffer(r[1], dtype=np.uint8).reshape(HEIGHT, WIDTH).astype(np.float32) for r in rows]
        block_moving = 0
        block_total = 0
        for i in range(1, min(len(rows), MAX_PAIRS_PER_BLOCK + 1)):
            a, b = targets[i-1], targets[i]
            diff = np.abs(a - b)
            for threshold in threshold_counts:
                threshold_counts[threshold] += int((diff > threshold).sum())
            moving = diff > MOVING_THRESHOLD
            if not moving.any():
                continue
            best_dy, best_dx = best_shift_map(a, b)
            sel_dy = best_dy[moving]
            sel_dx = best_dx[moving]
            for value, counter in ((sel_dy, dy_hist), (sel_dx, dx_hist)):
                vals, counts = np.unique(value, return_counts=True)
                for v, c in zip(vals.tolist(), counts.tolist()):
                    counter[str(int(v))] = counter.get(str(int(v)), 0) + c
            block_moving += int(moving.sum())
            block_total += moving.size
            total_moving += int(moving.sum())
            total_pixels += moving.size
            pairs_used += 1
        per_block[str(bi)] = {"moving_pixels": block_moving, "pixels": block_total}
    conn.close()

    def frac(hist, pred):
        total = sum(hist.values())
        return sum(c for k, c in hist.items() if pred(int(k))) / total if total else None

    out = {
        "db": os.path.basename(DB),
        "pairs_used": pairs_used,
        "blocks": BLOCKS,
        "shift_range_hr_px": SHIFT_RANGE,
        "shift_step_hr_px": 2,
        "moving_threshold_hr_mae": MOVING_THRESHOLD,
        "moving_pixel_share": total_moving / total_pixels if total_pixels else None,
        "moving_pixel_share_by_threshold": {str(k): v / total_pixels if total_pixels else None
                                            for k, v in threshold_counts.items()},
        "dy_histogram": dict(sorted(dy_hist.items(), key=lambda kv: int(kv[0]))),
        "dx_histogram": dict(sorted(dx_hist.items(), key=lambda kv: int(kv[0]))),
        "fraction": {
            "abs_dy_le_2": frac(dy_hist, lambda v: abs(v) <= 2),
            "abs_dy_le_4": frac(dy_hist, lambda v: abs(v) <= 4),
            "abs_dx_le_2": frac(dx_hist, lambda v: abs(v) <= 2),
            "abs_dx_le_4": frac(dx_hist, lambda v: abs(v) <= 4),
            "dy_zero": frac(dy_hist, lambda v: v == 0),
            "dx_zero": frac(dx_hist, lambda v: v == 0),
        },
        "per_block": per_block,
        "note": "位移為整數偶數（保持 Bayer 色樣），單位為高解析像素；內容自前幀往後幀位移。",
    }
    payload = json.dumps(out, ensure_ascii=False, indent=2)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs-motion-direction.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(payload)
    print(payload)


if __name__ == "__main__":
    sys.exit(main())
