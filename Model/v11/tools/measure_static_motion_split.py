"""v11 量測（第五版）：靜態區域的多幀相位累積是否回到壓縮誤差地板。

分區量測：把「靜態像素」與「運動像素」分開評估累積誤差。
  靜態：|target(f) - target(f+k)| < thr（同一位置比較，代表該內容沒動）
  運動：其餘

若靜態區域的累積誤差接近單幀壓縮誤差（BBB 約 3.5~4.2 MAE），
代表「不同 phase 給出的是同一場景不同像素的直接量測」這個機制成立，
瓶頸就落在局部運動補償（正是 Temporal SR 要解的問題）。

輸出 Model/v11/runs-static-motion-split.json（UTF-8, no BOM）。
"""
import json
import os
import sqlite3
import sys

import numpy as np

DBS = {
    "BBB": r"C:\Users\dinos\BrainHole\training-data-preparation\拆好的訓練資料\big-buck-bunny-30s-q4-shared-v4-raw8-fixed.sqlite",
    "ToS": r"C:\Users\dinos\BrainHole\training-data-preparation\拆好的訓練資料\tears-of-steel-q4-shared-v4-raw8.sqlite",
}
CFG = {
    "BBB": {"blocks": [0, 7, 15, 22, 29], "starts": [0, 37, 120, 300, 600]},
    "ToS": {"blocks": [0, 15, 29], "starts": [0, 5000, 12000, 18000]},
}
WIDTH, HEIGHT = 640, 16
LR_W, LR_H = 160, 4
STATIC_THR = 2.0


def phase_offsets(phase):
    p = phase - 1
    return (p % 4) * 2, (p // 4) * 2


def best_even_shift(a, b):
    H, W = a.shape
    best = (0, 0, float("inf"))
    for dy in (-4, -2, 0, 2, 4):
        for dx in range(-12, 13, 2):
            y0, y1 = 4 + max(0, -dy), H - 4 - max(0, dy)
            x0, x1 = 16 + max(0, -dx), W - 16 - max(0, dx)
            if y1 - y0 < 4 or x1 - x0 < 32:
                continue
            ac = a[y0:y1, x0:x1].astype(np.float32)
            bc = b[y0 + dy:y1 + dy, x0 + dx:x1 + dx].astype(np.float32)
            mae = float(np.abs(ac - bc).mean())
            if mae < best[2]:
                best = (dy, dx, mae)
    return best


def place(dec, ph, hr, sy=0, sx=0):
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


def run(db, cfg):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    per_t = {}
    for bi in cfg["blocks"]:
        rows = conn.execute(
            "SELECT frame_index, phase_id, raw_bayer_target, decoded FROM samples "
            "WHERE block_index=? ORDER BY frame_index", (bi,)).fetchall()
        phases = [r[1] for r in rows]
        targets = [np.frombuffer(r[2], dtype=np.uint8).reshape(HEIGHT, WIDTH) for r in rows]
        decs = [r[3] for r in rows]
        for T in (1, 2, 4, 8, 16):
            for s in cfg["starts"]:
                if s + T > len(rows):
                    continue
                tgt0 = targets[s].astype(np.float32)
                acc = {"static": [], "moving": []}
                for k in range(T):
                    hr = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
                    place(decs[s + k], phases[s + k], hr)
                    covered = ~np.isnan(hr)
                    err = np.abs(hr - tgt0)
                    if k == 0:
                        static_mask = np.ones_like(covered)
                    else:
                        diff = np.abs(targets[s + k].astype(np.float32) - tgt0)
                        static_mask = diff < STATIC_THR
                    for tag, m in (("static", covered & static_mask),
                                   ("moving", covered & ~static_mask)):
                        if m.any():
                            acc[tag].append(float(err[m].mean()))
                            acc[tag + "_n"] = acc.get(tag + "_n", 0) + int(m.sum())
                key = f"T={T}"
                slot = per_t.setdefault(key, {"static": [], "moving": []})
                slot["static"] += acc["static"]
                slot["moving"] += acc["moving"]
                slot.setdefault("static_n", 0)
                slot["static_n"] += acc.get("static_n", 0)
                slot.setdefault("moving_n", 0)
                slot["moving_n"] += acc.get("moving_n", 0)
    conn.close()
    out = {}
    for key, slot in per_t.items():
        out[key] = {
            "static_accum_mae": float(np.mean(slot["static"])) if slot["static"] else None,
            "moving_accum_mae": float(np.mean(slot["moving"])) if slot["moving"] else None,
            "static_pixel_share": slot["static_n"] / (slot["static_n"] + slot["moving_n"])
                                if (slot["static_n"] + slot["moving_n"]) else None,
            "n_groups": len(slot["static"]),
        }
    return out


def main():
    res = {}
    for name, db in DBS.items():
        res[name] = {"db": os.path.basename(db), "static_threshold": STATIC_THR, "results": run(db, CFG[name])}
    payload = json.dumps(res, ensure_ascii=False, indent=2)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs-static-motion-split.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(payload)
    print(payload)


if __name__ == "__main__":
    sys.exit(main())
