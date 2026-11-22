"""v11 時序量測：相位幾何驗證、影格間運動量、多幀相位覆盖率。

輸入：Q=4 v4 raw8 SQLite（samples 內含 decoded / raw_bayer_target / phase_id）。

輸出（stdout JSON）：
  {
    "db": ..., "blocks": [...],
    "duplication": {...},
    "phase_geometry": {...},
    "motion": {...},
    "phase_coverage": {...}
  }

規則：只記錄觀測到的值；無法量測的欄位標為 null，不補造。
"""
import json
import os
import sqlite3
import sys
from collections import Counter

import numpy as np

DB = r"C:\Users\dinos\BrainHole\training-data-preparation\拆好的訓練資料\big-buck-bunny-30s-q4-shared-v4-raw8-fixed.sqlite"
BLOCKS = [0, 7, 15, 22, 29]
WIDTH, HEIGHT = 640, 16
LR_W, LR_H = 160, 4


def phase_offsets(phase):
    """Go Downsample 的逆向：phase -> (px, py)，px/py 為 8x8 tile 內 2x2 區塊的左上角位移。"""
    p = phase - 1
    return (p % 4) * 2, (p // 4) * 2


def load_block(conn, block_index):
    rows = conn.execute(
        "SELECT frame_index, source_frame_index, phase_id, raw_bayer_target, decoded "
        "FROM samples WHERE block_index=? ORDER BY frame_index", (block_index,)).fetchall()
    return rows


def best_even_shift(a, b, r=6):
    """在 ±r 的偶數位移中找使 a 與 b 最匹配的位移；回傳 (dy,dx,mae)。"""
    H, W = a.shape
    crop = (slice(4, H - 4), slice(16, W - 16))
    ac = a[crop].astype(np.float32)
    best = (0, 0, float("inf"))
    for dy in range(-r, r + 1, 2):
        for dx in range(-r, r + 1, 2):
            ys = slice(4 + max(0, -dy), H - 4 - max(0, dy))
            xs = slice(16 + max(0, -dx), W - 16 - max(0, dx))
            bc = b[ys, xs].astype(np.float32)
            mae = float(np.abs(ac[: bc.shape[0], : bc.shape[1]] - bc).mean())
            if mae < best[2]:
                best = (dy, dx, mae)
    return best


def main():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    out = {"db": os.path.basename(DB), "blocks": {}}

    # ---------- 1. 重複幀排列 ----------
    dup = []
    src_by_frame = {}
    for bi in BLOCKS:
        rows = load_block(conn, bi)
        for fi, sfi, ph, _, _ in rows:
            src_by_frame.setdefault(fi, sfi)
    frames = sorted(src_by_frame)
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] != 1:
            continue
        dup.append(src_by_frame[frames[i]] - src_by_frame[frames[i - 1]])
    dup_counter = Counter(dup)
    out["duplication"] = {
        "frame_count": len(frames),
        "source_delta_histogram": {str(k): v for k, v in sorted(dup_counter.items())},
        "identical_content_ratio": dup_counter.get(0, 0) / max(1, sum(dup_counter.values())),
        "note": "source_frame_index 差 0 代表該影格內容與前一幀完全相同（24->30fps 重複幀）",
    }

    # ---------- 2. 相位幾何驗證 ----------
    geo_errs = []
    for bi in BLOCKS:
        rows = load_block(conn, bi)
        for fi, sfi, ph, tgt, dec in rows[:60]:
            tgt = np.frombuffer(tgt, dtype=np.uint8).reshape(HEIGHT, WIDTH)
            dec = np.frombuffer(dec, dtype=np.uint8).reshape(LR_H, LR_W)
            px, py = phase_offsets(ph)
            # 明確重建：out[(2j+dy)*160+2i+dx] = raw[(8j+py+dy)*640+8i+px+dx]
            rec = np.zeros((LR_H, LR_W), dtype=np.uint8)
            for j in range(2):
                for i in range(80):
                    for dy in range(2):
                        for dx in range(2):
                            rec[2 * j + dy, 2 * i + dx] = tgt[8 * j + py + dy, 8 * i + px + dx]
            geo_errs.append(float(np.abs(rec.astype(np.float32) - dec.astype(np.float32)).mean()))
    out["phase_geometry"] = {
        "checked_samples": len(geo_errs),
        "mae_mean": float(np.mean(geo_errs)) if geo_errs else None,
        "mae_max": float(np.max(geo_errs)) if geo_errs else None,
        "note": "由 phase 反推的取樣位置與 decoded 的一致性（差值即 JPEG XS 有損誤差）",
    }

    # ---------- 3. 影格間運動量 ----------
    motion = {"same_source": [], "distinct_source": []}
    shift_hist = Counter()
    for bi in BLOCKS:
        rows = load_block(conn, bi)
        tgts = {fi: np.frombuffer(t, dtype=np.uint8).reshape(HEIGHT, WIDTH) for fi, _, _, t, _ in rows}
        srcs = {fi: sfi for fi, sfi, _, _, _ in rows}
        for i in range(1, len(rows)):
            f0, f1 = rows[i - 1][0], rows[i][0]
            if f1 - f0 != 1:
                continue
            dy, dx, mae = best_even_shift(tgts[f0], tgts[f1])
            key = "same_source" if srcs[f0] == srcs[f1] else "distinct_source"
            motion[key].append({"dy": dy, "dx": dx, "mae": mae})
            if key == "distinct_source":
                shift_hist[f"dy={dy},dx={dx}"] += 1
    for key, vals in motion.items():
        if not vals:
            motion[key] = {"count": 0}
            continue
        maes = [v["mae"] for v in vals]
        motion[key] = {
            "count": len(vals),
            "mae_mean": float(np.mean(maes)),
            "mae_median": float(np.median(maes)),
            "mae_p90": float(np.percentile(maes, 90)),
            "shift_histogram_top": shift_hist.most_common(8) if key == "distinct_source" else None,
        }
    out["motion"] = motion

    # ---------- 4. 多幀相位覆盖率與實測精度 ----------
    cov = {}
    for T in (1, 2, 4, 8, 16):
        meas = []
        hits = 0
        tot = 0
        for bi in BLOCKS[:2]:
            rows = load_block(conn, bi)
            for start in (0, 37, 120, 300, 600):
                if start + T > len(rows):
                    continue
                grp = rows[start:start + T]
                tgt = np.frombuffer(grp[0][3], dtype=np.uint8).reshape(HEIGHT, WIDTH).astype(np.float32)
                hr = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
                for _, _, ph, _, dec in grp:
                    px, py = phase_offsets(ph)
                    d = np.frombuffer(dec, dtype=np.uint8).reshape(LR_H, LR_W)
                    for j in range(2):
                        for i in range(80):
                            for dy in range(2):
                                for dx in range(2):
                                    hr[8 * j + py + dy, 8 * i + px + dx] = d[2 * j + dy, 2 * i + dx]
                m = ~np.isnan(hr)
                hits += int(m.sum())
                tot += m.size
                if m.any():
                    meas.append(float(np.abs(hr[m] - tgt[m]).mean()))
        cov[f"T={T}"] = {
            "coverage_ratio": hits / tot if tot else None,
            "measured_pixel_mae_mean": float(np.mean(meas)) if meas else None,
            "note": "以 T 幀各自的 phase 把 decoded 放回高解析網格後，實測到的像素比例與其誤差",
        }
    out["phase_coverage"] = cov

    conn.close()
    payload = json.dumps(out, ensure_ascii=False, indent=2)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs-motion-measure.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(payload)
    print(payload)


if __name__ == "__main__":
    sys.exit(main())
