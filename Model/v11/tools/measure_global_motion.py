"""v11 量測（第四版）：整幀重建 + 次像素全域運動估計。

用 30 個 block_index 條帶拼回 640x480 Bayer 整幀，取綠平面平均作為亮度代理，
以相位相關（phase correlation）估計相鄰幀的全域次像素位移。

輸出 Model/v11/runs-global-motion.json（UTF-8, no BOM）。
"""
import json
import os
import sqlite3
import sys

import numpy as np

DB = r"C:\Users\dinos\BrainHole\training-data-preparation\拆好的訓練資料\big-buck-bunny-30s-q4-shared-v4-raw8-fixed.sqlite"
WIDTH, HEIGHT, ROWS = 640, 16, 480
NBLOCK = 30
FRAME_RANGE = list(range(0, 400))
WINDOW = 64


def assemble(conn, frame):
    frame_rows = np.zeros((ROWS, WIDTH), dtype=np.uint8)
    got = 0
    for bi in range(NBLOCK):
        row = conn.execute(
            "SELECT raw_bayer_target FROM samples WHERE frame_index=? AND block_index=?",
            (frame, bi)).fetchone()
        if row is None:
            continue
        blk = np.frombuffer(row[0], dtype=np.uint8).reshape(HEIGHT, WIDTH)
        frame_rows[bi * HEIGHT:(bi + 1) * HEIGHT] = blk
        got += 1
    return frame_rows if got == NBLOCK else None


def luma_proxy(bayer):
    """RGGB：Bayer[0,0]=R, [0,1]=G, [1,0]=G, [1,1]=B。取綠平均並降取樣 2x。"""
    g = (bayer[0::2, 1::2].astype(np.float32) + bayer[1::2, 0::2].astype(np.float32)) * 0.5
    return g


def phase_correlate(a, b):
    """回傳 (dx, dy)，使 b 大致等於 a 平移 (dx, dy)（以 a 為參考）。"""
    A = np.fft.rfft2(a)
    B = np.fft.rfft2(b)
    R = A * np.conj(B)
    mag = np.abs(R)
    mag[mag == 0] = 1e-9
    R = R / mag
    cc = np.fft.irfft2(R, s=a.shape)
    idx = np.unravel_index(np.argmax(cc), cc.shape)
    dy, dx = int(idx[0]), int(idx[1])
    peak = float(cc[idx])
    # 次像素細化（拋物線）
    def refine(v, arr):
        n = arr.shape[0]
        i0 = (v - 1) % n
        i1 = v % n
        i2 = (v + 1) % n
        y0, y1, y2 = float(arr[i0]), float(arr[v]), float(arr[i2])
        denom = (y0 - 2 * y1 + y2)
        return 0.5 * (y0 - y2) / denom if denom != 0 else 0.0

    sub_dx = refine(dx, cc[idx[0], :])
    sub_dy = refine(dy, cc[:, idx[1]])
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
        sub_dx -= 1
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
        sub_dy -= 1
    return dx + sub_dx, dy + sub_dy, peak


def main():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    frames = {}
    for f in FRAME_RANGE:
        b = assemble(conn, f)
        if b is not None:
            frames[f] = b
    conn.close()

    srcmap = {}
    # 來源索引（重複幀判定）
    con2 = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    for f in frames:
        row = con2.execute("SELECT DISTINCT source_frame_index FROM samples WHERE frame_index=?",
                            (f,)).fetchone()
        srcmap[f] = row[0] if row else None
    con2.close()

    results = []
    keys = sorted(frames)
    for i in range(1, len(keys)):
        f0, f1 = keys[i - 1], keys[i]
        if f1 - f0 != 1:
            continue
        a = luma_proxy(frames[f0])
        b = luma_proxy(frames[f1])
        # 裁掉邊緣避免邊界效應
        a = a[WINDOW:-WINDOW, WINDOW:-WINDOW]
        b = b[WINDOW:-WINDOW, WINDOW:-WINDOW]
        dx, dy, peak = phase_correlate(a, b)
        results.append({
            "frame": f1,
            "same_source": srcmap[f0] == srcmap[f1],
            "dx": round(float(dx), 3),
            "dy": round(float(dy), 3),
            "peak": round(float(peak), 4),
        })

    same = [r for r in results if r["same_source"]]
    diff = [r for r in results if not r["same_source"]]
    dxs = np.array([r["dx"] for r in diff]) if diff else np.array([0.0])
    dys = np.array([r["dy"] for r in diff]) if diff else np.array([0.0])

    out = {
        "db": os.path.basename(DB),
        "frames_measured": len(results),
        "same_source_pairs": {
            "count": len(same),
            "max_abs_dx": max((abs(r["dx"]) for r in same), default=None),
            "max_abs_dy": max((abs(r["dy"]) for r in same), default=None),
        },
        "distinct_source_pairs": {
            "count": len(diff),
            "dx_abs_median": float(np.median(np.abs(dxs))),
            "dx_abs_p90": float(np.percentile(np.abs(dxs), 90)),
            "dx_abs_max": float(np.abs(dxs).max()),
            "dy_abs_median": float(np.median(np.abs(dys))),
            "dy_abs_p90": float(np.percentile(np.abs(dys), 90)),
            "dy_abs_max": float(np.abs(dys).max()),
            "frac_dy_gt_2": float(np.mean(np.abs(dys) > 2)),
            "frac_dy_gt_4": float(np.mean(np.abs(dys) > 4)),
            "frac_dx_gt_4": float(np.mean(np.abs(dxs) > 4)),
            "frac_dx_gt_12": float(np.mean(np.abs(dxs) > 12)),
        },
        "note": ("dx/dy 以綠平面平均亮度做相位相關估計，單位為高解析像素（已降取樣 2 倍後再放大回原解析度）。"
                 "相鄰幀間隔為 1/30 秒。"),
        "per_frame": results,
    }
    payload = json.dumps(out, ensure_ascii=False, indent=2)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs-global-motion.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(payload)
    summary = {k: v for k, v in out.items() if k != "per_frame"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
