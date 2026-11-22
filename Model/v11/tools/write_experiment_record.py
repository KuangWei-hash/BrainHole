"""建立 v11 時序量測的實驗紀錄 JSON（Model/experiments/records/）。

沿用 record_experiment.py 的欄位結構，但這次量測是在 Model/v11/tools/ 下的
獨立腳本（不走 run_logged 包裝），因此以等價的持久紀錄補登：
含指令、起訖時間（由產物 mtime 取得）、環境、資料指紋、產物雜湊。
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

REPO = r"C:\Users\dinos\BrainHole"
MODEL = os.path.join(REPO, "Model")
V11 = os.path.join(MODEL, "v11")
REG = os.path.join(MODEL, "experiments", "records")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mtime(path):
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(os.path.getmtime(path)))


def pkg_version(name):
    try:
        out = subprocess.check_output([sys.executable, "-c", f"import {name};print({name}.__version__)"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
        return out
    except Exception:  # noqa: BLE001
        return None


def main():
    runs = sorted(f for f in os.listdir(V11) if f.startswith("runs-") and f.endswith(".json"))
    tools = sorted(f for f in os.listdir(os.path.join(V11, "tools")) if f.endswith(".py"))
    artifacts = []
    for name in runs + [os.path.join("tools", t) for t in tools]:
        p = os.path.join(V11, name)
        artifacts.append({"path": f"Model/v11/{name.replace(os.sep, '/')}",
                          "bytes": os.path.getsize(p), "sha256": sha256_of(p)})

    manifest = json.load(open(os.path.join(V11, "來源清單.json"), encoding="utf-8"))
    db_files = {
        "BBB_Q4": "training-data-preparation/拆好的訓練資料/big-buck-bunny-30s-q4-shared-v4-raw8-fixed.sqlite",
        "ToS_Q4": "training-data-preparation/拆好的訓練資料/tears-of-steel-q4-shared-v4-raw8.sqlite",
    }
    db_fp = {}
    for key, rel in db_files.items():
        p = os.path.join(REPO, rel.replace("/", os.sep))
        db_fp[key] = {"path": rel, "bytes": os.path.getsize(p),
                      "header_id": None}

    started = mtime(os.path.join(V11, "runs-motion-measure.json"))
    finished = mtime(os.path.join(V11, "runs-static-motion-split.json"))

    record = {
        "schema_version": 1,
        "id": "diagnostic--v11-時序量測--run-20261121",
        "kind": "diagnostic",
        "name": "v11-時序量測",
        "status": "completed",
        "provenance": "manual_equivalent_record",
        "note": "相位幾何、覆蓋率、幀對運動、靜態/運動分區的多幀相位累積誤差量測；無訓練、無模型",
        "output": "Model/v11",
        "started_at": started,
        "finished_at": finished,
        "cwd": REPO,
        "command_argv": [
            r"C:\Users\dinos\BrainHole\Model\.venv\Scripts\python.exe",
            "Model/v11/tools/measure_temporal_motion.py",
            "Model/v11/tools/measure_aligned_accumulation.py",
            "Model/v11/tools/measure_aligned_accumulation_tos.py",
            "Model/v11/tools/measure_aligned_accumulation_wide.py",
            "Model/v11/tools/measure_global_motion.py",
            "Model/v11/tools/measure_static_motion_split.py",
        ],
        "output_preexisted": False,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: pkg_version(name) for name in ("numpy", "torch")},
        },
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_sources": db_fp,
        "metrics": {
            "phase_geometry_mae_mean": 4.227223955715696,
            "phase_geometry_samples": 300,
            "coverage_ratio_by_T": {"1": 0.0625, "2": 0.125, "4": 0.25, "8": 0.5, "16": 1.0},
            "bbb_static_accum_mae_by_T": {"1": 3.708687524795532, "2": 3.658749213218689,
                                          "4": 3.6638334380819444, "8": 4.0542600911524564,
                                          "16": 4.482494999103871},
            "bbb_moving_accum_mae_by_T": {"2": 12.26493860880534, "4": 12.691465455835516,
                                          "8": 15.295158888168614, "16": 20.82157915313809},
            "tos_static_accum_mae_by_T": {"1": 4.630468686421712, "2": 4.657101720571518,
                                          "4": 4.679104556639989, "8": 4.625324756868424,
                                          "16": 4.592938273041336},
            "tos_moving_accum_mae_by_T": {"2": 10.962520897388458, "4": 10.965912850697835,
                                          "8": 13.138888687700838, "16": 17.86610171235638},
            "global_motion_px_median": {"dx_abs": 0.021, "dy_abs": 0.292},
            "duplicate_frame_ratio": 0.20022246941045607,
            "sample_counts": {"bbb_windows_per_T": 25, "tos_windows_per_T": 12,
                              "bbb_pair_shift_estimates": 375, "global_motion_pairs": 399},
        },
        "result": {
            "artifacts": artifacts,
            "external_sources": {
                "repos": len(manifest["repos"]),
                "papers": len(manifest["papers"]),
                "papers_title_verified": sum(1 for p in manifest["papers"] if p.get("verified")),
            },
        },
        "limitations": [
            "靜態門檻固定為 |Δ|<2，未做門檻掃描",
            "oracle 對齊由 raw_bayer_target 求得，模型只能用 decoded",
            "整數偶數位移無法涵蓋小數位移",
            "樣本數偏小（BBB 25-30 視窗、ToS 12 視窗），只測兩部影片",
            "量測的是量測誤差，不是模型輸出 MAE",
        ],
    }
    os.makedirs(REG, exist_ok=True)
    path = os.path.join(REG, record["id"] + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
