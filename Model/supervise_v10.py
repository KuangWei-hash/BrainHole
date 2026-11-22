"""Monitor a Q=5 v-prediction run and execute its pre-registered follow-ups."""
import argparse
import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from record_experiment import atomic_text, now, refresh_index, REGISTRY


REPO = Path(__file__).absolute().parent.parent
MODEL = REPO / "Model"
RUNS = MODEL / "runs"


def metrics(path):
    if not path.exists():
        return []
    rows = path.read_text(encoding="utf-8").splitlines(keepends=True)
    return [json.loads(row) for row in rows if row.endswith("\n")]


def attached_process(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return lambda: kernel.WaitForSingleObject(handle, 0) == 258, lambda: kernel.CloseHandle(handle)


def group_stats(rows):
    best = min(rows, key=lambda row: row["validation"]["mae_8bit"])
    validation = best["validation"]
    return {"epochs": len(rows), "best_epoch": best["epoch"],
            "best_mae": validation["mae_8bit"], "best_psnr": validation["psnr_raw"],
            "bilinear_mae": validation["gammas"]["0.0"]["mae_8bit"],
            "last_mae": rows[-1]["validation"]["mae_8bit"],
            "last_training_seconds": rows[-1]["train"]["seconds"]}


def choose_followup(candidate, report):
    low = report["frequency_errors"]
    drift = low[report["primary"]]["low_frequency_mae"] / low["bilinear"]["low_frequency_mae"]
    usable = candidate["best_mae"] <= 1.05 * candidate["bilinear_mae"]
    if usable and drift > 1.05:
        return {"run_d": True, "low_frequency_ratio": drift,
                "reason": "C 達到固定子集門檻，30 幀的低頻 MAE 比底圖高逾 5%，D 只加入 0.1 低頻損失。"}
    reason = ("C 達到固定子集門檻，沒有量到足夠的低頻漂移支持 D。" if usable else
              "C 未達固定子集門檻。保留時間步和取樣診斷，暫無已證實的單一訓練修正支持 D。")
    return {"run_d": False, "low_frequency_ratio": drift, "reason": reason}


class Batch:
    def __init__(self, args):
        self.args = args
        self.session = args.launch.resolve().parent
        self.launch = json.loads(args.launch.read_text(encoding="utf-8-sig"))
        self.report = {"status": "running", "started": now(), "jobs": {}, "groups": {},
                       "comparisons": {}, "decisions": []}
        self.env = os.environ.copy()
        self.env["PATH"] = str(args.ffmpeg_dir.resolve()) + os.pathsep + self.env["PATH"]
        self.data = ["--db", str(args.db.resolve()), "--decoded-cache", str(args.decoded_cache.resolve())]

    def save(self):
        self.report["updated"] = now()
        atomic_text(self.session / "report.json", json.dumps(self.report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    def watch(self, name, out, alive):
        last, changed = 0, time.monotonic()
        while alive():
            rows = metrics(out / "metrics.jsonl")
            if rows and not all(math.isfinite(row["validation"]["mae_8bit"]) and
                                math.isfinite(row["train"]["objective"]) for row in rows):
                raise RuntimeError(f"{name}: nonfinite metrics")
            if len(rows) != last:
                last, changed = len(rows), time.monotonic()
                self.report["groups"][name] = group_stats(rows)
            job = self.report["jobs"][name]
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            job.update(gpu=gpu.stdout.strip() or gpu.stderr.strip(), free_workspace_bytes=shutil.disk_usage(RUNS).free,
                       seconds_since_epoch=round(time.monotonic() - changed))
            if job["seconds_since_epoch"] > 600:
                log = self.session / f"{name.lower()}-stdout.log"
                job["last_stdout_lines"] = log.read_text(encoding="utf-8", errors="replace").splitlines()[-5:]
            self.save()
            time.sleep(20)
        rows = metrics(out / "metrics.jsonl")
        if len(rows) != 20 or rows[-1]["epoch"] != 20 or not all((out / f).is_file() for f in ("best.pt", "last.pt")):
            raise RuntimeError(f"{name}: training ended with {len(rows)} recorded epochs")
        self.report["groups"][name] = group_stats(rows)

    def child(self, name, script, out, options, training=False):
        cmd = [sys.executable, "-B", "-u", f"Model/{script}", *self.data,
               "--out", str(out), "--device", "cuda", *options]
        with (self.session / f"{name.lower()}-stdout.log").open("wb") as stdout, \
             (self.session / f"{name.lower()}-stderr.log").open("wb") as stderr:
            process = subprocess.Popen(cmd, cwd=REPO, env=self.env, stdout=stdout, stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            self.report["jobs"][name] = {"pid": process.pid, "output": str(out), "command": cmd, "status": "running"}
            self.save()
            if training:
                self.watch(name, out, lambda: process.poll() is None)
            code = process.wait()
        self.report["jobs"][name].update(status="completed" if code == 0 else "failed", exit_code=code)
        self.save()
        if code:
            raise RuntimeError(f"{name} exited with {code}; see {name.lower()}-stderr.log")
        return out

    def train(self, name, mode, options):
        out = RUNS / f"v10-q5-{name.lower()}-{self.launch['tag']}"
        return self.child(name, "train_v10.py", out, ["--mode", mode, "--condition", "both",
                          "--channels", "48", "--epochs", "20", "--batch-size", "16", "--microbatch", "4",
                          "--lr", "0.0001", "--sample-steps", "32", "--validation-count", "256", "--seed", "42",
                          *options], training=True)

    def evaluate(self, name, model, control, stop):
        out = RUNS / f"eval-v10-q5-{name.lower()}-{self.launch['tag']}"
        options = ["--checkpoint", str(model / "best.pt"), "--control-checkpoint", str(control / "best.pt"),
                   "--start-frame", "720", "--stop-frame", str(stop), "--preview-frame", "721", "--steps", "32",
                   "--reference-checkpoint", str(RUNS / "relu128-four-v9-history2-q5" / "best.pt"),
                   "--fsr-report", str(RUNS / "eval-echosr-lite-q5-20261121" / "report.json"), "--video"]
        self.child(name, "evaluate_v10.py", out, options)
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.report["comparisons"][name] = {key: report[key] for key in (
            "samples", "first_evaluated_frame", "last_evaluated_frame", "primary", "raw_projection_metrics",
            "rgb_reference_metrics", "frequency_errors", "seam_diagnostics", "numerical_fsr_passed")}
        self.save()
        return report

    def summarize(self):
        title = f"實驗紀錄-v10-Q5-RTX5060-{self.launch['tag']}"
        lines = [f"# {title}", "", "目標為同部 BBB Q=5 影片生成合理、可看的細節。",
                 "雙路同條雙幀輸入和 RAW 殘差輸出，訓練有效幀 1–719 共 21,570 條／輪，固定 256 條選模。",
                 "使用 FP32、有效 batch 16、microbatch 4、seed 42、DDIM 32 步，歷史 ε 組停在第 15 輪。", "",
                 f"批次狀態 `{self.report['status']}`。", ""]
        for group, values in self.report["groups"].items():
            lines += [f"- {group} 已紀錄 {values['epochs']} 輪，最佳第 {values['best_epoch']} 輪，"
                      f"固定 256 條 RAW MAE {values['best_mae']:.4f}，底圖 {values['bilinear_mae']:.4f}。"]
        for name, comparison in self.report["comparisons"].items():
            raw = comparison["raw_projection_metrics"]
            primary, base, control = (raw[key]["mae_8bit"] for key in (
                comparison["primary"], "bilinear", "v10-deterministic-control"))
            lines += [f"- {name} 比較第 {comparison['first_evaluated_frame']} 到 {comparison['last_evaluated_frame']} 幀，"
                      f"共 {comparison['samples']} 條。擴散 RAW MAE {primary:.4f}，底圖 {base:.4f}，"
                      f"非生成控制 {control:.4f}。"]
        if "c_diagnostics" in self.report:
            diagnostic = self.report["c_diagnostics"]
            lines += [f"- C 終端步 t=999 未裁切 RAW MAE {diagnostic['terminal_mae']:.4f}，"
                      f"超出合法範圍 {diagnostic['terminal_clip_fraction']:.2%}。"
                      f"同 16 條的 32／128 步 RAW MAE 為 {diagnostic['mae_32']:.4f}／{diagnostic['mae_128']:.4f}。"]
        lines.extend(["", *[item["reason"] for item in self.report["decisions"]], ""])
        if "error" in self.report:
            lines += [self.report["error"], ""]
        for name, job in self.report["jobs"].items():
            path = Path(job["output"])
            link = (Path("runs") / path.name).as_posix()
            lines.append(f"- [{name}]({link})，`{job['status']}`。")
        lines += ["", "影片觀感、泛化和即時性需要各自的驗證。", "",
                  f"[批次監督紀錄](runs/{self.session.name}/report.json)", ""]
        atomic_text(MODEL / f"{title}.md", "\n".join(lines))
        refresh_index(REGISTRY)

    def run(self):
        c = REPO / self.launch["run"]
        self.report["jobs"]["C"] = {"pid": self.launch["pid"], "output": str(c), "status": "running"}
        alive, close = attached_process(self.launch["pid"])
        try:
            self.watch("C", c, alive)
        finally:
            close()
        self.report["jobs"]["C"]["status"] = "completed"
        a = metrics(RUNS / "v10-diffusion-q5-20epochs" / "metrics.jsonl")
        if a:
            self.report["historical_a"] = group_stats(a)
        diagnostic = self.child("C-diagnostics", "diagnose_v10.py", RUNS / f"diagnose-v10-c-{self.launch['tag']}",
                                ["--checkpoint", str(c / "best.pt"), "--count", "64", "--sample-count", "16",
                                 "--steps", "32", "128"])
        diagnostic = json.loads((diagnostic / "report.json").read_text(encoding="utf-8"))
        terminal = diagnostic["teacher_forced"]["999"]
        self.report["c_diagnostics"] = {"terminal_mae": terminal["x0_unclipped_raw_mae"],
                                        "terminal_clip_fraction": terminal["clip_fraction"],
                                        "mae_32": diagnostic["sampling"]["32"]["raw_mae"],
                                        "mae_128": diagnostic["sampling"]["128"]["raw_mae"]}
        b = self.train("B", "deterministic", [])
        clip_report = self.evaluate("C-B-clip", c, b, 751)
        decision = choose_followup(self.report["groups"]["C"], clip_report)
        self.report["decisions"].append(decision)
        self.save()
        chosen, name = c, "C"
        if decision["run_d"]:
            d = self.train("D", "diffusion", ["--prediction-type", "v", "--low-frequency-weight", "0.1"])
            self.evaluate("D-B-clip", d, b, 751)
            groups = self.report["groups"]
            if groups["D"]["best_mae"] < groups["C"]["best_mae"]:
                chosen, name = d, "D"
        chosen_stats = self.report["groups"][name]
        if chosen_stats["best_mae"] <= 1.05 * chosen_stats["bilinear_mae"]:
            self.evaluate(f"{name}-B-full", chosen, b, 900)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--decoded-cache", required=True, type=Path)
    parser.add_argument("--ffmpeg-dir", required=True, type=Path)
    args = parser.parse_args()
    batch = Batch(args)
    batch.save()
    try:
        batch.run()
    except (Exception, KeyboardInterrupt) as exc:
        batch.report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        batch.report["status"] = "completed"
    finally:
        batch.save()
        batch.summarize()


if __name__ == "__main__":
    main()
