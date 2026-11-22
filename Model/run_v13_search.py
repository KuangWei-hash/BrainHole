"""Run and record sequential v13 follow-ups against the same S2 development keys."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import os
import subprocess
import sys
import time

from record_experiment import atomic_text, digest, run_logged, write_json


TRIALS = ("phase", "clean", "consistency", "guided", "aligned", "synthetic-fades", "more-gradient",
          "band-detail", "wider", "more-width", "seed-repeat", "lower-lr", "lower-lr-detail", "final-refinement")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def quality(value, baseline):
    dimensions = [value["raw"]["mae_8bit"] / baseline["raw"]["mae_8bit"],
                  value["boundary_mae"] / baseline["boundary_mae"]]
    for overall, region in (("high_frequency_mae", "gradient_mae"), ("temporal", "temporal")):
        ratios = [area[region] / baseline["roi"][name][region] for name, area in value["roi"].items()]
        dimensions.append(0.5 * value[overall] / baseline[overall] + 0.5 * sum(ratios) / len(ratios))
    return sum(dimensions) / len(dimensions)


def screens(value, v12):
    return (value["raw"]["mae_8bit"] < v12["raw"]["mae_8bit"]
            and value["high_frequency_mae"] < v12["high_frequency_mae"]
            and value["boundary_mae"] < v12["boundary_mae"]
            and value["temporal"] < v12["temporal"])


def candidates(report, reference, *, include_reference=False):
    for name, value in report["methods"].items():
        if not include_reference and value.get("sha256") == reference["sha256"]:
            continue
        if value.get("alpha") != 0.5 or not screens(value, report["methods"]["v12"]):
            continue
        yield {"name": name, "checkpoint": value["checkpoint"], "sha256": value["sha256"],
               "score": quality(value, reference), "values": value}


def minimum_raw(report):
    name, value = min(((name, value) for name, value in report["methods"].items() if "checkpoint" in value),
                      key=lambda item: item[1]["raw"]["mae_8bit"])
    return {"name": name, "values": value}


def save_ledger(out, ledger):
    write_json(out / "report.json", ledger)
    best = ledger["best"]
    video = Path(best["review"]) / f"{best['name']}-valid.mp4"
    relative = Path(os.path.relpath(video, out)).as_posix()
    lines = ["# v13 長開發窗口試驗", "",
             f"狀態 `{ledger['status']}`｜截止 {ledger['deadline']}", "",
             f"目前數值初篩領先為 [原解析度有效窗口影片]({relative})。",
             "排序使用 RAW、同色細節、條界及時序。結果限 BBB Q=5 的 736–779 開發幀。",
             f"兩種強度中最低的 RAW MAE 為 {ledger['minimum_raw']['values']['raw']['mae_8bit']:.4f}。",
             "", "| 試驗 | 狀態 | RAW MAE | 數值排序 |", "| --- | --- | ---: | ---: |"]
    for row in ledger["trials"]:
        candidate = row.get("candidate")
        raw = f"{candidate['values']['raw']['mae_8bit']:.4f}" if candidate else "—"
        rank = f"{candidate['score']:.4f}" if candidate else "—"
        lines.append(f"| {row['stage']} | {row['status']} | {raw} | {rank} |")
    atomic_text(out / "summary.md", "\n".join(lines) + "\n")


def plan(stage, winner, prior, width_results):
    if stage == "phase":
        return {"variant": "R2-frames-phase"}, "establish the unsupervised phase-mask control on the longer window"
    if stage == "consistency":
        return {"variant": "R2-frames-consistency"}, "test an explicit decoded-only prior after learned weights remained nearly uniform"
    if stage == "guided":
        return {"variant": "R2-frames-guide"}, "S1 equal/adaptive weights behaved alike; supervise reliability at fixed architecture"
    if stage == "aligned":
        return {"variant": "R2-frames-align"}, "challenge the long development window with local alignment"
    same = {"variant": winner["variant"], "width": winner.get("width", 32),
            "context": winner.get("context", "compact"), "gradient_weight": winner.get("gradient_weight", 0.1),
            "band_weight": winner.get("band_weight", 0.)}
    if stage == "wider":
        return {**same, "width": 48}, "vary channel capacity at fixed optimizer and loss"
    if stage == "more-width":
        if not width_results:
            return None, "width 48 did not improve the development ranking"
        return {**same, "width": 64}, "check whether the width improvement extends to width 64"
    if stage == "more-gradient":
        weight = 0.2 if same["gradient_weight"] == 0.1 else 0.3
        return {**same, "gradient_weight": weight}, "increase same-color detail supervision at matched training budget"
    if stage == "band-detail":
        return {**same, "band_weight": 0.15}, "match the target's actual same-color high-frequency structure"
    if stage in ("lower-lr", "lower-lr-detail"):
        current = same["gradient_weight"]
        weight = current if stage == "lower-lr" else (0.2 if current == 0.3 else current + 0.1)
        return {**same, "gradient_weight": round(weight, 1), "lr": 5e-5, "warm_start": prior["checkpoint"]}, (
            "compare two warm starts from the same weights with a fresh optimizer")
    if stage == "seed-repeat":
        return {**same, "seed": 43}, "repeat the leading from-scratch configuration with another seed"
    return {**same, "lr": 2e-5, "warm_start": prior["checkpoint"]}, "test one slower refinement of the strongest candidate"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("db", "scene-plan", "cache", "initial-review", "reference-checkpoint", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    parser.add_argument("--native-checkpoint", type=Path)
    parser.add_argument("--fade-cache", type=Path)
    parser.add_argument("--stages", choices=TRIALS, nargs="+", default=TRIALS)
    args = parser.parse_args()
    if args.deadline.utcoffset() is None:
        parser.error("deadline must include its UTC offset")
    initial = read(args.initial_review)
    ref_hash = digest(args.reference_checkpoint)
    reference = next(row for row in initial["methods"].values()
                     if row.get("sha256") == ref_hash and row.get("alpha") == 0.5)
    if initial["split"] != "S2-dev" or initial["source_range"] != [721, 779]:
        raise ValueError("initial review must cover the S2 development keys")
    args.out.mkdir(parents=True)
    options = list(candidates(initial, reference, include_reference=True))
    best = min(options, key=lambda item: item["score"])
    best["review"] = str(args.initial_review.parent)
    ledger = {"deadline": args.deadline.isoformat(), "initial_review": str(args.initial_review),
              "initial_review_sha256": digest(args.initial_review), "reference_sha256": ref_hash,
              "initial_best": best, "best": best, "minimum_raw": minimum_raw(initial),
              "status": "running", "trials": []}
    save_ledger(args.out, ledger)
    work = Path(__file__).resolve().parent
    common = ["--db", str(args.db), "--scene-plan", str(args.scene_plan), "--cache", str(args.cache)]
    estimate = 35 * 60
    width_results = False
    warm_parent = best

    def invoke(script, argv, log):
        with log.open("wb") as stream:
            subprocess.run([sys.executable, "-B", script, *common, *argv], cwd=work,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    for stage in args.stages:
        if ledger["minimum_raw"]["values"]["raw"]["mae_8bit"] == 0:
            ledger["status"] = "exact"
            break
        now = datetime.now().astimezone()
        if now + timedelta(seconds=estimate) >= args.deadline:
            ledger["status"] = "time-budget"
            break
        if stage == "lower-lr":
            warm_parent = best
        parent = warm_parent if stage == "lower-lr-detail" else best
        picked = parent["values"]["checkpoint"]
        winner = dict(load_settings(picked))
        if stage == "seed-repeat" and winner.get("warm_start"):
            winner = load_settings(winner["warm_start"]["path"])
        if stage == "clean":
            settings = ({"variant": "R2-frames-clean", "native_checkpoint": str(args.native_checkpoint)}
                        if args.native_checkpoint else None)
            reason = "test the frozen native CFA corrector before fusion; use the phase-mask group as control"
        elif stage == "synthetic-fades":
            settings = ({"variant": winner["variant"], "width": winner.get("width", 32),
                         "context": winner.get("context", "compact"),
                         "gradient_weight": winner.get("gradient_weight", 0.1),
                         "band_weight": winner.get("band_weight", 0.), "fade_cache": str(args.fade_cache)}
                        if args.fade_cache and args.fade_cache.is_file() else None)
            if settings and "native_checkpoint" in winner:
                settings["native_checkpoint"] = winner["native_checkpoint"]["path"]
            reason = "isolate training-only photometric transition examples at unchanged update budget"
        else:
            settings, reason = plan(stage, winner, parent, width_results)
            if (settings and settings["variant"] == winner["variant"] and "native_checkpoint" in winner
                    and "warm_start" not in settings):
                settings["native_checkpoint"] = winner["native_checkpoint"]["path"]
            if settings and "fade_cache" in winner:
                settings["fade_cache"] = winner["fade_cache"]["path"]
        row = {"stage": stage, "reason": reason, "started_at": now.isoformat(), "settings": settings,
               "parent": picked, "status": "running" if settings else "skipped"}
        ledger["trials"].append(row)
        save_ledger(args.out, ledger)
        if not settings:
            continue
        print(stage, reason, settings, flush=True)
        run = args.out / f"{stage}-weights"
        review = args.out / f"{stage}-review"
        row.update({"run": str(run), "review": str(review)})
        save_ledger(args.out, ledger)
        flags = [item for key, value in settings.items() for item in ("--" + key.replace("_", "-"), str(value))]
        began = time.monotonic()
        try:
            invoke("train_v13.py", ["--stage", "S2", "--out", str(run), *flags,
                   "--deadline", args.deadline.isoformat()], args.out / f"{stage}-train.log")
            if datetime.now().astimezone() >= args.deadline:
                row.update({"status": "time-budget", "finished_at": datetime.now().astimezone().isoformat()})
                ledger["status"] = "time-budget"
                save_ledger(args.out, ledger)
                break
            best_epoch = min((json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()),
                             key=lambda item: item["validation"]["mae_8bit"])["epoch"]
            checkpoints = [run / f"epoch-{epoch}.pt" for epoch in (5, 10, 20)]
            if best_epoch not in (5, 10, 20):
                checkpoints.append(run / "best-mae.pt")
            invoke("evaluate_v13.py", ["--split", "S2-dev", "--out", str(review), "--checkpoint",
                   str(args.reference_checkpoint), *map(str, checkpoints), "--alpha", "0.5", "1"], args.out / f"{stage}-review.log")
            review_data = read(review / "report.json")
            raw_leader = minimum_raw(review_data)
            if raw_leader["values"]["raw"]["mae_8bit"] < ledger["minimum_raw"]["values"]["raw"]["mae_8bit"]:
                ledger["minimum_raw"] = raw_leader
            top = min(candidates(review_data, reference), key=lambda item: item["score"], default=None)
            if top is not None:
                top["review"] = str(review)
            row.update({"status": "completed", "candidate": top, "improved": bool(top and top["score"] < best["score"])})
            if row["improved"]:
                best = top
                ledger["best"] = best
                if stage == "wider":
                    width_results = True
            elapsed = time.monotonic() - began
            estimate = max(15 * 60, int(elapsed * (1.7 if stage in ("more-gradient", "wider") else 1.25)))
            print(stage, "score", top["score"] if top else "failed screening", "leader", best["name"], flush=True)
        except subprocess.CalledProcessError as error:
            row.update({"status": "failed", "returncode": error.returncode})
            print(stage, "failed; logs retained", flush=True)
        row["finished_at"] = datetime.now().astimezone().isoformat()
        save_ledger(args.out, ledger)
    if ledger["status"] == "running":
        ledger["status"] = "completed"
    save_ledger(args.out, ledger)


def load_settings(path):
    config_path = Path(path).parent / "config.json"
    return read(config_path)


if __name__ == "__main__":
    run_logged("checks", main)
