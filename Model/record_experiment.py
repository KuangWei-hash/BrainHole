"""Durable experiment records outside ignored runs/, including failed attempts."""
import argparse
from datetime import datetime
import gzip
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import time
import traceback
import uuid

MODEL = Path(__file__).resolve().parent
REPOSITORY = MODEL.parent
REGISTRY = MODEL/"experiments"
_ACTIVE = None


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(REPOSITORY).as_posix()
    except ValueError:
        return str(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_replace(temporary, path):
    # Windows can deny replacement while another process reads a shared file.
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        atomic_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def environment():
    packages = {}
    for name in ("torch", "numpy", "Pillow"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": sys.version, "platform": platform.platform(), "packages": packages}
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        result["available_gpu"] = torch.cuda.get_device_name()
    return result


def source_snapshot(registry):
    files = sorted([*MODEL.glob("*.py"), *MODEL.glob("requirements*.txt"),
                    *(MODEL/"v10").rglob("*.py"), *(MODEL/"v10").glob("*.md"),
                    *(MODEL/"v11").glob("*.py"), *(MODEL/"v11").glob("*.md"),
                    *(MODEL/"v12").glob("*.py"), *(MODEL/"v12").glob("*.md"),
                    *(MODEL/"v13").rglob("*.py"), *(MODEL/"v13").rglob("*.md"), *(MODEL/"v13").glob("*.json"),
                    MODEL/"AGENTS.md", MODEL/"作法.MD", MODEL/"驗收標準.md"])
    hashes = {path.relative_to(MODEL).as_posix(): digest(path) for path in files}
    identifier = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = registry/"source_snapshots"/identifier
    destination.mkdir(parents=True, exist_ok=True)
    for path in files:
        target = destination/path.relative_to(MODEL)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name+"."+uuid.uuid4().hex+".tmp")
            try:
                shutil.copy2(path, temporary)
                if digest(temporary) != hashes[path.relative_to(MODEL).as_posix()]:
                    raise RuntimeError(f"source changed during capture: {path}")
                try:
                    temporary.rename(target)
                except FileExistsError:
                    pass
                except PermissionError:
                    if not target.exists():
                        atomic_replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    return {"id": identifier, "directory": relative(destination), "sha256": hashes}


def argument_value(args, name):
    for index, item in enumerate(args):
        if item.startswith(name+"="):
            return item.split("=", 1)[1]
        if item == name and index+1 < len(args):
            return args[index+1]
    return None


def data_sources(args):
    sources = {}
    for name in ("--db", "--val-db", "--decoded-cache", "--val-decoded-cache", "--checkpoint", "--warm-start",
                 "--scene-plan", "--source-video", "--fsr-report", "--fsr-cli", "--reference-checkpoint",
                 "--candidate-video", "--anchor-checkpoint", "--control-checkpoint", "--audit-freeze",
                 "--native-checkpoint", "--initial-review", "--fade-cache", "--band-ensemble", "--gated-ensemble"):
        value = argument_value(args, name)
        if value is None:
            continue
        # Some options accept a sequence of files.
        values = [value]
        if name in ("--checkpoint", "--candidate-video", "--band-ensemble") and name in args:
            values = []
            for item in args[args.index(name)+1:]:
                if item.startswith("--"):
                    break
                values.append(item)
        files = []
        for value in values:
            path = Path(value).resolve()
            if path.is_file():
                files.extend(inventory([path], hashes=True))
                wal = Path(str(path)+"-wal")
                if wal.is_file() and wal.stat().st_size:
                    files.extend(inventory([wal], hashes=True))
            else:
                files.append({"path": relative(path), "missing": True})
        sources[name] = files
    return sources


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inventory(paths, hashes=False):
    result = []
    for path in sorted(paths):
        if path.is_file():
            item = {"path": relative(path), "bytes": path.stat().st_size}
            if hashes:
                item["sha256"] = digest(path)
            result.append(item)
    return result


def collect_result(kind, output, registry, identifier, final=True):
    result = {}
    if kind == "training":
        if (output/"config.json").exists():
            result["config"] = read_json(output/"config.json")
        if (output/"metrics.jsonl").exists():
            result["epochs"] = [json.loads(line) for line in
                                (output/"metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        epochs = result.get("epochs", [])
        if epochs:
            best = min(epochs, key=lambda item: item["validation"]["mae_8bit"])
            result["best_epoch_by_validation_mae"] = best["epoch"]
            result["best_validation"] = best["validation"]
        config = result.get("config", {})
        budget_status = output/"budget_status.json"
        if budget_status.exists():
            result["budget_status"] = read_json(budget_status)
        result["coverage"] = ("unknown" if not config else
                              "partial_epochs" if budget_status.exists() else
                              "partial_steps" if config.get("max_steps") or config.get("val_steps")
                              else "full_epochs")
        if final:
            result["missing_expected_artifacts"] = [name for name in
                ("config.json", "metrics.jsonl", "last.pt") if not (output/name).exists()]
            if not any((output/name).exists() for name in ("best.pt", "best-mae.pt")):
                result["missing_expected_artifacts"].append("best.pt")
    elif kind == "evaluation" and (output/"report.json").exists():
        result["report"] = read_json(output/"report.json")
        blocks = output/"blocks.jsonl"
        if blocks.exists() and final:
            destination = registry/"details"/(identifier+".blocks.jsonl.gz")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            try:
                with blocks.open("rb") as source, temporary.open("wb") as target:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
                        shutil.copyfileobj(source, compressed)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            result["archived_blocks"] = {"path": relative(destination), "format": "gzip JSONL",
                                         "uncompressed_sha256": digest(blocks)}
    elif kind in ("benchmark", "diagnostic") and output.exists():
        result["report"] = read_json(output)
    elif kind == "decoding" and output.exists():
        import numpy as np
        with np.load(output, allow_pickle=False) as cache:
            result["metadata"] = json.loads(str(cache["metadata"]))
    if kind == "inference":
        paths = [output.with_suffix(suffix) for suffix in (".raw", ".png")]
    elif output.is_dir():
        paths = output.rglob("*")
    else:
        paths = [output]
    result["artifacts"] = inventory(paths, hashes=final)
    return result


def save_record(registry, record, *, index=True):
    record["recorded_at"] = now()
    write_json(registry/"records"/(record["id"]+".json"), record)
    if index:
        refresh_index(registry)


def refresh_index(registry):
    paths = sorted((registry/"records").glob("*.json"))
    rows = [(path, read_json(path)) for path in paths]
    lines = ["# 實驗總表", "", "本表自動由 experiments/records 產生；詳細解讀見各次 Markdown 實驗紀錄。",
             "所有完成、失敗、中斷、流程檢查皆保留；不同驗證集合或訓練預算不可直接當作公平架構比較。", "",
             "completed 表示執行完成，不代表品質驗收通過；品質門檻與限制保存在評估 report 及詳細解讀中。", "",
             "歷史補登只記錄可從既有產物確認的資料，原始指令、起訖時間與當時程式快照缺失處不補造。", ""]
    for kind, title in (("training", "訓練"), ("evaluation", "評估"), ("benchmark", "效能量測"),
                        ("checks", "程式檢查"),
                        ("decoding", "解碼資料"), ("diagnostic", "輸入診斷"),
                        ("inference", "推理"), ("artifact_note", "其他歷史產物")):
        subset = [(path, record) for path, record in rows if record["kind"] == kind]
        if not subset:
            continue
        lines += [f"## {title}（{len(subset)}）", ""]
        if kind == "training":
            lines += ["| 實驗 | 狀態 | 架構／幀數 | 輪數／最佳輪 | 驗證 MAE／PSNR | 驗證實際／集合 | 範圍 |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for path, record in subset:
                data = record.get("result", {})
                config, best = data.get("config", {}), data.get("best_validation", {})
                score = f"{best['mae_8bit']:.3f}" if best else "—"
                if best and "psnr_raw" in best:
                    score += f" / {best['psnr_raw']:.3f} dB"
                available = config.get("validation_samples")
                evaluated = config.get("validation_selected")
                if evaluated is None:
                    evaluated = (min(available, config["val_steps"]*config["batch_size"])
                                 if available is not None and config.get("val_steps") else available)
                validation_count = f"{evaluated} / {available}" if available is not None else "—"
                architecture = config.get("architecture", "v1" if config else "—")
                lines.append(f"| [{record['name']}](experiments/records/{path.name}) | {record['status']} | "
                             f"{architecture} / {config.get('history', '—')} | "
                             f"{len(data.get('epochs', []))} / {data.get('best_epoch_by_validation_mae', '—')} | "
                             f"{score} | {validation_count} | {data.get('coverage', '—')} |")
        else:
            lines += ["| 實驗 | 狀態 | 範圍／說明 |", "| --- | --- | --- |"]
            for path, record in subset:
                report = record.get("result", {}).get("report", {})
                if kind == "evaluation":
                    detail = f"{report.get('samples', '—')} 個區塊；設定起點 {report.get('start_frame', '—')}"
                    if "first_evaluated_frame" in report:
                        detail += f"；實際 {report['first_evaluated_frame']}…{report['last_evaluated_frame']} 幀"
                else:
                    detail = record.get("note", kind)
                lines.append(f"| [{record['name']}](experiments/records/{path.name}) | {record['status']} | {detail} |")
        lines.append("")
    lines += ["## 既有詳細解讀", "",
              *[f"- [{path.stem}]({path.name})" for path in sorted(MODEL.glob("實驗紀錄-*.md"))], ""]
    design_docs = sorted((MODEL/"v10").glob("*.md")) if (MODEL/"v10").is_dir() else []
    if design_docs:
        lines += ["## 設計文件", "",
                  *[f"- [{path.stem}](v10/{path.name})" for path in design_docs], ""]
    lines += ["## 資料準備驗證報告", ""]
    for path in sorted((REPOSITORY/"training-data-preparation"/"驗證報告").glob("*.md")):
        lines.append(f"- [{path.stem}](../{path.relative_to(REPOSITORY).as_posix()})")
    atomic_text(registry.parent/"實驗總表.md", "\n".join(lines)+"\n")


def identifier_for(kind, output):
    name = output.stem if output.suffix else output.name
    return kind+"--"+re.sub(r"[^\w.-]+", "_", name)


def update_active_record():
    if _ACTIVE is None:
        return
    registry, record, output, preexisting = _ACTIVE
    if not preexisting:
        record["result"] = collect_result(record["kind"], output, registry, record["id"], final=False)
    save_record(registry, record, index=False)


def run_logged(kind, function, default_output=None, *, argv=None, registry_root=None):
    global _ACTIVE
    args = list(sys.argv[1:] if argv is None else argv)
    if "--help" in args or "-h" in args:
        return function()
    registry = Path(registry_root) if registry_root is not None else REGISTRY
    output = argument_value(args, "--out") or default_output
    # Invalid invocations without an output still get a failed-attempt record.
    output = Path(output).resolve() if output is not None else MODEL/"runs"/"unspecified-output"
    identifier = identifier_for(kind, output)+"--run-"+uuid.uuid4().hex[:12]
    if kind == "training":
        preexisting = output.exists() and (not output.is_dir() or any(output.iterdir()))
    elif kind == "evaluation":
        preexisting = output.exists()
    elif kind == "inference":
        preexisting = any(output.with_suffix(suffix).exists() for suffix in (".raw", ".png"))
    else:
        preexisting = output.exists()
    record = {"schema_version": 1, "id": identifier, "kind": kind, "name": output.name,
              "status": "running", "provenance": "automatic_invocation", "output": relative(output),
              "started_at": now(), "finished_at": None, "cwd": str(Path.cwd()),
              "command_argv": [sys.executable, sys.argv[0], *args],
              "output_preexisted": preexisting, "source_snapshot": None,
              "environment": environment()}
    save_record(registry, record, index=False)
    _ACTIVE = (registry, record, output, preexisting)
    failure = None
    try:
        record["source_snapshot"] = source_snapshot(registry)
        record["data_sources"] = data_sources(args)
        save_record(registry, record, index=False)
        value = function()
    except BaseException as error:
        failure = error
        record["status"] = ("interrupted" if isinstance(error, KeyboardInterrupt) else
                            "completed" if isinstance(error, SystemExit) and error.code in (None, 0) else "failed")
        record["exception"] = {"type": type(error).__name__, "message": str(error),
                               "traceback": traceback.format_exc()}
        raise
    else:
        record["status"] = "completed"
        return value
    finally:
        try:
            record["finished_at"] = now()
            record["environment"] = environment()
            if not preexisting:
                record["result"] = collect_result(kind, output, registry, identifier)
            else:
                record["note"] = "Output preexisted; prior artifacts were not attributed to this attempt."
            save_record(registry, record)
        except Exception:
            if failure is None:
                raise
            print("Experiment archival also failed:\n"+traceback.format_exc(), file=sys.stderr)
        finally:
            _ACTIVE = None


def backfill(registry=REGISTRY):
    existing_records = [read_json(path) for path in (registry/"records").glob("*.json")]
    existing_outputs = {(record["kind"], record["output"]) for record in existing_records}
    existing_artifacts = {artifact["path"] for record in existing_records
                          for artifact in record.get("result", {}).get("artifacts", [])}
    candidates = []
    for folder in sorted((MODEL/"runs").iterdir()):
        if folder.is_dir() and (folder/"config.json").exists():
            candidates.append(("training", folder))
        if folder.is_dir() and (folder/"report.json").exists():
            candidates.append(("evaluation", folder))
    candidates += [("benchmark", path) for path in sorted((MODEL/"runs").glob("*-latency.json"))]
    candidates += [("diagnostic", path) for path in sorted((MODEL/"runs").rglob("*diagnostic.json"))]
    candidates += [("decoding", path) for path in sorted((MODEL/"runs").glob("decoded-*.npz"))]
    # Standalone historical inference outputs have no recoverable structured command.
    candidates += [("artifact_note", path) for path in sorted((MODEL/"runs").glob("*.raw"))
                   if relative(path) not in existing_artifacts]
    created = 0
    for kind, output in candidates:
        identifier = identifier_for(kind, output)
        if (kind, relative(output)) in existing_outputs:
            continue
        result = collect_result(kind, output, registry, identifier)
        status = "artifact_only" if kind == "artifact_note" else "completed"
        if kind == "training" and (result.get("missing_expected_artifacts") or
                len(result.get("epochs", [])) != result.get("config", {}).get("epochs")):
            status = "incomplete"
        record = {"schema_version": 1, "id": identifier, "kind": kind, "name": output.name,
                  "status": status, "output": relative(output), "result": result,
                  "provenance": "historical_artifact_backfill", "started_at": None, "finished_at": None,
                  "command_argv": None, "source_snapshot": None,
                  "note": "Original command, timestamps and original source snapshot were not retained; "
                          "only existing artifacts are archived. Unknown details are not reconstructed."}
        if kind == "artifact_note":
            record["result"]["artifacts"] = inventory([output, output.with_suffix(".png")], hashes=True)
            record["note"] = "Standalone historical inference artifacts; exact invocation not retained."
        save_record(registry, record)
        created += 1
    refresh_index(registry)
    print(f"Archived {created} historical experiment records to {registry}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill", action="store_true", help="archive existing runs without replacing records")
    options = parser.parse_args()
    if options.backfill:
        backfill()
    else:
        refresh_index(REGISTRY)
