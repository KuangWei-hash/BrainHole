"""Rebuild a paired Go JPEG XS dataset without replacing prior data; archive validation."""
import argparse
import base64
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

from data import connect
from record_experiment import MODEL, REPOSITORY, digest, inventory, now, run_logged, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reference", type=Path,
                        help="an existing dataset of the same source video; when given, the new "
                             "dataset's targets, timing and phases are compared against it. "
                             "Omit for a source video that has no prior dataset.")
    parser.add_argument("--encoder-config", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"),
                        help="dataset-level split recorded in the metadata; assign per video")
    args = parser.parse_args()
    requested_config = json.loads(args.encoder_config.read_text(encoding="utf-8"))
    expected_q = requested_config.get("quantization")
    if type(expected_q) is not int or not 0 <= expected_q <= 15:
        parser.error("encoder config must declare an integer quantization in 0..15")
    work = args.out.with_suffix("")
    if args.database.exists() or args.out.exists() or work.exists():
        parser.error("outputs already exist; choose new database/report paths")
    work.mkdir(parents=True)
    preparation = REPOSITORY/"training-data-preparation"
    paths = sorted(set([*preparation.rglob("*.go"), preparation/"go.mod", preparation/"go.sum",
                        *preparation.glob("encoder*.json"), args.encoder_config.resolve(),
                        *[p for p in (preparation/"internal/jpegxs/testdata").rglob("*") if p.is_file()]]))
    hashes = {str(p.relative_to(REPOSITORY)): digest(p) for p in paths}
    snapshot_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    snapshot = MODEL/"experiments"/"source_snapshots"/snapshot_id
    for path in paths:
        destination = snapshot/path.relative_to(REPOSITORY)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(path, destination)
    report = {"status": "running", "started_at": now(), "commands": [],
              "requested_quantization": expected_q,
              "requested_split": args.split,
              "go_source_snapshot": {"directory": str(snapshot), "sha256": hashes},
              "inputs": inventory([p for p in (args.source, args.reference, args.encoder_config) if p is not None], hashes=True)}
    write_json(args.out, report)
    env = dict(os.environ, GOCACHE=str(REPOSITORY/".cache"/"go-build"))

    def command(argv, name, cwd=REPOSITORY):
        log = work/(name+".log")
        entry = {"argv": [str(x) for x in argv], "cwd": str(cwd), "started_at": now(), "log": str(log)}
        report["commands"].append(entry)
        write_json(args.out, report)
        print(f"{name}: started", flush=True)
        start = time.perf_counter()
        with log.open("wb") as stream:
            result = subprocess.run(entry["argv"], cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        entry.update(finished_at=now(), seconds=time.perf_counter()-start, exit_code=result.returncode)
        write_json(args.out, report)
        if result.returncode:
            print(log.read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
            result.check_returncode()
        print(f"{name}: completed ({entry['seconds']:.1f}s)", flush=True)

    try:
        command(["go", "test", "-count=1", "-timeout", "150s", "./..."], "go-tests", preparation)
        generator = work/"video-to-dataset.exe"
        exporter = work/"dataset-export.exe"
        command(["go", "build", "-o", generator.resolve(), "./cmd/video-to-dataset"], "build-generator", preparation)
        command(["go", "build", "-o", exporter.resolve(), "./cmd/dataset-export"], "build-exporter", preparation)
        command([generator, "-input", args.source.resolve(), "-output", args.database.resolve(),
                 "-encoder-config", args.encoder_config.resolve(), "-split", args.split], "generate")
        stream_path = args.database.with_suffix(".bhxs")
        command([exporter, "-input", args.database.resolve(), "-output", stream_path.resolve(),
                 "-with-decoded=false"], "export")
        with closing(connect(args.database)) as db:
            report["schema"] = db.execute("PRAGMA user_version").fetchone()[0]
            report["integrity"] = db.execute("PRAGMA quick_check").fetchone()[0]
            report["dataset_config"] = json.loads(db.execute("SELECT value FROM dataset_info WHERE key='config'").fetchone()[0])
            report["dataset_status"] = db.execute("SELECT value FROM dataset_info WHERE key='status'").fetchone()[0]
            report["errors"] = db.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
            samples, frames, phases = db.execute("SELECT COUNT(*),COUNT(DISTINCT frame_index),COUNT(DISTINCT phase_id) FROM samples").fetchone()
            report.update(samples=samples, frames=frames, phases=phases)
            report["invalid_frames"] = db.execute("SELECT COUNT(*) FROM (SELECT frame_index FROM samples GROUP BY frame_index HAVING COUNT(*)!=30 OR MIN(block_index)!=0 OR MAX(block_index)!=29 OR MIN(phase_id)!=MAX(phase_id))").fetchone()[0]
            report["invalid_samples"] = db.execute("SELECT COUNT(*) FROM samples WHERE phase_id!=(frame_index%16)+1 OR length(decoded)!=640 OR length(raw_bayer_target)!=10240 OR length(jpegxs_payload)!=valid_length OR valid_length>638 OR codestream_length>640").fetchone()[0]
            if args.reference is None:
                # A source video without a prior dataset: there is nothing to
                # compare against, so the pairing/target equivalence check is
                # explicitly reported as skipped rather than silently passed.
                report["reference_samples"] = None
                report["identical_targets_and_pairing"] = None
                report["changed_payloads"] = None
                report["reference_comparison"] = "skipped: no prior dataset exists for this source video"
            else:
                db.execute("ATTACH DATABASE ? AS reference", (args.reference.resolve().as_uri()+"?mode=ro",))
                report["reference_samples"] = db.execute("SELECT COUNT(*) FROM reference.samples").fetchone()[0]
                report["identical_targets_and_pairing"] = db.execute("SELECT COUNT(*) FROM samples a JOIN reference.samples b USING(frame_index,block_index) WHERE a.raw_bayer_target=b.raw_bayer_target AND a.phase_id=b.phase_id AND a.source_frame_index=b.source_frame_index AND a.source_timestamp_us=b.source_timestamp_us").fetchone()[0]
                report["changed_payloads"] = db.execute("SELECT COUNT(*) FROM samples a JOIN reference.samples b USING(frame_index,block_index) WHERE a.jpegxs_payload!=b.jpegxs_payload").fetchone()[0]
                report["reference_comparison"] = "compared against the prior dataset of the same source video"
            report["payload_bytes"] = dict(zip(("min", "mean", "max", "total"), db.execute("SELECT MIN(valid_length),AVG(valid_length),MAX(valid_length),SUM(valid_length) FROM samples").fetchone()))
            report["max_codestream_bytes"] = db.execute("SELECT MAX(codestream_length) FROM samples").fetchone()[0]
            report["decoded_all_128"] = db.execute("SELECT COUNT(*) FROM samples WHERE decoded=?", (bytes([128])*640,)).fetchone()[0]
            header = json.loads(db.execute("SELECT definition FROM shared_header WHERE singleton=1").fetchone()[0])
            batch_path, expected_path = work/"reference-decode.batch", work/"go-decoded.raw"
            with batch_path.open("wb") as batch, expected_path.open("wb") as expected:
                for payload, size, header_id, decoded in db.execute("SELECT jpegxs_payload,codestream_length,header_id,decoded FROM samples ORDER BY frame_index,block_index"):
                    coded = bytearray(base64.b64decode(header["prefix"])+payload+base64.b64decode(header["suffix"]))
                    if header["patch_length"]:
                        offset = header["length_offset"]
                        coded[offset:offset+4] = struct.pack(">I", len(coded))
                    if len(coded) != size or header_id != header["id"]:
                        raise ValueError("invalid reconstructed stream")
                    batch.write(struct.pack(">I", len(coded))+coded)
                    expected.write(decoded)
        if (report["schema"] != 4 or report["integrity"] != "ok" or report["dataset_status"] != "complete"
                or report["errors"] or report["invalid_frames"] or report["invalid_samples"]
                or (args.reference is not None and (samples != report["reference_samples"]
                    or samples != report["identical_targets_and_pairing"]))
                or report["dataset_config"]["Encoder"]["quantization"] != expected_q
                or report["dataset_config"].get("Split", "train") != args.split):
            raise ValueError("dataset validation failed; see report")
        rust = preparation/"tools/jpegxs-bridge/target/release/brainhole-jpegxs.exe"
        rust_output = work/"rust-decoded.raw"
        command([rust, "decode-batch", "--input", batch_path.resolve(), "--output", rust_output.resolve()], "reference-decode")
        report["go_rust_decode_parity"] = {"samples": samples, "go_sha256": digest(expected_path),
                                           "rust_sha256": digest(rust_output), "reference_decoder_sha256": digest(rust)}
        if digest(expected_path) != digest(rust_output):
            raise ValueError("Go/Rust decoded data differ")
        report["bhxs_body_bps"] = (report["payload_bytes"]["total"]+2*samples+4*frames)*8/(frames/30)
        report["outputs"] = inventory([args.database, stream_path, *work.iterdir()], hashes=True)
        report.update(status="completed", finished_at=now())
        write_json(args.out, report)
        print(json.dumps({k: report[k] for k in ("samples", "frames", "phases", "payload_bytes", "bhxs_body_bps", "decoded_all_128", "go_rust_decode_parity")}, indent=2), flush=True)
    except BaseException:
        report.update(status="failed", finished_at=now())
        write_json(args.out, report)
        raise


if __name__ == "__main__":
    run_logged("diagnostic", main)
