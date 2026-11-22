"""Assemble the .summary.json stored next to a dataset from its validation report."""
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
extra_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

report = json.loads(report_path.read_text(encoding="utf-8"))
summary = {
    "status": report["status"],
    "started_at": report["started_at"],
    "finished_at": report["finished_at"],
    "schema": report["schema"],
    "integrity": report["integrity"],
    "dataset_config": report["dataset_config"],
    "dataset_status": report["dataset_status"],
    "errors": report["errors"],
    "samples": report["samples"],
    "frames": report["frames"],
    "phases": report["phases"],
    "invalid_frames": report["invalid_frames"],
    "invalid_samples": report["invalid_samples"],
    "reference_samples": report["reference_samples"],
    "identical_targets_and_pairing": report["identical_targets_and_pairing"],
    "changed_payloads": report["changed_payloads"],
    "reference_comparison": report.get("reference_comparison"),
    "payload_bytes": report["payload_bytes"],
    "max_codestream_bytes": report["max_codestream_bytes"],
    "decoded_all_128": report["decoded_all_128"],
    "go_rust_decode_parity": report["go_rust_decode_parity"],
    "bhxs_body_bps": report["bhxs_body_bps"],
    "outputs": report["outputs"],
}
if extra_path is not None:
    summary.update(json.loads(extra_path.read_text(encoding="utf-8-sig")))
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({k: summary[k] for k in ("status", "samples", "frames", "payload_bytes", "bhxs_body_bps",
                                          "decoded_all_128", "go_rust_decode_parity")}, indent=2))
