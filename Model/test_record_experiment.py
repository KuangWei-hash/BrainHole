import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from record_experiment import atomic_text, collect_result, read_json, run_logged, update_active_record


class RecordingTests(unittest.TestCase):
    def test_atomic_write_retries_transient_access_denied_and_preserves_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"index.md"
            output.write_text("old", encoding="utf-8")
            replace = Path.replace
            calls = []

            def temporarily_locked(source, target):
                calls.append(target)
                if len(calls) < 3:
                    raise PermissionError("sharing violation")
                return replace(source, target)

            with patch.object(Path, "replace", temporarily_locked), patch("record_experiment.time.sleep"):
                atomic_text(output, "new")
            self.assertEqual(len(calls), 3)
            self.assertEqual(output.read_text(), "new")
            with patch.object(Path, "replace", side_effect=PermissionError("denied")), patch("record_experiment.time.sleep"):
                with self.assertRaises(PermissionError):
                    atomic_text(output, "lost")
            self.assertEqual(output.read_text(), "new")
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_training_progress_and_completed_record(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output, registry = root/"run", root/"registry"

            def train():
                output.mkdir()
                (output/"config.json").write_text(json.dumps({"epochs": 1, "history": 2}), encoding="utf-8")
                (output/"metrics.jsonl").write_text(json.dumps({"epoch": 1, "validation": {
                    "mae_8bit": 20., "psnr_raw": 18.}})+"\n", encoding="utf-8")
                (output/"best.pt").write_bytes(b"test checkpoint")
                (output/"last.pt").write_bytes(b"test checkpoint")
                update_active_record()
                running = read_json(next((registry/"records").glob("*.json")))
                self.assertEqual(running["status"], "running")
                self.assertEqual(len(running["result"]["epochs"]), 1)
                return 42

            self.assertEqual(run_logged("training", train, argv=["--out", str(output)], registry_root=registry), 42)
            record = read_json(next((registry/"records").glob("*.json")))
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["result"]["best_epoch_by_validation_mae"], 1)
            self.assertEqual(record["result"]["missing_expected_artifacts"], [])
            self.assertIsNotNone(record["source_snapshot"])
            self.assertTrue((registry.parent/"實驗總表.md").exists())

    def test_failures_interruptions_and_reused_output_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output, registry = root/"existing", root/"registry"
            output.mkdir()
            (output/"config.json").write_text('{"epochs": 99}', encoding="utf-8")
            for error in (RuntimeError("failed attempt"), KeyboardInterrupt()):
                def fail():
                    raise error
                with self.assertRaises(type(error)):
                    run_logged("training", fail, argv=["--out", str(output)], registry_root=registry)
            records = [read_json(path) for path in (registry/"records").glob("*.json")]
            self.assertEqual(len(records), 2)
            self.assertEqual({record["status"] for record in records}, {"failed", "interrupted"})
            self.assertTrue(all("result" not in record for record in records))
            self.assertTrue(all(record["output_preexisted"] for record in records))
            self.assertTrue(all(record["exception"]["traceback"] for record in records))
            self.assertEqual(json.loads((output/"config.json").read_text())["epochs"], 99)

    def test_per_block_archive_is_lossless(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root/"evaluation"
            output.mkdir()
            (output/"report.json").write_text('{"samples":2}', encoding="utf-8")
            original = b'{"frame":1,"mae":2.3}\n{"frame":2,"mae":4.5}\n'
            (output/"blocks.jsonl").write_bytes(original)
            result = collect_result("evaluation", output, root/"registry", "test")
            saved = root/"registry/details/test.blocks.jsonl.gz"
            self.assertEqual(gzip.decompress(saved.read_bytes()), original)
            self.assertEqual(result["report"]["samples"], 2)


if __name__ == "__main__":
    unittest.main()
