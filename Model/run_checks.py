"""Run the Model test suite with persistent experiment recording."""
import argparse
import os
from pathlib import Path
import unittest

from record_experiment import run_logged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fsr-cli", type=Path, help="use a local copy of the official FSR executable")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("checks output already exists; choose a new directory")
    if args.fsr_cli:
        os.environ["BRAINHOLE_TEST_FSR_CLI"] = str(args.fsr_cli.resolve(strict=True))
    args.out.mkdir(parents=True)
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
    with (args.out / "tests.log").open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print((args.out / "tests.log").read_text(encoding="utf-8"), flush=True)
    if not result.wasSuccessful():
        raise RuntimeError("Model tests failed; see tests.log")


if __name__ == "__main__":
    run_logged("checks", main)
