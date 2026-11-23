"""Measure lossless RAW codebook storage and verify a position-addressable archive."""
import argparse
import json
import lzma
from pathlib import Path
import zipfile
import zlib

import numpy as np
import torch

from record_experiment import digest, run_logged, write_json


CODECS = {
    "zlib-9": (lambda data: zlib.compress(data, level=9), zlib.decompress),
    "xz-6": (lambda data: lzma.compress(data, preset=6), lzma.decompress),
}
PREDICTORS = ("raw", "temporal", "temporal-cfa-horizontal")


def encode(chronological, predictor):
    if predictor == "raw":
        return chronological
    delta = chronological.copy()
    delta[:, 1:] = chronological[:, 1:] - chronological[:, :-1]
    if predictor == "temporal-cfa-horizontal":
        original = delta.copy()
        delta[..., 2:] = original[..., 2:] - original[..., :-2]
    return delta


def decode(data, shape, predictor):
    values = np.frombuffer(data, dtype=np.uint8).reshape(shape)
    if predictor == "raw":
        return values
    if predictor == "temporal-cfa-horizontal":
        values = values.copy()
        for parity in range(2):
            values[..., parity::2] = np.cumsum(values[..., parity::2], axis=-1, dtype=np.uint8)
    return np.cumsum(values, axis=1, dtype=np.uint8)


def read_position(archive, metadata, position):
    """Restore a single vertical position in the checkpoint's original ranking."""
    _, unpack = CODECS[metadata["codec"]]
    frames = metadata["source_frames"][position]
    order = np.argsort(frames, kind="stable")
    data = unpack(archive.read(f"raw/{position:02d}"))
    chronological = decode(data, (1, len(frames), *metadata["strip_shape"]), metadata["predictor"])[0]
    return chronological[np.argsort(order)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    raw = checkpoint["raw"].transpose(0, 1).numpy()
    frames = checkpoint["source_frames"].transpose(0, 1).numpy()
    if raw.dtype != np.uint8 or raw.shape[2:] != (16, 640) or frames.shape != raw.shape[:2]:
        raise ValueError("expected ranked 640x16 RAW8 entries with a source frame for each entry")
    order = np.argsort(frames, axis=1, kind="stable")
    chronological = np.take_along_axis(raw, order[..., None, None], axis=1)

    results = []
    best_size = raw.nbytes
    best = None
    for predictor in PREDICTORS:
        values = encode(chronological, predictor)
        for name, (pack, unpack) in CODECS.items():
            whole = pack(values.tobytes())
            restored = decode(unpack(whole), raw.shape, predictor)
            if not np.array_equal(restored, chronological):
                raise ValueError(f"failed whole-table roundtrip for {predictor}/{name}")
            chunks = [pack(position.tobytes()) for position in values]
            independent = sum(map(len, chunks))
            results.append({"predictor": predictor, "codec": name, "whole_payload_bytes": len(whole),
                            "independent_position_payload_bytes": independent})
            if independent < best_size:
                best_size = independent
                best = (predictor, name, chunks)
    if best is None:
        raise ValueError("no tested position-addressable encoding reduced storage")

    predictor, codec, chunks = best
    metadata = {"version": 1, "predictor": predictor, "codec": codec, "strip_shape": list(raw.shape[2:]),
                "source_frames": frames.tolist(), "checkpoint_config": checkpoint["config"]}
    archive_path = args.out.with_name("codebook-lossless.zip")
    with zipfile.ZipFile(archive_path, "x") as archive:
        archive.writestr("metadata.json", json.dumps(metadata, separators=(",", ":")))
        for i, chunk in enumerate(chunks):
            archive.writestr(f"raw/{i:02d}", chunk)
    with zipfile.ZipFile(archive_path) as archive:
        saved = json.loads(archive.read("metadata.json"))
        for i, original in enumerate(raw):
            if not np.array_equal(read_position(archive, saved, i), original):
                raise ValueError(f"archive roundtrip failed at position {i}")

    report = {"checkpoint_bytes": args.checkpoint.stat().st_size, "checkpoint_sha256": digest(args.checkpoint),
              "raw_bytes": raw.nbytes, "entries": int(raw.shape[0] * raw.shape[1]), "results": results,
              "archive": {"path": str(archive_path), "bytes": archive_path.stat().st_size,
                          "sha256": digest(archive_path), "codec": codec, "predictor": predictor},
              "exact_roundtrip_bytes": raw.nbytes, "raw_bytes_per_position": raw[0].nbytes}
    write_json(args.out, report)
    print(f"{raw.nbytes:,} RAW bytes → {report['archive']['bytes']:,} archived bytes; byte-exact roundtrip", flush=True)


if __name__ == "__main__":
    run_logged("benchmark", main)
