"""Offline JPEG XS reference decoder and phase-aware CFA interpolation."""
import argparse
import base64
from contextlib import closing
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from data import connect


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upscale(low, phases, mode="bilinear"):
    """low Bx1x4x160; phase 0..15; output Bx1x16x640."""
    planes = F.pixel_unshuffle(low, 2)  # B,4,2,80; colors never mix.
    dtype, device = low.dtype, low.device
    x = torch.arange(320, device=device, dtype=dtype)[None, None, :]
    y = torch.arange(8, device=device, dtype=dtype)[None, :, None]
    px = (phases % 4).to(dtype)[:, None, None]
    py = (phases // 4).to(dtype)[:, None, None]
    gx = ((x-px)/4 / 79*2-1).expand(-1, 8, -1)
    gy = ((y-py)/4 * 2-1).expand(-1, -1, 320)
    grid = torch.stack((gx, gy), dim=-1)
    enlarged = F.grid_sample(planes, grid, mode=mode, padding_mode="border", align_corners=True)
    return F.pixel_shuffle(enlarged, 2)


class DecodedCache:
    def __init__(self, path, db_path):
        with np.load(path, allow_pickle=False) as cache:
            self.metadata = json.loads(str(cache["metadata"]))
            self.low = cache["low"].copy()
            keys = cache["keys"]
        if self.metadata["database_sha256"] != fingerprint(db_path):
            raise ValueError("decoded cache belongs to a different database")
        self.indices = {tuple(key): index for index, key in enumerate(keys.tolist())}

    def get(self, key):
        return torch.from_numpy(self.low[self.indices[key]].copy()).float().reshape(1, 4, 160)/255


def decode_keys(db_path, keys, decoder):
    """Decode inference inputs from actual payloads, without reading RAW targets."""
    with closing(connect(db_path)) as db:
        header = json.loads(db.execute("SELECT definition FROM shared_header WHERE singleton=1").fetchone()[0])
        if db.execute("PRAGMA user_version").fetchone()[0] == 4:
            values = []
            for frame, block in keys:
                row = db.execute("SELECT decoded,header_id FROM samples WHERE frame_index=? AND block_index=?",
                                 (frame, block)).fetchone()
                if row is None or len(row[0]) != 640 or row[1] != header["id"]:
                    raise ValueError("missing or invalid stored decoded input")
                values.append(np.frombuffer(row[0], dtype=np.uint8).copy().reshape(1, 4, 160))
            return torch.from_numpy(np.stack(values)).float()/255
        prefix, suffix = base64.b64decode(header["prefix"]), base64.b64decode(header["suffix"])
        records = bytearray()
        for frame, block in keys:
            row = db.execute("SELECT jpegxs_payload,valid_length,codestream_length,header_id FROM samples "
                             "WHERE frame_index=? AND block_index=?", (frame, block)).fetchone()
            if row is None or row[1] != len(row[0]) or row[3] != header["id"]:
                raise ValueError("missing or invalid inference payload")
            coded = bytearray(prefix+row[0]+suffix)
            if header["patch_length"]:
                offset = header["length_offset"]
                coded[offset:offset+4] = struct.pack(">I", len(coded))
            if len(coded) != row[2] or len(coded) > 640:
                raise ValueError("invalid reconstructed inference codestream")
            records.extend(struct.pack(">I", len(coded))+coded)
    with tempfile.TemporaryDirectory() as folder:
        source, destination = Path(folder)/"input.batch", Path(folder)/"output.raw"
        source.write_bytes(records)
        subprocess.run([str(Path(decoder).resolve()), "decode-batch", "--input", str(source),
                        "--output", str(destination)], check=True)
        decoded = np.frombuffer(destination.read_bytes(), dtype=np.uint8).copy()
    if decoded.size != len(keys)*640:
        raise ValueError("incorrect decoded inference length")
    return torch.from_numpy(decoded).float().reshape(len(keys), 1, 4, 160)/255


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--decoder", type=Path, default=Path(__file__).resolve().parents[1] /
                        "training-data-preparation/tools/jpegxs-bridge/target/release/brainhole-jpegxs.exe")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    with closing(connect(args.db)) as db:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (3, 4):
            raise ValueError("requires schema v3 or v4")
        header = json.loads(db.execute("SELECT definition FROM shared_header WHERE singleton=1").fetchone()[0])
        if version == 4:
            rows = db.execute("SELECT frame_index,block_index,decoded,header_id FROM samples "
                              "ORDER BY frame_index,block_index").fetchall()
            if not rows or any(len(r[2]) != 640 or r[3] != header["id"] for r in rows):
                raise ValueError("invalid schema-v4 decoded inputs")
            low = np.stack([np.frombuffer(r[2], dtype=np.uint8).reshape(4, 160) for r in rows])
            metadata = {"database_sha256": fingerprint(args.db), "header_id": header["id"],
                        "samples": len(rows), "schema_version": 4,
                        "decoder_source": "SQLite decoded column, produced by native Go JPEG XS decoder",
                        "note": "Actual codestream reconstructions stored during dataset preparation; no RAW-target sampling."}
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.out, keys=np.array([(r[0], r[1]) for r in rows], dtype=np.int64),
                     low=low, metadata=json.dumps(metadata))
            print(json.dumps(metadata, indent=2), flush=True)
            return
        rows = db.execute("SELECT frame_index,block_index,jpegxs_payload,valid_length,codestream_length,header_id "
                          "FROM samples ORDER BY frame_index,block_index").fetchall()
    prefix, suffix = base64.b64decode(header["prefix"]), base64.b64decode(header["suffix"])
    unique, mapping = {}, []
    for frame, block, payload, length, full_length, header_id in rows:
        if header_id != header["id"] or length != len(payload):
            raise ValueError(f"invalid header/payload at {frame}/{block}")
        if payload not in unique:
            coded = bytearray(prefix+payload+suffix)
            if header["patch_length"]:
                offset = header["length_offset"]
                coded[offset:offset+4] = struct.pack(">I", len(coded))
            if len(coded) != full_length or len(coded) > 640:
                raise ValueError("invalid reconstructed codestream length")
            unique[payload] = (len(unique), bytes(coded))
        mapping.append(unique[payload][0])
    print(f"Decoding {len(unique)} unique streams for {len(rows)} samples", flush=True)
    with tempfile.TemporaryDirectory() as folder:
        source, destination = Path(folder)/"input.batch", Path(folder)/"output.raw"
        with source.open("wb") as stream:
            for _, coded in unique.values():
                stream.write(struct.pack(">I", len(coded))+coded)
        subprocess.run([str(args.decoder.resolve()), "decode-batch", "--input", str(source),
                        "--output", str(destination)], check=True)
        decoded = np.frombuffer(destination.read_bytes(), dtype=np.uint8)
    if decoded.size != len(unique)*640:
        raise ValueError("decoder returned unexpected block count")
    low = decoded.reshape(-1, 4, 160)[mapping].copy()
    metadata = {"database_sha256": fingerprint(args.db), "header_id": header["id"],
                "decoder_sha256": fingerprint(args.decoder), "samples": len(rows),
                "note": "Decoded only from codestream, not high-resolution labels."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, keys=np.array([(r[0], r[1]) for r in rows], dtype=np.int64),
             low=low, metadata=json.dumps(metadata))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    from record_experiment import run_logged
    run_logged("decoding", main)
