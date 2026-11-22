"""Read-only schema-v3/v4 datasets. Missing history is skipped, never fabricated."""
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DecodedDataset(Dataset):
    """Pair each encoded window with actual decoded RAW (v3 inputs; legacy v2 labels)."""
    def __init__(self, dataset, cache):
        self.dataset, self.cache = dataset, cache
        self.keys, self.headers = dataset.keys, dataset.headers

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        frame, block = self.keys[index]
        lows = decoded_window(self.cache, self.dataset.input_keys(frame, block))
        return (*sample, lows)


def connect(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def decoded_window(cache, keys):
    return torch.stack([cache.get(key) if key is not None else torch.zeros(1, 4, 160)
                        for key in keys])


class RawDataset(Dataset):
    def __init__(self, path, history=1, frame_start=0, frame_stop=None, v9_upper_strip=False):
        self.path = str(Path(path).resolve())
        self.history = history
        self.v9_upper_strip = v9_upper_strip
        self.connection = None
        with closing(connect(path)) as db:
            if db.execute("PRAGMA user_version").fetchone()[0] not in (3, 4):
                raise ValueError("requires SQLite schema v3 or v4 RAW8 targets")
            rows = db.execute("SELECT frame_index,block_index,header_id FROM samples ORDER BY frame_index,block_index").fetchall()
        self.headers = {row[2] for row in rows}
        keys = {(f, b) for f, b, _ in rows if f >= frame_start and (frame_stop is None or f < frame_stop)}
        self.keys = [(f, b) for f, b in sorted(keys)
                     if all((f-k, b) in keys for k in range(history))
                     and (not v9_upper_strip or b == 0 or (f, b-1) in keys)]
        if not self.keys:
            raise ValueError("no complete history windows in selected frame range")

    def __len__(self):
        return len(self.keys)

    def input_keys(self, frame, block):
        keys = [(frame-k, block) for k in range(self.history)]
        if self.v9_upper_strip:
            keys.append((frame, block-1) if block > 0 else None)
        return keys

    def __getitem__(self, index):
        if self.connection is None:
            self.connection = connect(self.path)
        frame, block = self.keys[index]
        rows = self.connection.execute(
            "SELECT frame_index,phase_id,jpegxs_payload,valid_length,raw_bayer_target,header_id "
            "FROM samples WHERE block_index=? AND frame_index BETWEEN ? AND ? ORDER BY frame_index DESC",
            (block, frame-self.history+1, frame)).fetchall()
        if len(rows) != self.history:
            raise ValueError("database changed or history is incomplete")
        input_keys = self.input_keys(frame, block)
        if self.v9_upper_strip:
            upper = self.connection.execute(
                "SELECT frame_index,phase_id,jpegxs_payload,valid_length,raw_bayer_target,header_id "
                "FROM samples WHERE frame_index=? AND block_index=?", (frame, block-1)).fetchone() if block > 0 else None
            if block > 0 and upper is None:
                raise ValueError("upper strip disappeared from database")
            rows.append(upper)
        tokens = np.full((len(input_keys), 640), 256, dtype=np.int64)
        phases, lengths = [], []
        for k, row in enumerate(rows):
            if row is None:
                phases.append(-1)
                lengths.append(0)
                continue
            f, phase, payload, length, raw, header = row
            if f != input_keys[k][0] or phase != f % 16 + 1:
                raise ValueError("invalid frame/phase pairing")
            if header != rows[0][5]:
                raise ValueError("history crosses encoder/header settings")
            if not 0 < length <= 638 or length != len(payload) or len(raw) != 10240:
                raise ValueError("invalid payload/RAW length; 640-byte record reserves 2 bytes for length")
            # Length is supplied separately; the model sees a distinct padding token.
            tokens[k, :2] = (length >> 8, length & 255)
            tokens[k, 2:2+length] = np.frombuffer(payload, dtype=np.uint8)
            phases.append(phase-1)
            lengths.append(length)
        target = np.frombuffer(rows[0][4], dtype=np.uint8).copy().reshape(1, 16, 640)
        return (torch.from_numpy(tokens), torch.tensor(phases), torch.tensor(lengths),
                torch.from_numpy(target).float() / 255)
