"""Multi-frame windows from a RAW8 schema-v4 database.

Frame ranges are inclusive and scene starts mark the first frame of each scene.
"""
from bisect import bisect_right
from pathlib import Path
import sqlite3
from contextlib import closing

import torch
from torch.utils.data import Dataset

WIDTH, HEIGHT = 640, 16
LR_WIDTH, LR_HEIGHT = 160, 4
BLOCKS = 30


def connect(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def build_windows(path, frames=4, frame_start=0, frame_stop=None, blocks=None, scene_starts=()):
    """List of ``(block_index, [frame_index, ...])`` windows, oldest frame first."""
    if frames < 1:
        raise ValueError("frames must be positive")
    stop = frame_stop if frame_stop is not None else 1 << 60
    query = ("SELECT frame_index, block_index FROM samples "
             "WHERE frame_index BETWEEN ? AND ? ORDER BY block_index, frame_index")
    with closing(connect(path)) as conn:
        rows = conn.execute(query, (frame_start, stop)).fetchall()
    grouped = {}
    for frame, block in rows:
        grouped.setdefault(block, []).append(frame)
    windows = []
    for block, frames_seen in sorted(grouped.items()):
        if blocks is not None and block not in blocks:
            continue
        for index in range(frames - 1, len(frames_seen)):
            window = frames_seen[index-frames+1:index+1]
            if window[-1] - window[0] != frames - 1:
                continue
            cut = bisect_right(scene_starts, window[0])
            if cut < len(scene_starts) and scene_starts[cut] <= window[-1]:
                continue
            windows.append((block, window))
    return windows


class V11Dataset(Dataset):
    """T-frame windows of one block.

    Each item is ``(decoded, phases, target, key)`` with
    ``decoded`` ``[T, 1, 4, 160]`` float, ``phases`` ``[T]`` int64 (0-based),
    ``target`` ``[1, 16, 640]`` float and ``key`` the window's frame index.
    """

    def __init__(self, path, frames=4, frame_start=0, frame_stop=None, blocks=None,
                 scene_starts=(), windows=None):
        self.path = path
        self.frames = frames
        self.windows = (build_windows(path, frames, frame_start, frame_stop, blocks, scene_starts)
                        if windows is None else [(block, frame_ids[-frames:]) for block, frame_ids in windows])
        if not self.windows:
            raise ValueError("no complete windows in the requested range")
        self.conn = None

    def __len__(self):
        return len(self.windows)

    def _connection(self):
        if self.conn is None:
            self.conn = connect(self.path)
        return self.conn

    def __getitem__(self, index):
        block, frames = self.windows[index]
        conn = self._connection()
        decoded, phases = read_inputs(conn, frames, block)
        target_row = conn.execute(
            "SELECT raw_bayer_target FROM samples WHERE frame_index=? AND block_index=?",
            (frames[-1], block)).fetchone()
        target = torch.frombuffer(bytearray(target_row[0]), dtype=torch.uint8)
        target = target.float().div_(255.0).view(1, HEIGHT, WIDTH)
        return decoded, phases, target, torch.tensor(frames[-1])

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None


def read_inputs(conn, frames, block):
    rows = conn.execute(
        "SELECT frame_index, phase_id, decoded FROM samples WHERE block_index=? "
        "AND frame_index BETWEEN ? AND ? ORDER BY frame_index DESC",
        (block, frames[0], frames[-1]),
    ).fetchall()
    if [row[0] for row in rows] != list(reversed(frames)):
        raise ValueError("incomplete history for the requested window")
    decoded = torch.frombuffer(bytearray(b"".join(row[2] for row in rows)), dtype=torch.uint8)
    phases = torch.tensor([int(r[1]) - 1 for r in rows], dtype=torch.long)
    return decoded.float().div_(255.0).view(len(rows), 1, LR_HEIGHT, LR_WIDTH), phases


def inference_inputs(path, frame, block, frames=4):
    """Target-free inputs for the T frames ending at ``frame`` (current frame first)."""
    if frames < 1:
        raise ValueError("frames must be positive")
    with closing(connect(path)) as conn:
        return read_inputs(conn, range(frame - frames + 1, frame + 1), block)
