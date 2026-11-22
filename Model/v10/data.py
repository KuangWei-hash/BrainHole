"""Shared dataset pairing, train-only scaling and target-free inference inputs."""
from contextlib import closing

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from data import RawDataset, DecodedDataset, connect
from .model import make_bilinear
from .diffusion import anchor_image, target_residual


class V10Dataset(Dataset):
    def __init__(self, db, cache, frame_start=0, frame_stop=None):
        self.raw = RawDataset(db, 2, frame_start, frame_stop)
        self.paired = DecodedDataset(self.raw, cache)
        self.keys, self.headers = self.raw.keys, self.raw.headers

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        _, phases, _, target, decoded = self.paired[index]
        return decoded, phases, target, torch.tensor(self.keys[index])


def inference_inputs(db_path, cache, frame, block, header_id):
    """Only reads phase/header metadata and decoded cache; never selects targets."""
    if frame < 1 or not 0 <= block < 30:
        raise ValueError("requires a nonnegative strip index and complete previous frame")
    with closing(connect(db_path)) as db:
        rows = db.execute("SELECT frame_index,phase_id,header_id FROM samples "
                          "WHERE block_index=? AND frame_index BETWEEN ? AND ? ORDER BY frame_index DESC",
                          (block, frame-1, frame)).fetchall()
    if (len(rows) != 2 or [r[0] for r in rows] != [frame, frame-1]
            or any(r[2] != header_id or r[1] != r[0] % 16+1 for r in rows)):
        raise ValueError("incomplete history or incompatible phase/header")
    decoded = torch.stack([cache.get((f, block)) for f, _, _ in rows])[None]
    phases = torch.tensor([[p-1 for _, p, _ in rows]], dtype=torch.long)
    return decoded, make_bilinear(decoded, phases), phases


@torch.inference_mode()
def fit_residual_scale(loader, device, max_batches=0, anchor=None):
    """Caller must pass train split only. Population std, CFA channels, floor .01."""
    total, square, count, seen = torch.zeros(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64), 0, 0
    for index, (decoded, phases, target, _) in enumerate(loader, 1):
        decoded, phases, target = decoded.to(device), phases.to(device), target.to(device)
        bilinear = make_bilinear(decoded, phases)
        base = anchor_image(anchor, decoded, bilinear, phases)
        residual = target_residual(target, base, detail_only=anchor is not None).double()
        total += residual.sum((0, 2, 3)).cpu()
        square += residual.square().sum((0, 2, 3)).cpu()
        count += residual.shape[0]*8*320
        seen += len(decoded)
        if max_batches and index >= max_batches:
            break
    variance = (square/count-(total/count).square()).clamp_min(0)
    return {"method": "population_std_train_only_floor_0.01", "samples": seen,
            "mean": (total/count).tolist(), "std": variance.sqrt().clamp_min(0.01).tolist()}


def stratified_indices(keys, count=256):
    if count < 1:
        raise ValueError("validation sample count must be positive")
    buckets = [[i for i, (f, _) in enumerate(keys) if f % 16 == p] for p in range(16)]
    chosen = []
    for bucket in buckets:
        n = min(len(bucket), count//16)
        if n:
            chosen.extend(bucket[i] for i in torch.linspace(0, len(bucket)-1, n).round().long().tolist())
    already = set(chosen)
    remainder = [i for i in range(len(keys)) if i not in already]
    chosen.extend(remainder[:max(0, min(count, len(keys))-len(chosen))])
    return sorted(chosen)
