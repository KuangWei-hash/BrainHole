"""Compact v14 codebook, but matched on the real JPEG XS encoded payload bytes
instead of a pixel-derived descriptor. Each strip is still split into width-wide
tiles whose value is a quantized DCT residual (same math as
train_v14_compact_codebook.py / train_v14_demo_compact_codebook.py); only the
KEY used to pick a candidate changes: raw encoded bytes (padded to a fixed
length), compared by L1 distance, instead of the 16-dim mean/gradient descriptor.

k candidates per (phase, position) are a uniform subsample of the training
range's same-phase frames (no averaging/clustering: byte keys aren't a
meaningful quantity to average, so each candidate is one real observed frame).
"""
import argparse
from contextlib import closing
from pathlib import Path

import torch
from torch.nn import functional as F

from evaluate import demosaic
from evaluate_v10 import SingleVideo
from evaluate_v13 import verify_video
from evaluate_v14_current_phase import phase_ids, sample
from evaluate_v14_strips import manifest
from record_experiment import digest, run_logged, write_json
import v14 as compact
from v11.data import connect
from v13.data import atomic_save, labels


TRAIN_RANGE = (0, 1389)
TEST_RANGE = (1748, 1852)
K = 8
WIDTH = 40
RANK = 160


def load_range_with_payload(db, first, last):
    phases = phase_ids(db, first, last)
    raw = labels(db, manifest(first, last)).reshape(-1, 30, 16, 640)
    with closing(connect(db)) as conn:
        rows = conn.execute("SELECT frame_index, block_index, jpegxs_payload FROM samples "
                            "WHERE frame_index BETWEEN ? AND ?", (first, last)).fetchall()
    frames = raw.shape[0]
    payloads = [[None] * frames for _ in range(30)]
    for frame, block, payload in rows:
        payloads[block][frame - first] = payload
    if any(p is None for column in payloads for p in column):
        raise ValueError("missing jpegxs_payload for some strip")
    return raw, torch.tensor(phases), payloads


def pad_keys(payload_list, maxlen):
    out = torch.zeros((len(payload_list), maxlen), dtype=torch.uint8)
    for i, payload in enumerate(payload_list):
        out[i, :len(payload)] = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    return out


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True)
    compact.RANK = RANK
    tiles = compact.tiles_for(WIDTH)
    basis = compact.dct_basis(WIDTH)

    train_raw, train_phases, train_payloads = load_range_with_payload(args.db, *TRAIN_RANGE)
    test_raw, test_phases, test_payloads = load_range_with_payload(args.db, *TEST_RANGE)
    maxlen = max(len(p) for column in (train_payloads + test_payloads) for p in column)
    print(f"max JPEG XS payload length observed: {maxlen} bytes", flush=True)

    from evaluate_v14_current_phase import sample as sample_fn
    test_samples = torch.stack([sample_fn(frame, int(phase)) for frame, phase in zip(test_raw, test_phases)])
    test_flat_samples = test_samples.flatten(0, 1)
    test_flat_phases = test_phases.repeat_interleave(30)
    base = compact.bilinear(test_flat_samples, test_flat_phases).reshape(-1, 1, 16, 640)
    truth = test_raw.reshape(-1, 1, 16, 640)
    baseline_mae = (base.short() - truth.short()).abs().double().mean().item()
    print(f"bilinear baseline RAW MAE: {baseline_mae:.4f}", flush=True)

    # Per (phase, position): pick K candidate training frames and precompute their
    # per-tile DCT residual codes plus padded encoded-payload keys.
    candidate_keys = [[None] * 30 for _ in range(16)]     # [phase][position] -> (K, maxlen) uint8
    candidate_codes = [[None] * 30 for _ in range(16)]    # [phase][position] -> (K, tiles, 4*RANK) float
    key_bytes_total = 0
    for phase in range(16):
        present = (train_phases == phase).nonzero().flatten()
        if len(present) < K:
            raise ValueError(f"not enough phase-{phase} frames in the train range")
        chosen = present[torch.linspace(0, len(present) - 1, K).round().long()]
        subset_raw = train_raw[chosen]
        _, residuals = compact.phase_data(subset_raw, sample(subset_raw, phase), phase, basis, WIDTH, tiles)  # (30*tiles, K, 4*RANK)
        residuals = residuals.reshape(30, tiles, K, 4 * compact.RANK)
        for position in range(30):
            payloads = [train_payloads[position][i] for i in chosen.tolist()]
            candidate_keys[phase][position] = pad_keys(payloads, maxlen)
            candidate_codes[phase][position] = residuals[position].transpose(0, 1).contiguous()  # (K, tiles, 4*RANK)
            key_bytes_total += sum(len(p) for p in payloads)

    # Query: nearest candidate by L1 byte distance within the matching phase.
    patches = torch.zeros(len(test_flat_phases), tiles, 16, WIDTH)
    distance = torch.zeros(len(test_flat_phases), tiles)
    for i in range(len(test_flat_phases)):
        phase = int(test_phases[i // 30])
        position = i % 30
        query_key = pad_keys([test_payloads[position][i // 30]], maxlen)[0].short()
        keys = candidate_keys[phase][position].short()
        codes = candidate_codes[phase][position]
        d = (keys - query_key).abs().float().mean(dim=-1)  # (K,) mean-per-byte, same 0-255 scale as pixel MAE
        best = d.argmin()
        decoded = (codes[best].float().reshape(tiles, 4, compact.RANK) @ basis).reshape(tiles, 4, 8, WIDTH // 2)
        patch = F.pixel_shuffle(decoded, 2).reshape(tiles, 16, WIDTH)
        patches[i] = patch
        distance[i] = d[best].float()

    print(f"byte-key distance (mean-per-byte over {maxlen} bytes): "
          f"min={distance.min():.3f} mean={distance.mean():.3f} max={distance.max():.3f}", flush=True)
    best_gate, best_mae = None, float("inf")
    for limit, strength in compact.GATES:
        pred = compact.reconstruct(base.reshape(-1, 1, 16, 640), patches, distance, test_flat_phases, limit, strength)
        mae = (pred.short() - truth.short()).abs().double().mean().item()
        if mae < best_mae:
            best_mae, best_gate = mae, {"limit": limit, "strength": strength}
    print(f"encoded-key compact codebook: RAW MAE {best_mae:.4f} (gate {best_gate}), "
          f"vs bilinear {baseline_mae:.4f} ({(1 - best_mae / baseline_mae) * 100:+.1f}%)", flush=True)

    book = {"candidate_keys": candidate_keys, "candidate_codes": candidate_codes, "maxlen": maxlen,
            "gate": best_gate, "config": {"architecture": "v14-demo-encoded-key-compact", "patch_width": WIDTH,
            "dct_rank": RANK, "entries_per_tile": K, "train_range": TRAIN_RANGE, "test_range": TEST_RANGE}}
    path = args.out / "codebook.pt"
    atomic_save(book, path)
    print(f"codebook saved to {path}: {path.stat().st_size/1e6:.3f} MB on disk "
          f"(key bytes alone: {key_bytes_total/1e6:.3f} MB)", flush=True)

    pred = compact.reconstruct(base.reshape(-1, 1, 16, 640), patches, distance, test_flat_phases,
                               **best_gate).reshape_as(truth)
    names = ("target", "bilinear", "encoded-key-compact")
    videos = {name: SingleVideo(args.out, name) for name in names}
    for name, value in zip(names, (truth, base, pred)):
        rgb = demosaic(value.float() / 255).reshape(-1, 30, 3, 16, 640)
        for frame in rgb:
            videos[name].write(dict(enumerate(frame)))
    for video in videos.values():
        video.close()
    for name in names:
        verify_video(args.out / f"{name}.mp4", len(test_raw))

    write_json(args.out / "report.json", {"baseline_raw_mae": baseline_mae, "raw_mae": best_mae, "gate": best_gate,
                                          "codebook_bytes": path.stat().st_size, "key_bytes_total": key_bytes_total,
                                          "max_payload_length": maxlen, "db_sha256": digest(args.db)})


if __name__ == "__main__":
    run_logged("training", main)
