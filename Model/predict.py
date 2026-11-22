"""Reconstruct one SQLite block using a trained checkpoint."""
import argparse
from pathlib import Path

import torch
from data import RawDataset, decoded_window
from network import build_model
from train import preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--block", required=True, type=int)
    parser.add_argument("--decoded-cache", type=Path, help="optional predecoded inputs; otherwise decode payload now")
    parser.add_argument("--decoder", type=Path, default=Path(__file__).resolve().parents[1] /
                        "training-data-preparation/tools/jpegxs-bridge/target/release/brainhole-jpegxs.exe")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "runs" / "prediction")
    args = parser.parse_args()
    if any(args.out.with_suffix(suffix).exists() for suffix in (".raw", ".png")):
        parser.error("output already exists; choose a new --out prefix")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    dataset = RawDataset(args.db, config["history"], v9_upper_strip=config.get("v9_upper_strip", False))
    if dataset.headers != {config["header_id"]}:
        raise ValueError("codec/header configuration differs from training")
    try:
        index = dataset.keys.index((args.frame, args.block))
    except ValueError:
        parser.error("block not present or history incomplete")
    model = build_model(config)
    model.load_state_dict(saved["model"])
    model.eval()
    tokens, phases, lengths, target = dataset[index]
    decoded = None
    if getattr(model, "requires_decoded", False):
        from baseline import DecodedCache, decode_keys
        keys = dataset.input_keys(args.frame, args.block)
        if args.decoded_cache:
            cache = DecodedCache(args.decoded_cache, args.db)
            decoded = decoded_window(cache, keys)
        else:
            present = [key for key in keys if key is not None]
            values = iter(decode_keys(args.db, present, args.decoder))
            decoded = torch.stack([next(values) if key is not None else torch.zeros(1, 4, 160) for key in keys])
    with torch.inference_mode():
        if decoded is not None:
            result = model(tokens[None], phases[None], lengths[None], decoded[None])
        else:
            result = model(tokens[None], phases[None], lengths[None])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".raw").write_bytes((result[0, 0]*255).round().byte().numpy().tobytes())
    preview(args.out.with_suffix(".png"), result, target[None])
    print(f"RAW: {args.out.with_suffix('.raw')} (640x16 RGGB); PNG top=target, bottom=prediction")


if __name__ == "__main__":
    from record_experiment import run_logged
    run_logged("inference", main, Path(__file__).parent/"runs"/"prediction")
