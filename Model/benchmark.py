"""Batch=1 model-call latency on prepared tensors; excludes decoder and I/O."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from baseline import DecodedCache
from data import RawDataset
from network import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--decoded-cache", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, nargs="+", type=Path)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1:
        parser.error("iterations and warmup must be positive")
    if args.out.exists():
        parser.error("output already exists")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = DecodedCache(args.decoded_cache, args.db)
    results = {}
    with torch.inference_mode():
        for path in args.checkpoint:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            config = saved["config"]
            if config["header_id"] != cache.metadata["header_id"]:
                raise ValueError("checkpoint/header mismatch")
            dataset = RawDataset(args.db, config["history"])
            frame, block = dataset.keys[len(dataset)//2]
            tokens, phase, lengths, _ = dataset[len(dataset)//2]
            low = torch.stack([cache.get((frame-k, block)) for k in range(config["history"])])
            inputs = [value[None].to(device) for value in (tokens, phase, lengths, low)]
            model = build_model(config).to(device).eval()
            model.load_state_dict(saved["model"])
            if not getattr(model, "requires_decoded", False):
                inputs = inputs[:3]
            for _ in range(args.warmup):
                model(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                model(*inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
            results[path.parent.name] = {"mean_ms": float(np.mean(times)),
                                         "p50_ms": float(np.percentile(times, 50)),
                                         "p95_ms": float(np.percentile(times, 95)),
                                         "history": config["history"],
                                         "parameters": sum(p.numel() for p in model.parameters())}
            if dataset.connection is not None:
                dataset.connection.close()
            del model
    report = {"device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
              "torch": str(torch.__version__), "batch_size": 1, "iterations": args.iterations,
              "warmup": args.warmup, "results": results,
              "scope": "Prepared device tensors, model forward plus synchronization only; "
                       "excludes JPEG XS decoding, data transfer, SQLite I/O, queueing and streaming scheduling."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    from record_experiment import run_logged
    run_logged("benchmark", main)
