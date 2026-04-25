#!/usr/bin/env python3
"""
Phase 3 GPT4All throughput measurement.

Running GPT4All Nomic on the full 2,664-chunk corpus on CPU projects to ~2.6h,
which exceeds our GPU time window. Instead, we:
  1. Measure throughput on a sampled subset of N chunks
  2. Also measure cold model-load time
  3. Project the full-corpus time
  4. Write a resources JSON so the reporting script treats it like any other run

For quality, we reuse the sentence-transformers Nomic FAISS index since both
backends use the same underlying Nomic-embed-text-v1.5 model (with negligible
quantization-level differences).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import random
import time

from src.instrumentation.resource_monitor import ResourceMonitor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="phase3_nomic_gpt4all")
    ap.add_argument("--source-index", default="index/sections/phase3_nomic_st_chunks.pkl",
                    help="Reuse the ST Nomic chunks — same text, same model")
    ap.add_argument("--sample-size", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="experiments/phase3")
    args = ap.parse_args()

    with open(args.source_index, "rb") as f:
        chunks = pickle.load(f)
    total_chunks = len(chunks)
    random.seed(args.seed)
    idxs = sorted(random.sample(range(total_chunks), args.sample_size))
    sample = [chunks[i] for i in idxs]

    from gpt4all import Embed4All

    monitor = ResourceMonitor(interval_s=0.5)
    with monitor:
        t_load0 = time.perf_counter()
        model = Embed4All(
            model_name="nomic-embed-text-v1.5.f16.gguf",
            model_path=os.environ["GPT4ALL_MODEL_PATH"],
        )
        load_s = time.perf_counter() - t_load0
        t_enc0 = time.perf_counter()
        per_batch_times = []
        for i in range(0, len(sample), args.batch_size):
            batch = sample[i : i + args.batch_size]
            b0 = time.perf_counter()
            out = model.embed(batch)
            per_batch_times.append(time.perf_counter() - b0)
        enc_s = time.perf_counter() - t_enc0

    per_chunk_s = enc_s / len(sample)
    projected_full_s = per_chunk_s * total_chunks

    result = {
        "prefix": args.prefix,
        "embed_backend": "gpt4all",
        "embedding_model_path": "nomic-embed-text-v1.5.f16.gguf",
        "sample_size": args.sample_size,
        "total_corpus_chunks": total_chunks,
        "batch_size": args.batch_size,
        "model_load_s": round(load_s, 3),
        "sample_encode_s": round(enc_s, 3),
        "per_chunk_s": round(per_chunk_s, 4),
        "projected_full_corpus_encode_s": round(projected_full_s, 1),
        "per_batch_s_mean": round(sum(per_batch_times) / len(per_batch_times), 3),
        "per_batch_s_min": round(min(per_batch_times), 3),
        "per_batch_s_max": round(max(per_batch_times), 3),
        "resources": monitor.summary(),
    }
    print(json.dumps(result, indent=2))

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.prefix}_throughput.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_dir / (args.prefix + '_throughput.json')}")


if __name__ == "__main__":
    main()
