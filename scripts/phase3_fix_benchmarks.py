#!/usr/bin/env python3
"""
Regenerate the ideal_retrieved_chunks field in tests/benchmarks.yaml.

The stored chunk IDs were calibrated against an older chunker and now point
at irrelevant chunks (README Finding #5 / HANDOFF Open Item #1). For each
benchmark we pick the top-5 chunks whose text is most semantically similar
to the benchmark's expected_answer, measured with MiniLM-L6.

Usage:
    source .phase3_env.sh
    python -m scripts.phase3_fix_benchmarks                  # dry-run (prints)
    python -m scripts.phase3_fix_benchmarks --write          # edit yaml in-place
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import re

import numpy as np
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHUNKS_PKL = ROOT / "index" / "sections" / "phase3_qwen4b_chunks.pkl"
BENCHMARKS_YAML = ROOT / "tests" / "benchmarks.yaml"
TOP_K = 5


def _clean_chunk(text: str) -> str:
    """docling output has per-character spacing — collapse to normal whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="Edit tests/benchmarks.yaml in place (default: dry-run).")
    args = ap.parse_args()

    with open(CHUNKS_PKL, "rb") as f:
        chunks = pickle.load(f)
    assert isinstance(chunks, list), f"unexpected chunks type: {type(chunks)}"
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PKL}")

    cleaned = [_clean_chunk(c) for c in chunks]

    bench = yaml.safe_load(BENCHMARKS_YAML.read_text())["benchmarks"]
    print(f"Loaded {len(bench)} benchmarks from {BENCHMARKS_YAML}")

    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    chunk_emb = model.encode(cleaned, batch_size=64, convert_to_tensor=True,
                             show_progress_bar=True)

    updates: list[tuple[str, list[int]]] = []
    for b in bench:
        bid = b["id"]
        ans_emb = model.encode([b["expected_answer"]], convert_to_tensor=True)
        sims = util.cos_sim(ans_emb, chunk_emb)[0]
        top = sims.topk(TOP_K)
        ids = top.indices.cpu().tolist()
        scores = top.values.cpu().tolist()
        old = b.get("ideal_retrieved_chunks", [])
        updates.append((bid, ids))
        print(f"\n[{bid}]")
        print(f"  old: {old}")
        print(f"  new: {ids}  (sims: {[round(s, 3) for s in scores]})")

    if not args.write:
        print("\n(dry-run — pass --write to edit tests/benchmarks.yaml)")
        return

    text = BENCHMARKS_YAML.read_text()
    for bid, ids in updates:
        id_block_re = re.compile(
            rf'(- id: "{re.escape(bid)}".*?ideal_retrieved_chunks:\s*)\[[^\]]*\]',
            re.DOTALL,
        )
        new_list = "[" + ", ".join(str(i) for i in ids) + "]"
        new_text, n = id_block_re.subn(lambda m: m.group(1) + new_list, text, count=1)
        if n != 1:
            raise RuntimeError(f"Failed to update {bid} (matched {n} times)")
        text = new_text
    BENCHMARKS_YAML.write_text(text)
    print(f"\nWrote {BENCHMARKS_YAML}")


if __name__ == "__main__":
    main()
