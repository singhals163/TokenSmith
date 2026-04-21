#!/usr/bin/env python3
"""
Phase 3 retrieval-only evaluator.

Important: the `ideal_retrieved_chunks` IDs in tests/benchmarks.yaml were
annotated against an older chunking schema and no longer point to relevant
chunks under the current chunker. We still report Recall@k / MRR for
transparency, but the primary quality metrics here are model-agnostic and do
not depend on chunk-ID correctness:

    - Keyword coverage: fraction of benchmark['keywords'] that appear in any
      retrieved chunk's text (case-insensitive substring).
    - Answer semantic similarity: max cosine similarity between a retrieved
      chunk and the benchmark['expected_answer'], computed with a neutral
      evaluator (MiniLM). Higher = the retriever surfaced chunks that are
      semantically closer to the gold answer.

Usage:
    python -m scripts.phase3_retrieval_eval \\
        --prefix phase3_qwen4b \\
        --model models/Qwen3-Embedding-4B-Q4_K_M.gguf \\
        --backend llama_cpp
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import time
from statistics import mean

import re

import faiss
import numpy as np
import yaml

from src.embedder import CachedEmbedder


_EVAL_ST = None  # cached evaluator


def _evaluator():
    global _EVAL_ST
    if _EVAL_ST is None:
        from sentence_transformers import SentenceTransformer
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _EVAL_ST = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    return _EVAL_ST


def _strip_all_ws(text: str) -> str:
    # The docling extraction leaves every character separated by a single space,
    # even across word boundaries — "V a r i a b l e l e n g t h" and
    # "records arise" are indistinguishable. Stripping all whitespace is the
    # only robust way to make substring/keyword matching work.
    return re.sub(r"\s+", "", text).lower()


def _collapse_char_spacing(text: str) -> str:
    # Undo the single-char spacing for the semantic evaluator, producing readable
    # (if sometimes word-merged) text. Cleaner than feeding raw char-spaced
    # text into a MiniLM tokenizer that has never seen this distribution.
    return re.sub(r"(?<=\S) (?=\S)", "", text).strip()


def keyword_coverage(retrieved_texts: list[str], keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    blob = " ".join(_strip_all_ws(t) for t in retrieved_texts)
    hits = sum(1 for kw in keywords if _strip_all_ws(kw) in blob)
    return hits / len(keywords)


def answer_sim_max(retrieved_texts: list[str], expected_answer: str) -> float:
    if not retrieved_texts:
        return 0.0
    m = _evaluator()
    texts = [_collapse_char_spacing(t) for t in retrieved_texts]
    va = m.encode([expected_answer], convert_to_numpy=True, normalize_embeddings=True)
    vt = m.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=16)
    sims = (vt @ va.T).reshape(-1)
    return float(sims.max())


def load_benchmarks(path: pathlib.Path) -> list[dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    bms = data["benchmarks"] if isinstance(data, dict) and "benchmarks" in data else data
    # Keep any benchmark that has an expected_answer; keywords may be empty.
    return [b for b in bms if b.get("expected_answer")]


def retrieval_scores(retrieved: list[int], ideal: list[int], k_values: list[int]):
    """For each k in k_values return recall@k. Also return MRR over top max(k)."""
    ideal_set = set(ideal)
    out = {}
    for k in k_values:
        topk = set(retrieved[:k])
        hits = len(topk & ideal_set)
        out[f"recall@{k}"] = hits / max(1, len(ideal_set))
    # MRR: first position where a retrieved chunk is in ideal_set
    rr = 0.0
    for rank, idx in enumerate(retrieved, start=1):
        if idx in ideal_set:
            rr = 1.0 / rank
            break
    out["mrr"] = rr
    out["hit@1"] = 1.0 if retrieved and retrieved[0] in ideal_set else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="Index prefix, e.g. phase3_qwen4b")
    ap.add_argument("--model", required=True, help="Embedding model path / HF name")
    ap.add_argument("--backend", required=True, choices=["llama_cpp", "gpt4all", "sentence_transformers"])
    ap.add_argument("--artifacts-dir", default="index/sections")
    ap.add_argument("--benchmarks", default="tests/benchmarks.yaml")
    ap.add_argument("--pool", type=int, default=50, help="candidates to pull from FAISS")
    ap.add_argument("--out-dir", default="experiments/phase3")
    ap.add_argument("--n-gpu-layers", type=int, default=-1, help="only used for llama_cpp backend; -1=all, 0=CPU-only")
    args = ap.parse_args()

    artifacts_dir = pathlib.Path(args.artifacts_dir)
    faiss_index = faiss.read_index(str(artifacts_dir / f"{args.prefix}.faiss"))
    with open(artifacts_dir / f"{args.prefix}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    bms = load_benchmarks(pathlib.Path(args.benchmarks))
    print(f"Loaded {len(bms)} benchmarks with ideal_retrieved_chunks")

    kwargs = {}
    if args.backend == "llama_cpp":
        kwargs["n_gpu_layers"] = args.n_gpu_layers
    embedder = CachedEmbedder(args.model, backend=args.backend, **kwargs)

    per_q = []
    k_values = [1, 5, 10]
    print(f"\n{'='*96}")
    print(f"{'id':<28} {'kw@5':>6} {'kw@10':>6} {'sim@5':>6} {'sim@10':>7} {'R@5':>6} {'MRR':>6} {'emb_ms':>8}")
    print("-" * 96)

    for b in bms:
        q = b["question"]
        ideal = list(b.get("ideal_retrieved_chunks") or [])
        keywords = list(b.get("keywords") or [])
        expected = b.get("expected_answer", "")
        t0 = time.perf_counter()
        qvec = embedder.encode([q]).astype("float32")
        emb_ms = (time.perf_counter() - t0) * 1000
        if qvec.shape[1] != faiss_index.d:
            raise ValueError(f"dim mismatch: index={faiss_index.d} vs query={qvec.shape[1]}")
        distances, indices = faiss_index.search(qvec, args.pool)
        retrieved = [int(i) for i in indices[0] if 0 <= i < len(chunks)]
        s = retrieval_scores(retrieved, ideal, k_values) if ideal else {f"recall@{k}": 0.0 for k in k_values} | {"mrr": 0.0, "hit@1": 0.0}

        top5 = [chunks[i] for i in retrieved[:5]]
        top10 = [chunks[i] for i in retrieved[:10]]
        s["kw@5"] = keyword_coverage(top5, keywords)
        s["kw@10"] = keyword_coverage(top10, keywords)
        s["sim@5"] = answer_sim_max(top5, expected)
        s["sim@10"] = answer_sim_max(top10, expected)
        s["top_ids"] = retrieved[:10]
        s.update({"id": b.get("id", "?"), "emb_ms": emb_ms, "ideal_n": len(ideal), "keywords_n": len(keywords)})
        per_q.append(s)
        print(
            f"{s['id']:<28} {s['kw@5']:>6.2f} {s['kw@10']:>6.2f} {s['sim@5']:>6.3f} {s['sim@10']:>7.3f} "
            f"{s['recall@5']:>6.2f} {s['mrr']:>6.3f} {emb_ms:>8.1f}"
        )

    agg = {
        "prefix": args.prefix,
        "backend": args.backend,
        "model": args.model,
        "n_queries": len(per_q),
        "index_size": faiss_index.ntotal,
        "index_dim": faiss_index.d,
        "kw_coverage@5": mean(p["kw@5"] for p in per_q),
        "kw_coverage@10": mean(p["kw@10"] for p in per_q),
        "answer_sim@5": mean(p["sim@5"] for p in per_q),
        "answer_sim@10": mean(p["sim@10"] for p in per_q),
        "recall@5": mean(p["recall@5"] for p in per_q),
        "recall@10": mean(p["recall@10"] for p in per_q),
        "mrr": mean(p["mrr"] for p in per_q),
        "mean_emb_ms": mean(p["emb_ms"] for p in per_q),
    }
    print("-" * 96)
    print(
        f"{'AGGREGATE':<28} {agg['kw_coverage@5']:>6.2f} {agg['kw_coverage@10']:>6.2f} "
        f"{agg['answer_sim@5']:>6.3f} {agg['answer_sim@10']:>7.3f} "
        f"{agg['recall@5']:>6.2f} {agg['mrr']:>6.3f} {agg['mean_emb_ms']:>8.1f}"
    )
    print("=" * 96)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.prefix}_retrieval_eval.json", "w") as f:
        json.dump({"aggregate": agg, "per_query": per_q}, f, indent=2)
    print(f"Wrote {out_dir / (args.prefix + '_retrieval_eval.json')}")


if __name__ == "__main__":
    main()
