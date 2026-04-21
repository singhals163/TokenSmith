#!/usr/bin/env python3
"""
Phase 3 full pytest harness runner — runs tests/test_benchmarks.py once per
embedding variant with the generator pipeline (Qwen2.5-1.5B-Instruct), and
aggregates the metric scores.

For each variant:
  1. Write a temporary config yaml
  2. Invoke pytest tests/test_benchmarks.py with that config
  3. Copy tests/results/benchmark_results.json into experiments/phase3/{prefix}_pytest.json
  4. Tally final_score, semantic_similarity, keyword_similarity

Generator runs on GPU via llama-cpp-python (n_gpu_layers=-1). Pass
--cpu-only to disable this (sets CUDA_VISIBLE_DEVICES="" for the subprocess).
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from statistics import mean

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "tests" / "results"
OUT_DIR = ROOT / "experiments" / "phase3"


VARIANTS = [
    {
        "name": "qwen4b",
        "prefix": "phase3_qwen4b",
        "embed_model": "models/Qwen3-Embedding-4B-Q4_K_M.gguf",
        "embed_backend": "llama_cpp",
    },
    {
        "name": "nomic_st",
        "prefix": "phase3_nomic_st",
        "embed_model": "nomic-ai/nomic-embed-text-v1.5",
        "embed_backend": "sentence_transformers",
    },
    {
        "name": "minilm",
        "prefix": "phase3_minilm",
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embed_backend": "sentence_transformers",
    },
    {
        # Phase-1-v2: same Qwen4B embedder, but against the pypdfium-extracted
        # markdown instead of docling. Measures accuracy cost of the ~156x
        # faster extraction path.
        "name": "qwen4b_pypdfium",
        "prefix": "phase1v2_pypdfium_qwen4b",
        "embed_model": "models/Qwen3-Embedding-4B-Q4_K_M.gguf",
        "embed_backend": "llama_cpp",
    },
]


GEN_MODEL = "models/qwen2.5-1.5b-instruct-q5_k_m.gguf"


def build_config_yaml(variant: dict) -> pathlib.Path:
    cfg = {
        "embed_model": variant["embed_model"],
        "embed_backend": variant["embed_backend"],
        "model_path": GEN_MODEL,
        "gen_model": GEN_MODEL,
        "top_k": 10,
        "num_candidates": 50,
        "ensemble_method": "rrf",
        "ranker_weights": {"faiss": 1, "bm25": 0, "index_keywords": 0},
        "rrf_k": 60,
        "max_gen_tokens": 300,
        "chunk_mode": "recursive_sections",
        "chunk_size": 2000,
        "chunk_overlap": 200,
        "use_hyde": False,
        "hyde_max_tokens": 300,
        "use_indexed_chunks": False,
        # Use the production cross-encoder reranker so the prompt fits the 4k
        # context window. All variants hit the same reranker so the comparison
        # still isolates the embedding model.
        "rerank_mode": "cross_encoder",
        "rerank_top_k": 5,
        "use_double_prompt": False,
        "enable_history": False,
        "max_history_turns": 3,
        "index_prefix": variant["prefix"],
        "output_mode": "terminal",
        "metrics": ["semantic", "keyword", "bleu"],
        "system_prompt_mode": "baseline",
    }
    path = OUT_DIR / f"{variant['prefix']}_pytest_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg))
    return path


def run_variant(variant: dict, cpu_only: bool = False) -> dict:
    cfg_path = build_config_yaml(variant)
    log_path = ROOT / "logs" / f"phase3_pytest_{variant['name']}.log"
    log_path.parent.mkdir(exist_ok=True)

    rf = RESULTS_DIR / "benchmark_results.json"
    if rf.exists():
        rf.unlink()

    env = os.environ.copy()
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_benchmarks.py",
        f"--config={cfg_path}",
        f"--index-prefix={variant['prefix']}",
        f"--model-path={GEN_MODEL}",
        f"--embed-model={variant['embed_model']}",
        "--output-mode=terminal",
        "--artifacts_dir=index/sections",
        "-s",
        "-q",
    ]
    print(f"[{variant['name']}] launching: {' '.join(cmd)}")
    t0 = time.perf_counter()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    wall_s = time.perf_counter() - t0
    print(f"[{variant['name']}] pytest exit={proc.returncode} wall={wall_s:.1f}s log={log_path}")

    results = []
    if rf.exists():
        with open(rf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        dst = OUT_DIR / f"{variant['prefix']}_pytest_results.jsonl"
        shutil.copy2(rf, dst)

    # Aggregate scores
    def _safe_mean(xs):
        xs = [x for x in xs if x is not None]
        return mean(xs) if xs else None

    sem = [r.get("scores", {}).get("semantic_similarity") for r in results]
    kw = [r.get("scores", {}).get("keyword_similarity") for r in results]
    bleu = [r.get("scores", {}).get("bleu_similarity") for r in results]
    final = [r.get("scores", {}).get("final_score") for r in results]

    summary = {
        "variant": variant["name"],
        "prefix": variant["prefix"],
        "embed_model": variant["embed_model"],
        "embed_backend": variant["embed_backend"],
        "gen_model": GEN_MODEL,
        "pytest_exit": proc.returncode,
        "wall_time_s": round(wall_s, 1),
        "n_benchmarks": len(results),
        "mean_semantic_similarity": _safe_mean(sem),
        "mean_keyword_similarity": _safe_mean(kw),
        "mean_bleu": _safe_mean(bleu),
        "mean_final_score": _safe_mean(final),
        "per_benchmark": [
            {
                "id": r.get("test_id"),
                "semantic_similarity": r.get("scores", {}).get("semantic_similarity"),
                "keyword_similarity": r.get("scores", {}).get("keyword_similarity"),
                "bleu_similarity": r.get("scores", {}).get("bleu_similarity"),
                "final_score": r.get("scores", {}).get("final_score"),
                "passed": r.get("passed"),
            }
            for r in results
        ],
    }
    out_summary = OUT_DIR / f"{variant['prefix']}_pytest_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="Comma-separated variant names to run (e.g. qwen4b)")
    ap.add_argument("--cpu-only", action="store_true", help="Force CPU for all llama.cpp models")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for variant in VARIANTS:
        if only and variant["name"] not in only:
            continue
        try:
            s = run_variant(variant, cpu_only=args.cpu_only)
            all_summaries.append(s)
        except Exception as e:
            print(f"[{variant['name']}] FAILED: {e}")
            all_summaries.append({"variant": variant["name"], "error": str(e)})

    # Combined summary — merge with existing rather than overwrite, so
    # --only runs don't lose earlier variants.
    combined = OUT_DIR / "pytest_all_summary.json"
    existing = []
    if combined.exists():
        try:
            existing = json.loads(combined.read_text())
        except Exception:
            existing = []
    by_name = {s.get("variant"): s for s in existing}
    for s in all_summaries:
        by_name[s.get("variant")] = s
    combined.write_text(json.dumps(list(by_name.values()), indent=2))

    print("\n" + "=" * 96)
    print(f"{'variant':<12} {'backend':<22} {'n':>3} {'sem_sim':>8} {'kw_sim':>8} {'final':>8} {'wall_s':>8}")
    print("-" * 96)
    for s in all_summaries:
        if "error" in s:
            print(f"{s['variant']:<12} ERROR: {s['error']}")
            continue
        print(
            f"{s['variant']:<12} {s['embed_backend']:<22} {s['n_benchmarks']:>3} "
            f"{(s['mean_semantic_similarity'] or 0):>8.3f} "
            f"{(s['mean_keyword_similarity'] or 0):>8.3f} "
            f"{(s['mean_final_score'] or 0):>8.3f} {s['wall_time_s']:>8.1f}"
        )
    print("=" * 96)
    print(f"Wrote {combined}")


if __name__ == "__main__":
    main()
