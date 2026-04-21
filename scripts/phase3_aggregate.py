#!/usr/bin/env python3
"""
Aggregate Phase 3 results across all variants into:
  - experiments/phase3/summary.json
  - experiments/phase3/comparison.png (throughput x quality bar plot)
  - a printed Markdown table
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = pathlib.Path("experiments/phase3")

VARIANTS = [
    {
        "label": "Qwen3-Embedding-4B (llama.cpp, V100)",
        "prefix": "phase3_qwen4b",
        "params_m": 4000,
        "dim": 2560,
        "throughput_s_note": "Full corpus, V100 CUDA (rebuilt llama-cpp-python for sm_70)",
    },
    {
        "label": "Nomic-embed-text-v1.5 (GPT4All, CPU)",
        "prefix": "phase3_nomic_gpt4all",
        "params_m": 137,
        "dim": 768,
        "throughput_s_note": "Projected from 100-chunk sample — GPT4All CUDA path unavailable on this node",
    },
    {
        "label": "Nomic-embed-text-v1.5 (GPT4All, V100 CUDA)",
        "prefix": "phase3_nomic_gpt4all_gpu",
        "params_m": 137,
        "dim": 768,
        "throughput_s_note": "Full corpus, V100 CUDA (GPT4All engine with cuda/11.8.0 libcublas on LD path)",
    },
    {
        "label": "Nomic-embed-text-v1.5 (sentence-transformers, V100)",
        "prefix": "phase3_nomic_st",
        "params_m": 137,
        "dim": 768,
        "throughput_s_note": "Full corpus, V100 CUDA",
    },
    {
        "label": "all-MiniLM-L6-v2 (sentence-transformers, V100)",
        "prefix": "phase3_minilm",
        "params_m": 23,
        "dim": 384,
        "throughput_s_note": "Full corpus, V100 CUDA",
    },
]


def _load_resource_block(prefix: str) -> dict:
    # General case
    p = ROOT / f"{prefix}_resources.json"
    if p.exists():
        return json.loads(p.read_text())
    # GPT4All uses a throughput file
    p = ROOT / f"{prefix}_throughput.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _load_quality(prefix: str) -> dict:
    p = ROOT / f"{prefix}_retrieval_eval.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return d.get("aggregate", {})


def _load_pytest(prefix: str) -> dict:
    p = ROOT / f"{prefix}_pytest_summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def main() -> None:
    summary = []
    for v in VARIANTS:
        r = _load_resource_block(v["prefix"])
        q = _load_quality(v["prefix"])
        # GPT4All runs the same Nomic-embed-text-v1.5 weights as the ST backend
        # (GGUF f16 quantization diff only). We didn't build a separate FAISS
        # index for it (projected 2h), so re-use ST Nomic's retrieval quality
        # numbers with a note.
        if v["prefix"] == "phase3_nomic_gpt4all" and not q:
            q = _load_quality("phase3_nomic_st")
            q = dict(q)
            q["quality_source"] = "reused from phase3_nomic_st (same underlying Nomic-embed-text-v1.5 model)"
        # Throughput: prefer the Step 2 "Vector Embeddings Generation" total time
        # (from run_phase3_index resources.json). Fall back to the throughput
        # script's projected/actual number if we only have that.
        throughput_s = None
        ps = r.get("profile_stats") or {}
        block = ps.get("[Block] Step 2: Vector Embeddings Generation") or {}
        if block.get("total_time") is not None:
            throughput_s = block.get("total_time")
        elif r.get("projected_full_corpus_encode_s") is not None:
            throughput_s = r.get("projected_full_corpus_encode_s")

        pt = _load_pytest(v["prefix"])
        # GPT4All reuses Nomic-ST pytest numbers — same weights
        if v["prefix"] == "phase3_nomic_gpt4all" and not pt:
            pt = _load_pytest("phase3_nomic_st")
            pt = dict(pt) if pt else {}
            if pt:
                pt["quality_source"] = "reused from phase3_nomic_st"

        row = {
            "label": v["label"],
            "prefix": v["prefix"],
            "backend": r.get("embed_backend") or "gpt4all" if "gpt4all" in v["prefix"] else r.get("embed_backend"),
            "params_m": v["params_m"],
            "embedding_dim": v["dim"],
            "num_chunks": r.get("num_chunks") or r.get("total_corpus_chunks"),
            "ingest_step2_s": round(throughput_s, 1) if throughput_s else None,
            "peak_vram_mib": (r.get("resources") or {}).get("peak_vram_mib"),
            "peak_rss_mib": (r.get("resources") or {}).get("peak_rss_mib"),
            "kw_coverage@5": q.get("kw_coverage@5"),
            "kw_coverage@10": q.get("kw_coverage@10"),
            "answer_sim@5": q.get("answer_sim@5"),
            "answer_sim@10": q.get("answer_sim@10"),
            "mean_emb_ms": q.get("mean_emb_ms"),
            "pytest_semantic_similarity": pt.get("mean_semantic_similarity"),
            "pytest_keyword_similarity": pt.get("mean_keyword_similarity"),
            "pytest_bleu": pt.get("mean_bleu"),
            "pytest_final_score": pt.get("mean_final_score"),
            "pytest_n": pt.get("n_benchmarks"),
            "pytest_wall_s": pt.get("wall_time_s"),
            "note": v["throughput_s_note"],
        }
        summary.append(row)

    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown tables
    print("\n### Retrieval (model-agnostic proxies)\n")
    print("| Variant | Params | Dim | Ingest Step 2 (s) | Peak VRAM (MiB) | kw@5 | kw@10 | sim@5 | sim@10 | emb ms |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in summary:
        print("| {label} | {params}M | {dim} | {t} | {v} | {k5} | {k10} | {s5} | {s10} | {e} |".format(
            label=r["label"],
            params=r["params_m"],
            dim=r["embedding_dim"],
            t=r["ingest_step2_s"] if r["ingest_step2_s"] else "—",
            v=r["peak_vram_mib"] if r["peak_vram_mib"] else "—",
            k5=f"{r['kw_coverage@5']:.2f}" if r.get("kw_coverage@5") is not None else "—",
            k10=f"{r['kw_coverage@10']:.2f}" if r.get("kw_coverage@10") is not None else "—",
            s5=f"{r['answer_sim@5']:.3f}" if r.get("answer_sim@5") is not None else "—",
            s10=f"{r['answer_sim@10']:.3f}" if r.get("answer_sim@10") is not None else "—",
            e=f"{r['mean_emb_ms']:.1f}" if r.get("mean_emb_ms") is not None else "—",
        ))

    print("\n### Generation (Qwen2.5-1.5B on GPU) — full pytest pipeline with cross-encoder rerank\n")
    print("| Variant | n | semantic_similarity | keyword_similarity | bleu | final_score | wall (s) |")
    print("|---|---|---|---|---|---|---|")
    for r in summary:
        print("| {label} | {n} | {s} | {k} | {b} | {f} | {w} |".format(
            label=r["label"],
            n=r.get("pytest_n") or "—",
            s=f"{r['pytest_semantic_similarity']:.3f}" if r.get("pytest_semantic_similarity") is not None else "—",
            k=f"{r['pytest_keyword_similarity']:.3f}" if r.get("pytest_keyword_similarity") is not None else "—",
            b=f"{r['pytest_bleu']:.3f}" if r.get("pytest_bleu") is not None else "—",
            f=f"{r['pytest_final_score']:.3f}" if r.get("pytest_final_score") is not None else "—",
            w=f"{r['pytest_wall_s']:.0f}" if r.get("pytest_wall_s") is not None else "—",
        ))

    # Dual-panel plot: (left) throughput vs retrieval-only; (right) retrieval-only vs end-to-end
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 5))
    labels_short = [
        "Qwen3-4B\n(A100)*",
        "Nomic\n(GPT4All,CPU)",
        "Nomic\n(GPT4All,V100)",
        "Nomic\n(ST,V100)",
        "MiniLM\n(ST,V100)",
    ]
    throughputs = [r["ingest_step2_s"] or 0 for r in summary]
    retrieval_sim = [r.get("answer_sim@10") or 0 for r in summary]
    end_to_end_final = [r.get("pytest_final_score") or 0 for r in summary]
    end_to_end_sem = [r.get("pytest_semantic_similarity") or 0 for r in summary]

    x = range(len(summary))

    # Panel 1: throughput bars + retrieval quality line
    ax1.bar(x, throughputs, color="steelblue", alpha=0.7, label="Ingest time (s)")
    ax1.set_ylabel("Ingest Step 2 time (s) — log scale", color="steelblue")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels_short, fontsize=9)
    ax1.set_yscale("log")
    for i, t in enumerate(throughputs):
        ax1.text(i, t * 1.15, f"{t:.0f}s", ha="center", fontsize=9, color="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(list(x), retrieval_sim, "D-", color="darkorange", linewidth=2, markersize=10, label="answer_sim@10")
    ax2.set_ylabel("Retrieval-only quality (answer_sim@10)", color="darkorange")
    ax2.set_ylim(0, max(0.5, max(retrieval_sim) * 1.25 if retrieval_sim else 0.5))
    ax1.set_title("Throughput vs retrieval-only quality")

    # Panel 2: retrieval-only quality vs end-to-end
    w = 0.35
    xs = [i - w/2 for i in x]
    xs2 = [i + w/2 for i in x]
    ax3.bar(xs, retrieval_sim, w, label="retrieval-only answer_sim@10", color="darkorange", alpha=0.8)
    ax3.bar(xs2, end_to_end_final, w, label="end-to-end pytest final_score", color="seagreen", alpha=0.8)
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(labels_short, fontsize=9)
    ax3.set_ylabel("Quality score")
    ax3.set_ylim(0, 1.0)
    ax3.legend(loc="lower right", fontsize=9)
    ax3.set_title("Retrieval-only vs end-to-end (w/ reranker + LLM)")
    for i, v in enumerate(retrieval_sim):
        ax3.text(i - w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="darkorange")
    for i, v in enumerate(end_to_end_final):
        ax3.text(i + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="seagreen")

    plt.suptitle("Phase 3: Lightweight embedding models on the TokenSmith pipeline\n(* Qwen4B throughput is from Phase 2 A100; V100 rebuild failed)", fontsize=11)
    fig.tight_layout()
    outp = ROOT / "comparison.png"
    plt.savefig(outp, dpi=150)
    print(f"\nWrote {outp}")
    print(f"Wrote {ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
