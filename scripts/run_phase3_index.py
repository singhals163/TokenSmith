#!/usr/bin/env python3
"""
Phase 3: build one FAISS index per embedding backend+model variant.

Usage:
    python -m scripts.run_phase3_index --variant qwen4b
    python -m scripts.run_phase3_index --variant nomic_gpt4all
    python -m scripts.run_phase3_index --variant nomic_st
    python -m scripts.run_phase3_index --variant minilm

Each run writes FAISS artifacts to index/sections/{prefix}.* and a per-run
profiling + resources JSON to experiments/phase3/.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from src.config import RAGConfig
from src.index_builder import build_index
from src.preprocessing.chunking import DocumentChunker
from src.profiler import PROFILE_STATS


VARIANTS = {
    "qwen4b": {
        "model_path": "models/Qwen3-Embedding-4B-Q4_K_M.gguf",
        "backend": "llama_cpp",
        "prefix": "phase3_qwen4b",
    },
    "nomic_gpt4all": {
        # gpt4all will download this GGUF on first use into $GPT4ALL_MODEL_PATH
        "model_path": "nomic-embed-text-v1.5.f16.gguf",
        "backend": "gpt4all",
        "prefix": "phase3_nomic_gpt4all",
    },
    "nomic_gpt4all_gpu": {
        "model_path": "nomic-embed-text-v1.5.f16.gguf",
        "backend": "gpt4all",
        "prefix": "phase3_nomic_gpt4all_gpu",
        "embedder_kwargs": {"device": "kompute"},
    },
    "nomic_st": {
        "model_path": "nomic-ai/nomic-embed-text-v1.5",
        "backend": "sentence_transformers",
        "prefix": "phase3_nomic_st",
    },
    "minilm": {
        "model_path": "sentence-transformers/all-MiniLM-L6-v2",
        "backend": "sentence_transformers",
        "prefix": "phase3_minilm",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    parser.add_argument("--markdown", default="data/textbook--extracted_markdown.md")
    parser.add_argument("--profile-dir", default="experiments/phase3")
    args = parser.parse_args()

    variant = VARIANTS[args.variant]

    # Reset profile stats so each variant's file is clean
    PROFILE_STATS.clear()

    cfg = RAGConfig.from_yaml(pathlib.Path("config/config.yaml"))
    chunker = DocumentChunker(strategy=cfg.get_chunk_strategy(), keep_tables=False)
    artifacts_dir = cfg.get_artifacts_directory()

    md_path = pathlib.Path(args.markdown)
    if not md_path.exists():
        print(f"ERROR: markdown {md_path} not found", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(f"Phase 3 variant : {args.variant}")
    print(f"Backend         : {variant['backend']}")
    print(f"Model           : {variant['model_path']}")
    print(f"Index prefix    : {variant['prefix']}")
    print(f"Markdown        : {md_path}")
    print("=" * 70)

    build_index(
        markdown_file=str(md_path),
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=variant["model_path"],
        artifacts_dir=artifacts_dir,
        index_prefix=variant["prefix"],
        use_multiprocessing=False,
        use_headings=False,
        embed_backend=variant["backend"],
        profile_output_dir=args.profile_dir,
        embedder_kwargs=variant.get("embedder_kwargs"),
    )


if __name__ == "__main__":
    main()
