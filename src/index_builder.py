#!/usr/bin/env python3
"""
index_builder.py
PDF -> markdown text -> chunks -> embeddings -> BM25 + FAISS + metadata

Entry point (called by main.py):
    build_index(markdown_file, cfg, keep_tables=True, do_visualize=False)
"""

import os
import pickle
import pathlib
import re
import json
from typing import List, Dict, Optional
import subprocess

import faiss
from rank_bm25 import BM25Okapi
from src.embedder import CachedEmbedder

from src.preprocessing.chunking import DocumentChunker, ChunkConfig
from src.preprocessing.extraction import extract_sections_from_markdown

# --- NEW: Import profiler tools ---
from src.profiler import timeit, TimerBlock, print_profile_stats, PROFILE_STATS
from src.instrumentation.resource_monitor import ResourceMonitor

# ----- runtime parallelism knobs (avoid oversubscription) -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

DEFAULT_EXCLUSION_KEYWORDS = ['questions', 'exercises', 'summary', 'references']

# --- NEW: Phase 2 Hardware Detection ---
def detect_gpu() -> bool:
    """Dynamically checks if a CUDA (NVIDIA) or Metal (Apple) GPU is available."""
    try:
        # Check for NVIDIA CUDA
        subprocess.check_output('nvidia-smi', shell=True, stderr=subprocess.STDOUT)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check for Apple Silicon (Metal)
    import platform
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return True
        
    return False

# ------------------------ Main index builder -----------------------------

def build_index(
    markdown_file: str,
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    use_multiprocessing: bool = False,
    use_headings: bool = False,
    embed_backend: str = "llama_cpp",
    profile_output_dir: Optional[os.PathLike] = None,
) -> None:
    """
    Extract sections, chunk, embed, and build both FAISS and BM25 indexes.

    Persists:
        - {prefix}.faiss
        - {prefix}_bm25.pkl
        - {prefix}_chunks.pkl
        - {prefix}_sources.pkl
        - {prefix}_meta.pkl
    """
    all_chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict] = []

    with TimerBlock("[Block] Extract Sections from Markdown"):
        sections = extract_sections_from_markdown(
            markdown_file,
            exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS
        )

    page_to_chunk_ids = {}
    current_page = 1
    total_chunks = 0
    heading_stack = []

    with TimerBlock("[Block] Step 1: Chunking & Metadata Parsing"):
        for i, c in enumerate(sections):
            current_level = c.get('level', 1)
            chapter_num = c.get('chapter', 0)

            while heading_stack and heading_stack[-1][0] >= current_level:
                heading_stack.pop()
            
            if c['heading'] != "Introduction":
                heading_stack.append((current_level, c['heading']))

            path_list = [h[1] for h in heading_stack]
            full_section_path = " ".join(path_list)
            full_section_path = f"Chapter {chapter_num} " + full_section_path

            sub_chunks = chunker.chunk(c['content'])
            page_pattern = re.compile(r'--- Page (\d+) ---')

            for sub_chunk_id, sub_chunk in enumerate(sub_chunks):
                chunk_pages = set()
                fragments = page_pattern.split(sub_chunk)

                if fragments[0].strip():
                    page_to_chunk_ids.setdefault(current_page, set()).add(total_chunks+sub_chunk_id)
                    chunk_pages.add(current_page)

                for i in range(1, len(fragments), 2):
                    try:
                        new_page = int(fragments[i]) + 1
                        if fragments[i+1].strip():
                            page_to_chunk_ids.setdefault(new_page, set()).add(total_chunks + sub_chunk_id)
                            chunk_pages.add(new_page)
                        current_page = new_page
                    except (IndexError, ValueError):
                        continue

                clean_chunk = re.sub(page_pattern, '', sub_chunk).strip()
                
                if c["heading"] == "Introduction":
                    continue
                
                meta = {
                    "filename": markdown_file,
                    "mode": chunk_config.to_string(),
                    "char_len": len(clean_chunk),
                    "word_len": len(clean_chunk.split()),
                    "section": c['heading'],
                    "section_path": full_section_path,
                    "text_preview": clean_chunk[:100],
                    "page_numbers": sorted(list(chunk_pages)),
                    "chunk_id": total_chunks + sub_chunk_id
                }

                if use_headings:
                    chunk_prefix = f"Description: {full_section_path} Content: "
                else:
                    chunk_prefix = ""

                all_chunks.append(chunk_prefix+clean_chunk)
                sources.append(markdown_file)
                metadata.append(meta)

            total_chunks += len(sub_chunks)

        final_map = {}
        for page, id_set in page_to_chunk_ids.items():
            final_map[page] = sorted(list(id_set))

        output_file = artifacts_dir / f"{index_prefix}_page_to_chunk_map.json"
        with open(output_file, "w") as f:
            json.dump(final_map, f, indent=2)
        print(f"Saved page to chunk ID map: {output_file}")


    resource_monitor = ResourceMonitor(interval_s=0.5)
    with resource_monitor, TimerBlock("[Block] Step 2: Vector Embeddings Generation"):
        print(f"Embedding {len(all_chunks):,} chunks with {pathlib.Path(embedding_model_path).stem} ...")

        # PHASE 2+3: Use CachedEmbedder with pluggable backend
        print(f"Embedding backend: {embed_backend}")
        embedder = CachedEmbedder(embedding_model_path, backend=embed_backend)

        # PHASE 2 OPTIMIZATION: Dynamic Hardware Routing (only meaningful for llama_cpp)
        has_gpu = detect_gpu()
        if has_gpu:
            print("Hardware Router: GPU detected. Forcing sequential execution to maximize VRAM throughput.")
            should_multiprocess = False
        else:
            print("Hardware Router: CPU-only environment detected.")
            should_multiprocess = use_multiprocessing

        # Multi-process pool only supported for the llama_cpp backend
        if should_multiprocess and embed_backend == "llama_cpp":
            print("Starting CPU multi-process pool for embeddings...")
            pool = embedder.start_multi_process_pool()
            try:
                embeddings = embedder.encode_multi_process(
                    all_chunks,
                    pool,
                    batch_size=32,
                )
            finally:
                embedder.stop_multi_process_pool(pool)
        else:
            print("Starting high-throughput sequential embedding...")
            embeddings = embedder.encode(
                all_chunks,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
            )

    with TimerBlock("[Block] Step 3: Build FAISS Index"):
        print(f"Building FAISS index for {len(all_chunks):,} chunks...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, str(artifacts_dir / f"{index_prefix}.faiss"))
        print(f"FAISS Index built successfully: {index_prefix}.faiss")

    with TimerBlock("[Block] Step 4: Build BM25 Index"):
        print(f"Building BM25 index for {len(all_chunks):,} chunks...")
        tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in all_chunks]
        bm25_index = BM25Okapi(tokenized_chunks)
        with open(artifacts_dir / f"{index_prefix}_bm25.pkl", "wb") as f:
            pickle.dump(bm25_index, f)
        print(f"BM25 Index built successfully: {index_prefix}_bm25.pkl")

    with TimerBlock("[Block] Step 5: Dump Index Artifacts"):
        with open(artifacts_dir / f"{index_prefix}_chunks.pkl", "wb") as f:
            pickle.dump(all_chunks, f)
        with open(artifacts_dir / f"{index_prefix}_sources.pkl", "wb") as f:
            pickle.dump(sources, f)
        with open(artifacts_dir / f"{index_prefix}_meta.pkl", "wb") as f:
            pickle.dump(metadata, f)
        print(f"Saved all index artifacts with prefix: {index_prefix}")
    
    # --- NEW: Print stats at the end of the indexing process ---
    print_profile_stats()

    # Phase 3: persist per-run profile + resource summary alongside experiments
    if profile_output_dir is not None:
        out_dir = pathlib.Path(profile_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print_profile_stats(filepath=str(out_dir / f"{index_prefix}_profiling.txt"))
        resources = resource_monitor.summary()
        profile_dump = {name: dict(stats) for name, stats in PROFILE_STATS.items()}
        with open(out_dir / f"{index_prefix}_resources.json", "w") as f:
            json.dump({
                "index_prefix": index_prefix,
                "embed_backend": embed_backend,
                "embedding_model_path": embedding_model_path,
                "num_chunks": len(all_chunks),
                "embedding_dim": int(embeddings.shape[1]),
                "resources": resources,
                "profile_stats": profile_dump,
            }, f, indent=2)
        print(f"[Phase 3] Wrote profiling + resources to {out_dir}")

# ------------------------ Helper functions ------------------------------

@timeit("Helper: preprocess_for_bm25")
def preprocess_for_bm25(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9_'#+-]", " ", text)
    tokens = text.split()
    return tokens