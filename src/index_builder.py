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
from typing import List, Dict

import faiss
from rank_bm25 import BM25Okapi
from src.embedder import SentenceTransformer

from src.preprocessing.chunking import DocumentChunker, ChunkConfig
from src.preprocessing.extraction import extract_sections_from_markdown

# --- NEW: Import profiler tools ---
from src.profiler import timeit, TimerBlock, print_profile_stats

# ----- runtime parallelism knobs (avoid oversubscription) -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

DEFAULT_EXCLUSION_KEYWORDS = ['questions', 'exercises', 'summary', 'references']

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
    use_headings: bool = False
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


    with TimerBlock("[Block] Step 2: Vector Embeddings Generation"):
        print(f"Embedding {len(all_chunks):,} chunks with {pathlib.Path(embedding_model_path).stem} ...")
        embedder = SentenceTransformer(embedding_model_path)

        if use_multiprocessing:
            print("Starting multi-process pool for embeddings...")
            pool = embedder.start_multi_process_pool(workers=4)
            try:
                embeddings = embedder.encode_multi_process(
                    all_chunks, 
                    pool, 
                    batch_size=32
                )
            finally:
                embedder.stop_multi_process_pool(pool)
        else:
            embeddings = embedder.encode(
                all_chunks, 
                batch_size=8, 
                show_progress_bar=True,
                convert_to_numpy=True 
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

# ------------------------ Helper functions ------------------------------

@timeit("Helper: preprocess_for_bm25")
def preprocess_for_bm25(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9_'#+-]", " ", text)
    tokens = text.split()
    return tokens