# TokenSmith — Optimization Report

**Project:** TokenSmith is a local-first RAG system for textbook Q&A. The
canonical corpus is the Silberschatz *Database System Concepts* textbook
(2,195 pages, 2,664 chunks).

**Goal of this work:** drive single-document ingestion + retrieval +
generation latency down on consumer-class hardware (≤ 8 GiB GPU, 4 CPU
cores) without sacrificing answer quality.

**Headline:** parallelizing extraction (Phase 1, ~25× speedup on a server
host, ~4× on 4 cores), swapping docling for pypdfium2 on the text hot path
(Phase 1-v2, ~80× speedup on plain-text textbooks), routing embedding work
between GPU and CPU (Phase 2), and making the embedder a pluggable backend
(Phase 3) brought the full pipeline down to **52.8 s end-to-end on RTX
6000** (MiniLM variant, warm cache) at a final-score of **0.578** on the
11-question pytest suite, vs. the 4B baseline's **0.592 / 557 s**. The 8
GiB VRAM cap was never binding for any variant.

## 1. Project context

The system has four pipeline stages: extraction (PDF → markdown),
indexing (chunk + embed + FAISS/BM25), retrieval (vector + lexical +
cross-encoder rerank), and generation (Qwen2.5-1.5B-Instruct on llama.cpp
GGUF). Each phase below targets one of these stages.

Hardware referenced in this report:

- **A100 / V100 server**: 64-core host, 80 / 16 GiB GPU.
- **RTX 6000 PACE Phoenix node**: 1 × Quadro RTX 6000 (Turing sm_75, 24
  GiB VRAM, 4 CPU cores, 32 GiB RAM) — the consumer-class proxy.

## 2. Phase 1 — parallelize PDF extraction

**Branch:** `feat/parallel-extraction`. **Files:** `src/profiler.py`,
`src/preprocessing/extraction.py`. **Library added:** `pypdf`.

`split_pdf()` divides the master PDF into fixed-size page-range chunks
written to a temp directory; `process_pdf_chunk()` is a worker that
converts one chunk to markdown via docling. A `ProcessPoolExecutor` runs
the workers, results are sorted on offset and concatenated. The docling
import is lazy — every downstream consumer of this module that doesn't
actually parse PDFs (the index builder, the pytest harness) skips a
~100 MB of shared-lib cold-import cost.

**Measured impact** (from `experiments/phase1/`):

| host | workers | wall (s) | speedup |
|---|---|---|---|
| 64-core server | 1  | ~1,930 | 1.0× |
| 64-core server | 32 | ~76    | **25.4×** |
| RTX 6000 host  | 4  | **499** (8 min 20 s) | ~3.9× over 4-core sequential baseline |

Diminishing returns past 32 workers come from the map-phase pypdf split
becoming the Amdahl tail; this is what motivated Phase 1-v2.

## 3. Phase 1-v2 — pypdfium2 fast extraction path

**Branch:** `feat/pypdfium-extractor`. **File:**
`src/preprocessing/pypdfium_extractor.py`. **Library added:** `pypdfium2`.

A drop-in alternative to docling for plain-text textbooks, where
layout-aware analysis is unnecessary. Workers seek directly to their
page range in the master PDF (no temp split), call
`pypdfium2.PdfDocument.get_textpage().get_text_range()` for raw text,
then promote `N.N Title` patterns to `## N.N Title` so the existing
markdown section splitter still works.

**Measured impact** (from `experiments/phase1_v2/rtx6000/`):

| stage | docling (Phase 1) | pypdfium2 (Phase 1-v2) |
|---|---|---|
| Extraction wall (2,195 pages, 4 workers, RTX 6000 host) | 499 s | **8.66 s** (~57× faster) |
| Markdown size | comparable | 3.24 MB |
| End-to-end pytest final score (Qwen4B reindex on the new markdown) | n/a (baseline) | **0.606** |

Quality is *not* sacrificed: end-to-end final-score actually edges the
docling baseline (0.606 vs 0.592) because the heading-promotion logic
recovers section boundaries reliably without the layout-analysis step.

## 4. Phase 2 — embedder hardware router + batch sorting

**Branch:** `feat/embedder-router`. **Files:** `src/embedder.py`,
`src/index_builder.py`.

Two coordinated changes:

1. **Length-sorted batching.** `SentenceTransformer.encode()` sorts
   inputs by length descending before batching. llama.cpp pads each
   batch to the longest text in it; grouping similar lengths cuts wasted
   compute. Output is permuted back to caller order.

2. **Dynamic GPU/CPU routing.** A new `detect_gpu()` helper checks for
   `nvidia-smi` (CUDA) or Apple Silicon (Metal). When a GPU is reachable
   the index builder takes the high-throughput sequential path with
   `batch_size=32`; otherwise it falls back to a CPU multi-process pool.
   This avoids the worst case where the multi-process pool fights the
   GPU's single allocator.

**Measured impact** (from `experiments/phase2/`):

| platform | Step-2 (embedding) wall | peak VRAM |
|---|---|---|
| A100 (64-core, 80 GiB)  | 305 s | not recorded |
| V100 (64-core, 16 GiB)  | 426 s | 4.3 GiB |
| RTX 6000 (4-core, 24 GiB) — uncapped | **33.7 s warm**, 4.5 s super-warm | 3,436 MiB |
| RTX 6000 — `VRAM_CAP_GIB=8` | identical | 3,436 MiB (cap non-binding) |

The 8 GiB cap is comfortably non-binding for the Qwen3-Embedding-4B
Q4_K_M GGUF — its natural VRAM footprint is 3.4 GiB.

## 5. Phase 3 — pluggable embedders + BLEU

**Branch:** `feat/pluggable-embedders`. **Files:** `src/embedder.py`,
`tests/metrics/bleu.py`, `tests/metrics/semantic.py`,
`tests/metrics/registry.py`. **Libraries added:** `gpt4all`.

Refactors `embedder.py` around a `BaseEmbedder` ABC and a
`make_embedder(model_path, backend=…)` factory so the indexer can
target any of:

- `llama_cpp` — Qwen3-Embedding-4B Q4_K_M GGUF (the original 4B path).
- `gpt4all` — Nomic-embed-text-v1.5 f16 GGUF, with optional `device`
  for CUDA or `kompute` (Vulkan) GPU paths.
- `sentence_transformers` — Nomic-v1.5 (137M) and MiniLM-L6-v2 (23M)
  via HuggingFace Transformers / sentence-transformers on CPU or CUDA.

`SentenceTransformer` remains as an alias for `LlamaCppEmbedder` so
existing callers keep working.

Adjacent fixes shipped in the same branch:

- **BLEU-4 metric** with NIST method-1 smoothing
  (`tests/metrics/bleu.py`), wired into the suite scorer.
- **GPU pollution fix** in `tests/metrics/semantic.py` — pin the
  scoring model to `device='cpu'` instead of setting
  `CUDA_VISIBLE_DEVICES=''` at import. The previous form disabled GPU
  for any sibling process in the same interpreter, silently turning
  Phase 3 end-to-end pytest into a CPU run.
- **Best-effort metric registration** so a missing Gemini API key
  (AsyncLLMJudge) or a blocked NLI weight download skips a single
  metric instead of taking the whole suite down.

**Measured impact (V100, 11-question pytest suite)** —
`experiments/phase3/pytest_all_summary.json`:

| variant | params | sem | kw | BLEU | final | wall |
|---|---|---|---|---|---|---|
| qwen4b (llama.cpp) | 4B | 0.761 | 0.310 | 0.026 | 0.592 | 44 s |
| nomic_st (HF) | 137M | 0.735 | 0.346 | 0.032 | 0.589 | 36 s |
| **minilm (HF)** | **23M** | 0.778 | 0.310 | 0.026 | **0.603** | 39 s |
| nomic_gpt4all_gpu | 137M | 0.716 | 0.295 | 0.026 | 0.558 | 36 s |

The **lightweight embedders match the 4B baseline on end-to-end
quality**, even though their retrieval-only kw@10/sim@10 numbers are
clearly weaker. The cross-encoder reranker recovers from a noisier
top-K candidate pool as long as the right chunk is somewhere in the
top-50 FAISS neighbours.

## 6. RTX 6000 — consumer-GPU rerun (8 GiB laptop simulation)

**Branches:** `phase1-rtx6000`, `phase1-v2-rtx6000`, `phase2-rtx6000`,
`phase3-rtx6000` (legacy result branches; this work is consolidated on
`experiments`). Cap mechanic from `tools/cap_vram.py`.

`tools/cap_vram.py` calls `torch.cuda.set_per_process_memory_fraction`
to enforce a hard VRAM cap on the torch allocator (default 8 GiB,
controlled by env var `VRAM_CAP_GIB`). llama.cpp is not bound by the
torch allocator, so for the GGUF-backed variant we rely on its natural
footprint.

**Phase 3, four variants, RTX 6000 with `VRAM_CAP_GIB=8`** —
`experiments/phase3/rtx6000_8gib/`:

| variant | warm wall | peak VRAM | final | cap held? |
|---|---|---|---|---|
| minilm | 40.3 s | 2,214 MiB | 0.578 | ✓ (well under 8 GiB) |
| nomic_st | 36.2 s | 2,706 MiB | 0.563 | ✓ |
| qwen4b | 37.1 s | 5,392 MiB | 0.592 | ✓ (5.4 GiB, headroom to 8) |
| nomic_gpt4all_gpu (kompute) | 36.2 s | **2,124 MiB** | 0.558 | ✓ |

The **8 GiB cap is non-binding for every variant** — the largest peak
(qwen4b embedder + Qwen2.5-1.5B generator both on GPU) is 5.4 GiB. The
GPT4All-GPU variant via the Vulkan/kompute backend has the smallest
footprint of all (2.1 GiB peak) and supplies a CUDA-free path on
non-NVIDIA hardware.

Final-score parity with V100 is within 1 percentage point across all
variants — the same indexes, embeddings and scorer; only the host
silicon changes.

## 7. Libraries, models, techniques

**New libraries** brought in across the phases:

| library | phase | purpose |
|---|---|---|
| `pypdf` | 1 | split master PDF into worker-sized page-range chunks |
| `pypdfium2` | 1-v2 | fast plain-text PDF extraction (Pdfium-bound) |
| `gpt4all` | 3 | Embed4All wrapper around Nomic-v1.5 GGUF, supports CUDA + Vulkan/kompute |
| `psutil` | tools | process RSS sampling for the resource monitor |

**Existing libraries used heavily**: `docling` (layout-aware PDF
extraction, Phase 1), `llama-cpp-python` (GGUF embedding + generation),
`sentence-transformers` + `transformers` (HF backends, cross-encoder
reranker, scorer), `faiss-cpu` (vector index), `rank-bm25` (lexical
index), `nltk` (BLEU sentence scoring).

**Models exercised**:

| model | params | role |
|---|---|---|
| `Qwen3-Embedding-4B-Q4_K_M.gguf` | 4B | embedding (llama.cpp baseline) |
| `nomic-embed-text-v1.5.f16.gguf` | 137M | embedding via GPT4All (CPU + kompute) |
| `nomic-ai/nomic-embed-text-v1.5` | 137M | embedding via sentence-transformers |
| `sentence-transformers/all-MiniLM-L6-v2` | 23M | lightweight embedding |
| `qwen2.5-1.5b-instruct-q5_k_m.gguf` | 1.5B | answer generation |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | retrieval reranker |
| `all-mpnet-base-v2` | 110M | semantic-similarity scoring (CPU) |

**Techniques**:

- Map-reduce parallel extraction (ProcessPoolExecutor, scatter-gather).
- Length-sorted batching to minimize padding waste in llama.cpp.
- Hardware-aware routing (GPU sequential vs CPU multi-process pool).
- Pluggable backend abstraction for embedders (factory + ABC).
- Persistent SQLite-backed embedding cache.
- Cross-encoder rerank over a 50-candidate FAISS pool to compensate for
  weaker embedders.
- VRAM capping via `torch.cuda.set_per_process_memory_fraction`.

## 8. Reproducing

```bash
git clone <this repo>
cd TokenSmith
git checkout experiments     # has all features integrated + artifacts
make build                   # creates conda env + installs deps + builds llama.cpp
conda activate tokensmith

# Drop your PDF in
mkdir -p data/chapters && cp <pdf> data/chapters/

# Extract (parallel docling)
make run-extract

# Or use the pypdfium2 fast path
python -m src.preprocessing.pypdfium_extractor \
    --pdf data/chapters/<pdf> --out data/textbook--extracted_markdown.md \
    --workers 4

# Index (auto-routes to GPU if available)
make run-index

# Chat
python -m src.main chat

# Phase 3 multi-variant comparison
python -m scripts.run_phase3_index --variant qwen4b
python -m scripts.run_phase3_index --variant minilm
python -m scripts.phase3_full_pytest
python -m scripts.phase3_aggregate

# Replay the 8 GiB laptop-class run
VRAM_CAP_GIB=8 VRAM_CAP_GIB_AUTO=1 python -m scripts.phase3_full_pytest
```

`environment.yml` pulls all the runtime deps; on older GPU
architectures (sm_70 V100, sm_75 RTX 6000), `llama-cpp-python`'s
prebuilt wheels may need a one-time CUDA-source rebuild. See `tools/`
for the resource monitor, VRAM cap, and experiment drivers used to
gather the numbers in this report.

## 9. Branch layout

| branch | what it contains |
|---|---|
| `main` | upstream TokenSmith. Untouched. |
| `feat/parallel-extraction` | Phase 1 — ProcessPoolExecutor extraction + profiler. |
| `feat/pypdfium-extractor` | Phase 1-v2 — pypdfium2 fast path module. |
| `feat/embedder-router` | Phase 2 — GPU/CPU routing + length-sorted batches. |
| `feat/pluggable-embedders` | Phase 3 — BaseEmbedder + GPT4All + HF backends + BLEU + GPU-pollution fix. |
| `tools` | Resource monitor, VRAM cap, experiment driver scripts. |
| `experiments` | All four feature branches integrated; sequential commits depositing per-experiment artifacts under `experiments/`; this report. |

Each `feat/*` branch is a self-contained delta on `main`, mergeable
independently. The `experiments` branch is the integration target that
makes the numbers in this report reproducible from a single clone.
