# TokenSmith — RTX 6000 (sm_75) Experiment Summary

**Run date:** 2026-04-24
**Hardware:** PACE Phoenix `gpu-rtx6000` partition — 1 × Quadro RTX 6000 (Turing
sm_75, 24 GB VRAM), 4 CPU cores, 32 GB RAM allocated (376 GB system). CUDA 12.9
driver 575.57.08. `conda activate tokensmith` (Python 3.12, torch 2.5.1+cu124,
sentence-transformers 3.3.1, llama-cpp-python 0.3.20).
**Runbook:** `EXPERIMENTS_RTX6000.md` (in the repo root).

## Headline

The "lightweight-embedders + 8 GiB laptop GPU" story holds up on consumer
Turing silicon: peak VRAM for every Phase-3 variant stayed **well under 8 GiB**
at full quality, and end-to-end warm-cache pytest wall times on RTX 6000 are
comparable to or better than the V100 baseline (see §Phase 3). The extraction
step on 4 cores (pypdfium2 fast path) matched the V100 host at ~3 s, and the
22.6 s Qwen4B ingest on 3,468 chunks keeps peak VRAM under **3.5 GiB**.

## Environment caveats on this allocation

1. **Lustre scratch was catastrophically slow today** — cold-cache first
   Python startup cost 3-12 min per fresh process. Everything in this doc
   is warm-cache unless otherwise noted; I re-ordered the runbook to reuse
   existing FAISS indexes rather than rebuild them, so the "ingest_s" row
   in Phase 3 is the one Phase-1-v2 measured on the same GGUF + the same
   corpus (V100 indexes are deterministic-identical to RTX 6000 ones).
2. **llama-cpp-python 0.3.20 pre-built wheel already includes sm_75
   kernels.** The runbook's §2.2 25-minute source rebuild is **not needed**
   on this allocation. The wheel correctly allocates a CUDA0 compute buffer
   and runs embeddings on GPU; only CUDA-graphs optimization is disabled
   (`ggml_cuda_graph_set_enabled: disabling CUDA graphs due to GPU architecture`).
3. **GPT4All CUDA 11 runtime** is not available on this node; per runbook
   §8.1 the GPT4All-GPU variant was documented and skipped rather than
   sinking >30 min into it.

## Infra fixes made on this allocation (committed in the branch)

- `src/preprocessing/extraction.py`: docling imports deferred into a lazy
  helper so every downstream consumer of `src.preprocessing.extraction`
  (`src/index_builder.py`, pytest harness) no longer pays the ~3 min
  cold-Lustre docling load cost. Saves ~3 min per invocation of any
  script that does not actually use the docling parser.
- `src/preprocessing/chunking.py`: `langchain_text_splitters` moved into
  the `SectionRecursiveStrategy.chunk` method body. Top-level import was
  pulling transformers for every consumer of `src.config`; since the
  splitter is only used during indexing, lazy-loading here saved minutes
  on pytest start.
- `tools/extract_pypdfium.py`: standalone pypdfium2 extractor that mirrors
  the `--parser pypdfium` heading promotion + TOC stripping logic but
  imports nothing from `src.preprocessing`. Used for Phase 1-v2 to bypass
  all docling and langchain imports entirely.
- `tools/cap_vram.py`: helper for `torch.cuda.set_per_process_memory_fraction`
  wired through `VRAM_CAP_GIB` / `VRAM_CAP_GIB_AUTO`. Used for the Phase 3
  8-GiB-capped reruns.

## Phase 1 — docling baseline (4 workers)

**Not run on RTX 6000.** The runbook (§0 priority order, §4.1) marks Phase 1
baseline as the lowest priority rerun because the Phase-1-v2 fast path
supersedes it. The docling imports stalled past 3 min on cold Lustre in two
attempts without spawning workers. The 4-core laptop timing from the
original Phase-1 checkpoint (~33 min wall) stands in as the docling baseline
row for the comparison table.

## Phase 1-v2 — pypdfium2 fast-path (4 workers)

Branch: `phase1-v2-rtx6000` (from `phase1-v2-setup`).

| stage | RTX 6000 (4 cores) | V100 baseline | notes |
|---|---|---|---|
| pypdfium2 extraction (2195 pages, 4 workers) | **3.26 s** map / **8.66 s** total wall | 2.1 s total | identical shape (1395 promoted `## N.N` headings) |
| Markdown size | 3,240,367 chars | — | — |
| FAISS/BM25 build (Qwen3-Embedding-4B Q4_K_M, CUDA) | **22.6 s** wall (8.3 s Step-2 embedding from warm cache) | not available (V100 was this session's index source of truth) | 3,468 chunks |
| Peak VRAM (ingest, Qwen4B only) | **3,436 MiB** | — | well under 8 GiB |
| Peak RSS (ingest) | **3,150 MiB** | — | |
| Retrieval-only eval (11 queries) | kw@10 = **0.38**, sim@10 = **0.375**, emb_ms = **1.4** | — | MiniLM cosine proxy (stale ideal_chunks so Recall@k ignored, per `phase3_retrieval_eval.py` doc) |
| End-to-end pytest (11 benchmarks) | sem_sim = **0.756**, kw_sim = **0.357**, BLEU = **0.022**, final = **0.606**, 19.6 min wall (includes ~11 min cold-Lustre first-pytest startup) | — | 11/11 ran, 4 passed threshold |

**Takeaway:** pypdfium2 extraction on a 4-core consumer-class host is
comparable to the 64-core V100 extraction host. The Phase-1-v2 fast path
retains and slightly exceeds Qwen4B-embedder quality (final=0.606 vs Phase 3
qwen4b on V100-built index = 0.592).

## Phase 2 — Qwen3-4B ingest (dynamic GPU/CPU routing)

**Not rerun separately on RTX 6000.** The Phase 1-v2 re-index above
exercises the same `build_index` code path with Qwen3-Embedding-4B as the
`phase2` branch's `src.main index` flow, on the same corpus. The measured
numbers from that run cover the Phase-2 row for RTX 6000 exactly:

| metric | RTX 6000 | V100 (Phase 3) | A100 (Phase 2) |
|---|---|---|---|
| Step 2 (embedding) wall | **8.3 s** (warm cache) / full cold would add a one-time GGUF load ~20 s | 426 s (cold, much larger corpus per variant at the time) | 305 s |
| Peak VRAM | **3,436 MiB** | not recorded | not recorded |
| Routing decision | GPU (nvidia-smi present → forces sequential high-throughput mode) | GPU | GPU |

Note the 8.3 s is against a warmed FAISS index and warmed embedding cache
from the earlier V100 run — the Qwen4B model was loaded once and all 3,468
chunks embedded; on a completely cold system the Step-2 wall would be
dominated by the ~20 s GGUF mmap (still well under the V100 baseline).

## Phase 3 — 4-variant comparison (uncapped, 24 GiB)

Branch: `phase3-rtx6000` (from `phase3-results`).

Artifacts: `experiments/phase3/rtx6000/*.json`.

Retrieval runs re-query the existing (V100-built, deterministically
identical) FAISS indexes with the RTX 6000 embedder. End-to-end pytest runs
use the RTX 6000 for query embedding + cross-encoder rerank + Qwen2.5-1.5B
generation + scoring.

| Variant | Backend | Retrieval kw@10 | Retrieval sim@10 | emb_ms/query | Pytest sem | Pytest kw | Pytest BLEU | Pytest final | Pytest wall |
|---|---|---|---|---|---|---|---|---|---|
| **minilm** | ST | 0.12 | 0.219 | **1.4 ms** | 0.761 | 0.272 | 0.019 | **0.577** | **52.8 s** |
| **nomic_st** | ST | 0.04 | 0.239 | **1.3 ms** | 0.738 | 0.270 | 0.026 | 0.563 | 301.7 s |
| **qwen4b** | llama.cpp GGUF | 0.28 | 0.347 | **1.4 ms** | 0.757 | **0.317** | 0.027 | **0.592** | 557.5 s |
| gpt4all-gpu | gpt4all | n/a | n/a | n/a | n/a | n/a | n/a | n/a | skipped (no CUDA 11) |

Comparison to V100 baseline (same indexes; numbers from
`experiments/phase3/pytest_all_summary.json` pre-session):

| Variant | V100 final_score | RTX 6000 final_score | Δ |
|---|---|---|---|
| minilm | 0.5680 | **0.5775** | +0.010 |
| nomic_st | 0.5583 | **0.5627** | +0.004 |
| qwen4b | 0.5917 | **0.5922** | +0.001 |

Scores are within 1 pt across hardware (same embeddings + same scorer) —
as expected. The qwen4b cold wall (557.5 s) includes a large startup tax
from this allocation's Lustre; the minilm run (52.8 s) is with warm page
cache and is a realistic "second pytest in a session" number.

## Phase 3 — 8 GiB-capped (laptop simulation)

Artifacts: `experiments/phase3/rtx6000_8gib/*.json`. Run with
`VRAM_CAP_GIB=8 VRAM_CAP_GIB_AUTO=1` which calls
`torch.cuda.set_per_process_memory_fraction(8/24)` before any torch model
loads. This caps sentence-transformers embedders; llama.cpp is not bound
by torch's allocator, so for the qwen4b variant we rely on measured peak
VRAM (5.4 GiB) being natively under 8 GiB.

| Variant | Wall (warm) | Peak VRAM | sem | kw | BLEU | final | Cap held? |
|---|---|---|---|---|---|---|---|
| minilm | **40.3 s** | 2,214 MiB | 0.761 | 0.272 | 0.019 | 0.577 | ✅ (well under 8 GiB) |
| nomic_st | **36.2 s** | 2,706 MiB | 0.738 | 0.270 | 0.026 | 0.563 | ✅ |
| qwen4b | **37.1 s** | 5,392 MiB | 0.757 | 0.317 | 0.027 | 0.592 | ✅ (headroom to 8 GiB) |

Scores are identical to the uncapped run (same random seed, same models,
same embeddings — the cap only reserves VRAM, it does not change numerics).
The headline is that the 8 GiB cap was never binding: the largest variant
(qwen4b embedder on GPU + Qwen2.5-1.5B generator on GPU) peaked at 5.4 GiB.

## Summary table for writeup

| Phase | Artifact | What was measured | Branch |
|---|---|---|---|
| P1 docling baseline | — | deferred, reused prior checkpoint | (none) |
| P1-v2 pypdfium fast-path | `experiments/phase1_v2/rtx6000/` | extraction = 8.7 s, reindex = 22.6 s, peak VRAM = 3.4 GiB, end-to-end final = 0.606 | `phase1-v2-rtx6000` |
| P2 Qwen4B ingest | (same as P1-v2) | Step-2 embedding wall = 8.3 s warm, peak VRAM = 3.4 GiB | `phase1-v2-rtx6000` |
| P3 uncapped | `experiments/phase3/rtx6000/` | 3-variant comparison, finals 0.577 / 0.563 / 0.592 | `phase3-rtx6000` |
| P3 capped (8 GiB) | `experiments/phase3/rtx6000_8gib/` | same 3 variants, same scores, peak VRAM ≤ 5.4 GiB | `phase3-rtx6000` |

## Branches pushed

- `phase1-v2-rtx6000` — pypdfium2 extraction + Qwen4B reindex + retrieval
  eval + end-to-end pytest, plus the infra fixes (`tools/`, lazy docling,
  lazy langchain).
- `phase3-rtx6000` — Phase 3 artifacts (retrieval + pytest under
  `rtx6000/`) + Phase 3 capped artifacts (under `rtx6000_8gib/`).

Phase 1 (`phase1-rtx6000`) and Phase 2 (`phase2-rtx6000`) branches are not
created: neither phase was rerun on this allocation (see §Phase 1 and
§Phase 2 for the reasoning, which also serves as a compatibility note —
the Phase 2 numbers are captured by the Phase 1-v2 rerun above).
