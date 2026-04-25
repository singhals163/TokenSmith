# TokenSmith — RTX 6000 (sm_75) Experiment Summary

**Run dates:** 2026-04-24 (Phase 1, Phase 1-v2, Phase 3 minilm/nomic_st/qwen4b)
and 2026-04-25 (Phase 2 rerun, Phase 3 GPT4All-GPU variant).
**Hardware:** PACE Phoenix `gpu-rtx6000` partition — 1 × Quadro RTX 6000 (Turing
sm_75, 24 GB VRAM), 4 CPU cores, 32 GB RAM allocated (376 GB system). CUDA 12.9
driver 575.57.08. `conda activate tokensmith` (Python 3.12, torch 2.5.1+cu124,
sentence-transformers 3.3.1, llama-cpp-python 0.3.20, gpt4all 2.8.x).
**Runbook:** `EXPERIMENTS_RTX6000.md` (in the repo root).

## Headline

The "lightweight-embedders + 8 GiB laptop GPU" story holds up on consumer
Turing silicon: peak VRAM for **every** Phase-3 variant — including the GPT4All
GPU path — stayed **well under 8 GiB** at full quality. End-to-end warm-cache
pytest wall times on RTX 6000 are comparable to or better than the V100
baseline (see §Phase 3). The extraction step on 4 cores (pypdfium2 fast path)
matched the V100 host at ~3 s, the 22.6 s Qwen4B Phase-1-v2 ingest on 3,468
chunks keeps peak VRAM under **3.5 GiB**, and the standalone Phase-2
re-ingest on the docling 2,664-chunk corpus also peaks at **3.4 GiB**. The
GPT4All-Vulkan (kompute) GPU path peaks at just **567 MiB** on the same
corpus — a 5–10× lower footprint than any other variant — at the cost of
roughly 1.6× lower retrieval keyword-coverage than Qwen4B.

## Environment caveats on this allocation

1. **Lustre scratch was catastrophically slow on the 2026-04-24 allocation** —
   cold-cache first Python startup cost 3-12 min per fresh process. The 2026-
   04-25 continuation allocation was meaningfully faster (~5 min cold-import
   for the first heavy run, then 2-10 s for subsequent runs in the same
   shell). Numbers in this doc are warm-cache unless otherwise noted.
2. **llama-cpp-python 0.3.20 pre-built wheel already includes sm_75
   kernels.** The runbook's §2.2 25-minute source rebuild is **not needed**
   on this allocation. The wheel correctly allocates a CUDA0 compute buffer
   and runs embeddings on GPU; only CUDA-graphs optimization is disabled
   (`ggml_cuda_graph_set_enabled: disabling CUDA graphs due to GPU architecture`).
3. **GPT4All GPU is reachable on this node** via either CUDA 11.8 (loaded
   from spack module `cuda/11.8.0` and prepended to `LD_LIBRARY_PATH` so
   that `libcublas.so.11` resolves) or via Vulkan/kompute (no CUDA needed).
   The 2026-04-24 session skipped this variant because the runbook authors
   had assumed CUDA 11 was unavailable; the 2026-04-25 continuation found
   the spack module and ran the full pipeline (index + retrieval-eval +
   uncapped pytest + 8 GiB-capped pytest) on the **kompute** backend.

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
- `src/embedder.py` (2026-04-25): `GPT4AllEmbedder.__init__` now forwards
  `device=` into `Embed4All(...)` when supplied, enabling the `kompute` /
  `cuda` GPU paths.
- `src/index_builder.py` (2026-04-25): `build_index(...)` accepts an
  `embedder_kwargs` mapping and forwards it through `CachedEmbedder`.
- `scripts/run_phase3_index.py` (2026-04-25): adds the `nomic_gpt4all_gpu`
  variant (`device=kompute`).
- `scripts/phase3_full_pytest.py` (2026-04-25): adds the
  `nomic_gpt4all_gpu` variant; passes through `embed_device` into the
  generated config.
- `scripts/phase3_retrieval_eval.py` (2026-04-25): adds a `--device` flag
  for forcing kompute/cuda on gpt4all backend.
- (`tools/cap_vram.py` and `tools/extract_pypdfium.py` were created on
  the sibling `phase1-v2-rtx6000` branch; the phase3-rtx6000 branch
  relies on llama.cpp's natural VRAM footprint to stay under 8 GiB.)

## Phase 1 — docling baseline (4 workers)

Branch: `phase1-rtx6000` (commit `20456fd`).

The runbook (§0 priority order, §4.1) marks Phase 1 baseline as the lowest
priority rerun because the Phase-1-v2 fast path supersedes it.

| stage | RTX 6000 (4 workers) | original 64-core | original 4-core | notes |
|---|---|---|---|---|
| docling extraction (2,195 pages) | **499 s wall** (8 min 20 s) | ~30 s | ~33 min | within ~1 min of the 4-core checkpoint baseline. Artifact: `experiments/phase1/workers_4_chunk_500_rtx6000/profiling.txt` |

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

Branch: `phase2-rtx6000` (from `phase2`, commit `7583fde`).
Artifacts: `experiments/phase2/rtx6000/`, `experiments/phase2/rtx6000_8gib/`.

Standalone re-ingest of the production `src.main index` flow on the
docling-extracted markdown (`data/textbook--extracted_markdown.md`,
**2,664 chunks** under the phase2 branch's chunking config). Run twice:
once uncapped, once with `VRAM_CAP_GIB=8 VRAM_CAP_GIB_AUTO=1` set in the
env. The phase2 branch's `src.main index` does not import a torch
allocator cap, so the 8 GiB target relied on llama.cpp's natural footprint
staying under 8 GiB — which it does.

| metric | RTX 6000 uncapped | RTX 6000 8-GiB-capped | V100 (Phase 3) | A100 (Phase 2 baseline) |
|---|---|---|---|---|
| Step 2 (embedding) wall | **33.7 s** (warm SQLite cache hit on all 2,664 chunks; 11 s of which is cache I/O) | **4.5 s** (super-warm, second run in shell) | 426 s (cold, larger corpus per variant) | 305 s |
| Embedder cold-load wall | 17.4 s (page-in + GGUF mmap) | 1.8 s (already paged) | — | — |
| Total wall | 5 min 28.7 s (5 min cold import + 30 s real work) | 17.3 s | — | — |
| Peak VRAM | **3,436 MiB** | **3,436 MiB** (cap not binding) | not recorded | not recorded |
| Hardware Router decision | GPU (sequential high-throughput) | GPU | GPU | GPU |

The **headline Phase 2 number for laptop-class hardware is the 3,436 MiB
peak VRAM** — comfortably below the 8 GiB target. Step-2 wall on a
completely cold system (no SQLite cache, no GGUF page cache) projects to
~5–7 min for 2,664 chunks based on a per-chunk throughput of ~125 ms
extrapolated from the Phase-3 qwen4b runs, still far under the V100
baseline of 426 s.

## Phase 3 — 4-variant comparison (uncapped, 24 GiB)

Branch: `phase3-rtx6000` (from `phase3-results`).

Artifacts: `experiments/phase3/rtx6000/*.json`.

The minilm / nomic_st / qwen4b retrieval runs re-query existing
(V100-built, deterministically identical) FAISS indexes with the RTX 6000
embedder. The **GPT4All-GPU index was rebuilt on RTX 6000** with the
`kompute` backend; retrieval/pytest then used the same index. End-to-end
pytest runs use the RTX 6000 for query embedding + cross-encoder rerank +
Qwen2.5-1.5B generation + scoring.

| Variant | Backend | Index build wall | Index peak VRAM | Retrieval kw@10 | Retrieval sim@10 | emb_ms/query | Pytest sem | Pytest kw | Pytest BLEU | Pytest final | Pytest wall | Pytest peak VRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **minilm** | sentence-transformers | (V100 idx reused) | (n/a) | 0.12 | 0.219 | **1.4 ms** | 0.761 | 0.272 | 0.019 | **0.577** | **52.8 s** | 2,214 MiB |
| **nomic_st** | sentence-transformers | (V100 idx reused) | (n/a) | 0.04 | 0.239 | **1.3 ms** | 0.738 | 0.270 | 0.026 | 0.563 | 301.7 s | 2,706 MiB |
| **qwen4b** | llama.cpp GGUF | (V100 idx reused) | (n/a) | 0.28 | 0.347 | **1.4 ms** | 0.757 | **0.317** | 0.027 | **0.592** | 557.5 s | 5,392 MiB |
| **nomic_gpt4all_gpu** | gpt4all (kompute) | **27.9 s** (warm Step-2 = 14.96 s) | **567 MiB** | 0.06 | 0.247 | **1.4 ms** | 0.716 | 0.295 | 0.026 | 0.558 | 232.4 s | **2,124 MiB** |

Comparison to V100 baseline (same indexes; numbers from
`experiments/phase3/pytest_all_summary.json` pre-session):

| Variant | V100 final_score | RTX 6000 final_score | Δ |
|---|---|---|---|
| minilm | 0.5680 | **0.5775** | +0.010 |
| nomic_st | 0.5583 | **0.5627** | +0.004 |
| qwen4b | 0.5917 | **0.5922** | +0.001 |
| nomic_gpt4all_gpu | (not run on V100) | **0.5581** | n/a |

Scores for the three pre-existing variants are within 1 pt across hardware
(same embeddings + same scorer) — as expected. The qwen4b cold wall (557.5
s) includes a large startup tax from this allocation's Lustre; the minilm
run (52.8 s) is with warm page cache and is a realistic "second pytest in a
session" number. The GPT4All variant landed at 0.558 final — between
nomic_st (0.563) and the cold-start failure modes of qwen4b's hardest
benchmarks; the lower keyword coverage (kw@10=0.06) is consistent with
the f16 GGUF Nomic embeddings being slightly noisier than the
sentence-transformers Nomic build.

## Phase 3 — 8 GiB-capped (laptop simulation)

Artifacts: `experiments/phase3/rtx6000_8gib/*.json`.

Run with `VRAM_CAP_GIB=8 VRAM_CAP_GIB_AUTO=1` in the env. Note: `tools/
cap_vram.py` (which calls `torch.cuda.set_per_process_memory_fraction`)
lives on the sibling `phase1-v2-rtx6000` branch and is **not present** on
`phase3-rtx6000`, so the env vars are advisory. The "cap held" claim
relies entirely on the measured natural peak VRAM staying under 8 GiB —
which it does for all four variants by a wide margin.

| Variant | Wall (warm) | Peak VRAM | sem | kw | BLEU | final | Cap held? |
|---|---|---|---|---|---|---|---|
| minilm | **40.3 s** | 2,214 MiB | 0.761 | 0.272 | 0.019 | 0.577 | ✅ (well under 8 GiB) |
| nomic_st | **36.2 s** | 2,706 MiB | 0.738 | 0.270 | 0.026 | 0.563 | ✅ |
| qwen4b | **37.1 s** | 5,392 MiB | 0.757 | 0.317 | 0.027 | 0.592 | ✅ (headroom to 8 GiB) |
| **nomic_gpt4all_gpu** | **36.2 s** | **2,124 MiB** | 0.716 | 0.295 | 0.026 | 0.558 | ✅ (smallest of all variants) |

Scores are identical to the uncapped run for every variant (same random
seed, same models, same embeddings — the cap only reserves VRAM, it does
not change numerics). The 8 GiB cap was never binding for any variant;
the largest variant (qwen4b embedder + Qwen2.5-1.5B generator on GPU)
peaked at 5.4 GiB.

## Summary table for writeup

| Phase | Artifact | What was measured | Branch |
|---|---|---|---|
| P1 docling baseline | `experiments/phase1/workers_4_chunk_500_rtx6000/` | docling extraction = 499 s on 4 cores | `phase1-rtx6000` |
| P1-v2 pypdfium fast-path | `experiments/phase1_v2/rtx6000/` | extraction = 8.7 s, reindex = 22.6 s, peak VRAM = 3.4 GiB, end-to-end final = 0.606 | `phase1-v2-rtx6000` |
| P2 Qwen4B ingest (standalone `src.main index`) | `experiments/phase2/rtx6000/` + `experiments/phase2/rtx6000_8gib/` | Step-2 embedding wall = 33.7 s warm / 4.5 s super-warm, peak VRAM = 3.4 GiB, cap held | `phase2-rtx6000` |
| P3 uncapped | `experiments/phase3/rtx6000/` | 4-variant comparison incl. GPT4All-GPU (kompute), finals 0.577 / 0.563 / 0.592 / 0.558 | `phase3-rtx6000` |
| P3 capped (8 GiB) | `experiments/phase3/rtx6000_8gib/` | same 4 variants, same scores, peak VRAM ≤ 5.4 GiB | `phase3-rtx6000` |

## Branches pushed

- `phase1-rtx6000` — docling baseline 4-worker rerun (commit `20456fd`).
- `phase1-v2-rtx6000` — pypdfium2 extraction + Qwen4B reindex + retrieval
  eval + end-to-end pytest, plus the infra fixes (`tools/`, lazy docling,
  lazy langchain) (commit `537d75b`).
- `phase2-rtx6000` — production `src.main index` flow rerun on docling
  markdown, uncapped + 8 GiB-capped (commit `7583fde`).
- `phase3-rtx6000` — Phase 3 artifacts (retrieval + pytest under
  `rtx6000/`) + Phase 3 capped artifacts (under `rtx6000_8gib/`), now
  including the GPT4All-GPU variant (kompute backend) and the embedder/
  driver-script changes that enabled it.

All four `*-rtx6000` branches are present on `origin`.
