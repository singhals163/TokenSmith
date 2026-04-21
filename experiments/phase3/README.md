# Phase 3 — Lightweight Indexing Engine (GPT4All) Evaluation

**Proposal goal:** integrate GPT4All and benchmark lightweight embedding models
(e.g. the ~137M Nomic model) against the heavy 4B Qwen3 embedder. Measure
(1) ingestion throughput, (2) peak RAM/VRAM, (3) retrieval degradation via
BLEU / semantic similarity.

## Setup

- Hardware: 1× Tesla V100-PCIE-16GB (sm_70), 64-core host CPU
- Corpus: Silberschatz textbook, 2,664 chunks (same chunking as Phase 2)
- Evaluator: MiniLM-L6 used as a neutral answer-similarity scorer
- Cache redirects for reproducibility are in `.phase3_env.sh` (HF / pip /
  GPT4All / LD paths all point to `/storage/scratch1/1/ssinghal88/...`)

## Variants

| # | Backend | Model | Dim | Notes |
|---|---|---|---|---|
| A | `llama.cpp` | Qwen3-Embedding-4B-Q4_K_M | 2560 | V100 CUDA (rebuilt llama-cpp-python for sm_70) |
| B | `gpt4all` | nomic-embed-text-v1.5.f16 | 768 | CPU (GPT4All CUDA blob is still on cu11 on this node) |
| C | `sentence-transformers` | nomic-ai/nomic-embed-text-v1.5 | 768 | V100 CUDA |
| D | `sentence-transformers` | sentence-transformers/all-MiniLM-L6-v2 | 384 | V100 CUDA |

## Results — Retrieval (model-agnostic proxies)

| Variant | Params | Dim | Ingest Step 2 (s) | Peak VRAM (MiB) | kw@5 | kw@10 | sim@5 | sim@10 | emb ms |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-4B (llama.cpp, V100) | 4000M | 2560 | 426.0 | 4332 | 0.24 | 0.30 | 0.329 | 0.347 | 52.8 |
| Nomic-v1.5 (GPT4All, CPU) | 137M | 768 | 7212† | 972 | 0.00‡ | 0.04‡ | 0.177‡ | 0.239‡ | 11.1‡ |
| Nomic-v1.5 (sentence-transformers, V100) | 137M | 768 | 125.9 | 5990 | 0.00 | 0.04 | 0.177 | 0.239 | 11.1 |
| all-MiniLM-L6-v2 (sentence-transformers, V100) | 23M | 384 | 62.1 | 688 | 0.08 | 0.12 | 0.183 | 0.219 | 52.9 |

(For reference, the Phase 2 A100 run measured Qwen4B Step 2 at **305 s** — the V100 is ~40% slower at fp16, as expected given Volta vs Ampere.)

## Results — Generation (full pytest pipeline)

Runs the production path: FAISS retrieval → cross-encoder rerank (top 5) →
Qwen2.5-1.5B-Instruct generation → metric scoring against `expected_answer`.
Metrics: `semantic_similarity` (all-mpnet-base-v2 cosine, weight 0.7),
`keyword_similarity` (keyword coverage in generated answer, weight 0.3),
and `bleu` (sentence-BLEU-4 with method-1 smoothing, reported alongside
but weight 0 in the final score so we don't perturb the existing scorer).

All four variants run with the generator (Qwen2.5-1.5B) on the V100 via
`llama-cpp-python` (rebuilt from source with `-DCMAKE_CUDA_ARCHITECTURES=70`
for sm_70). The embedder only runs for the 11 query-embeds during pytest
(trivial), so wall-time is dominated by LLM generation.

| Variant | n | semantic_similarity | keyword_similarity | bleu | final_score | wall (s) |
|---|---|---|---|---|---|---|
| Qwen3-Embedding-4B (V100) | 11 | 0.761 | 0.310 | 0.026 | 0.592 | 44 |
| Nomic-v1.5 (GPT4All, CPU) | 11 | 0.735‡ | 0.346‡ | 0.032‡ | 0.589‡ | 36‡ |
| Nomic-v1.5 (sentence-transformers, V100) | 11 | 0.735 | 0.346 | 0.032 | 0.589 | 36 |
| all-MiniLM-L6-v2 (sentence-transformers, V100) | 11 | 0.778 | 0.310 | 0.026 | 0.603 | 39 |

(Weighted `final_score = 0.7*semantic + 0.3*keyword`. Wall time is pytest
wall-clock with warm GGUF page-cache; a cold first-run adds ~270 s for
Qwen4B embedder load and ~10 s for the lightweight variants.)

*† Projected from a 100-chunk sample (2.7 s/chunk on CPU; GPT4All's CUDA build on this node requires libcublas.so.11, which is absent).*
*‡ Retrieval quality for GPT4All Nomic was not re-computed — GPT4All and ST wrap the same Nomic-embed-text-v1.5 weights and differ only by quantization (f16 vs fp32), so we reuse the ST numbers. A dedicated end-to-end GPT4All quality run would cost ~2h just for index building.*

### Metric definitions

- `kw@k` — fraction of benchmark keywords that appear in any top-k retrieved
  chunk (substring match after whitespace strip, needed because docling
  extraction produces per-character spacing on every chunk).
- `sim@k` — max cosine similarity between any top-k retrieved chunk and the
  benchmark's `expected_answer`, computed with MiniLM-L6 as a fixed evaluator.
  Higher = retriever surfaced chunks that are semantically closer to the gold
  answer. This proxy is used **in place of** `ideal_retrieved_chunks`-based
  Recall@k because that field in `tests/benchmarks.yaml` turned out to be
  stale — it references chunk IDs from an older chunking schema (e.g.
  benchmark `acid_properties` lists chunks 1143–1148 which the current
  chunker puts in "File Organization / Variable-Length Records").

## Findings

1. **The retrieval-only quality gap largely closes by the time you measure
   the full RAG pipeline.** At the FAISS-only stage Qwen4B is clearly
   stronger (`sim@10=0.347` vs `0.239` for Nomic, `0.219` for MiniLM;
   `kw@10=0.30` vs `0.04` vs `0.12`). But once we add the cross-encoder
   reranker and the Qwen2.5-1.5B generator, end-to-end `semantic_similarity`
   is statistically indistinguishable: 0.761 / 0.735 / 0.778 for
   Qwen4B / Nomic / MiniLM. `final_score` lands at 0.592 / 0.589 / 0.603 —
   all three within ~1.5 points, with MiniLM actually leading. The reranker
   is doing meaningful rescue work — it starts from top-50 FAISS candidates
   and selects the best 5 — so first-stage recall just needs to be high
   enough for the right chunk to be in the pool. This is a non-obvious
   result and it changes the conclusion a student should draw from Phase 3:
   on this corpus, **lightweight embedders are near drop-in-replaceable
   when the rest of the pipeline is intact**. BLEU is roughly equal across
   variants (0.026–0.032), which isn't surprising — BLEU penalizes the
   paraphrasing an LLM tends to produce, so it saturates at low values for
   all generative-RAG outputs.

2. **Throughput shows the promised speedup and it compounds.** On the same
   V100, ST-Nomic ingests 2,664 chunks in 126 s vs the Qwen4B A100 baseline
   of 305 s (~2.4× speedup *against a stronger GPU*). MiniLM is faster still
   at 62 s (~5× vs A100-Qwen4B; ~2× vs V100-Nomic). Combined with point (1),
   lightweight models give a clear win for the proposal's "time-to-first-
   query" objective without a user-visible quality cost after reranking.

3. **GPT4All on this node is a strict CPU fallback.** `libllamamodel-
   mainline-cuda.so` requires `libcublas.so.11` — GPT4All's bundled engine
   is still on the CUDA 11 track while all other tooling in the env is on
   CUDA 12. It silently falls back to CPU and clocks in at 2.7 s/chunk
   (≈ 2 hours for the full corpus). For laptop users this number is
   representative of the "consumer hardware" scenario the proposal targets.

4. **Peak VRAM is not strictly monotone in model size.** ST-Nomic
   allocated ~6 GiB of VRAM on the V100 despite being 137M params — that
   looks like a sentence-transformers / torch allocator effect rather than a
   real working-set requirement (MiniLM sits at 688 MiB as expected). This
   is worth noting because the proposal promised VRAM measurements; a
   follow-up should force `torch.cuda.empty_cache()` between stages.

5. **The prebuilt llama-cpp-python wheel is not sm_70-compatible — fixed
   with a source rebuild.** The pip wheel ships CUDA kernels compiled for
   Ampere+ (sm_80+) only, so on the V100 it dies with
   `CUDA error: no kernel image is available for execution on the device`.
   The fix is a source rebuild after loading a CUDA 12.6 toolkit for nvcc:

   ```bash
   module load cuda/12.6.1
   pip install --upgrade --force-reinstall --no-binary=llama-cpp-python \\
       --no-deps --verbose \\
       --config-settings=cmake.define.GGML_CUDA=ON \\
       --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=70 \\
       llama-cpp-python
   ```

   Compilation ~25 min on this node. At runtime only `libcudart.so.12` is
   needed (already shipped by the `nvidia-cuda-runtime-cu12` pip package
   that torch pulls in). `.phase3_env.sh` has a commented block for the
   build-time module load.

## Files

- `phase3_<variant>_profiling.txt` — stage timings per run
- `phase3_<variant>_resources.json` — full profiler dump + RAM/VRAM snapshot
- `phase3_<variant>_retrieval_eval.json` — per-query retrieval-only scores
- `phase3_<variant>_pytest_summary.json` — per-variant end-to-end pytest summary
- `phase3_<variant>_pytest_results.jsonl` — per-benchmark pytest details (one JSON per line)
- `phase3_nomic_gpt4all_throughput.json` — sampled-throughput projection for GPT4All
- `summary.json` — aggregated across all variants (retrieval + generation + resources)
- `comparison.png` — dual-panel: (left) throughput vs retrieval-only; (right) retrieval-only vs end-to-end
- `pytest_all_summary.json` — combined pytest summaries

## Reproducing

```bash
source .phase3_env.sh
# one-time: pip install gpt4all einops matplotlib + a few version pins
# (the env file already has the LD_PRELOAD + HF cache redirects)

# Build indexes (per variant)
python -m scripts.run_phase3_index --variant minilm
python -m scripts.run_phase3_index --variant nomic_st
python -m scripts.run_phase3_index --variant qwen4b    # works on V100 after rebuild (see finding 5)
python -m scripts.phase3_gpt4all_throughput --sample-size 100 --batch-size 32

# Quality
python -m scripts.phase3_retrieval_eval --prefix phase3_minilm    --model sentence-transformers/all-MiniLM-L6-v2 --backend sentence_transformers
python -m scripts.phase3_retrieval_eval --prefix phase3_nomic_st  --model nomic-ai/nomic-embed-text-v1.5         --backend sentence_transformers
CUDA_VISIBLE_DEVICES="" python -m scripts.phase3_retrieval_eval --prefix phase3_qwen4b --model models/Qwen3-Embedding-4B-Q4_K_M.gguf --backend llama_cpp --n-gpu-layers 0

# End-to-end pytest (BLEU+ can be added by writing a BLEU metric; the current
# repo ships semantic + keyword)
python -m scripts.phase3_full_pytest

# Aggregate everything
python -m scripts.phase3_aggregate
```
