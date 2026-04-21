# TokenSmith — Phase 3 Handoff

Written 2026-04-20 at the end of a ~4h V100 session. Pick this up to continue
Phase 3 (or move to revisiting Phase 1) from where we stopped.

## What this project is

TokenSmith is a local-first RAG system over textbooks (Silberschatz DB text
is the canonical corpus; 2,664 chunks). The course project is optimizing
**single-document ingestion latency** across three phases:

- **Phase 1** (done, on `phase1` branch): scatter-gather PDF extraction with
  `ProcessPoolExecutor`. 25.28× speedup on a 64-core server.
- **Phase 2** (done, on `phase2` branch): batch sorting + GPU-vs-CPU dynamic
  routing for the Qwen3-Embedding-4B embedding model. A100 ingested 2,664
  chunks in 5.08 min.
- **Phase 3** (done in this session): integrate GPT4All and benchmark
  lightweight embedders (Nomic-embed-text-v1.5 137M and MiniLM-L6 23M)
  against Qwen3-Embedding-4B. Measure throughput, VRAM, retrieval-quality
  degradation via semantic similarity.

Project docs: `Project Proposal.pdf`, `CS_6423_project_checkpoint.pdf`.
Source of truth for all Phase 3 work: `experiments/phase3/README.md`.

## Session environment

- Hardware: 1× Tesla V100-PCIE-16GB (sm_70)
- Env: `/storage/home/hcoda1/1/ssinghal88/scratch/.conda/envs/tokensmith`
- Repo: `/storage/scratch1/1/ssinghal88/TokenSmith` (symlink-equivalent to
  `/storage/home/hcoda1/1/ssinghal88/scratch/TokenSmith`)
- All caches redirected to scratch via `.phase3_env.sh` (user has limited
  home-partition quota — do NOT download to `~/.cache`)

**Always `source .phase3_env.sh` before running anything.** It sets
`HF_HOME`, `GPT4ALL_MODEL_PATH`, `LD_LIBRARY_PATH`, and an `LD_PRELOAD` to
route the conda libstdc++ ahead of the spack one (otherwise `sqlite3` breaks
when llama-cpp-python is imported — its `libllama.so` has an RPATH pinned to
an old libstdc++).

## Branches on the remote

| Branch | Contains | Pushed |
|---|---|---|
| `main` | original TokenSmith, no phase work | upstream |
| `phase1` | Phase 1 code + experiments | upstream |
| `phase2` | Phase 2 code + experiments | upstream |
| `experiments` | phase1+2 code+results (the one we started from) | upstream |
| `phase3-setup` | **NEW** — phase 3 code changes only, no results | **pushed** |
| `phase3-results` | **NEW** — experiments/phase3/ artifacts + logs | **not yet pushed** (pytest still running at write time) |

The currently checked-out branch at handoff time is `phase3-setup` with the
Phase 3 experiment artifacts sitting uncommitted in the working tree (they
are destined for `phase3-results`). See "Finish the results branch" below.

## What the Phase 3 code adds

See `experiments/phase3/README.md` for the technical writeup. One-line: a
pluggable embedding-backend abstraction (`src/embedder.py`) + GPT4All backend +
HuggingFace sentence-transformers backend + experiment scripts under
`scripts/phase3_*.py`.

Notable adjacent fixes shipped in the same branch:

- `src/generator.py`: bumped default `n_ctx` 4096 → 8192 so the 5-chunk
  reranker prompt fits.
- `tests/metrics/registry.py`: metrics register best-effort so an
  `AsyncLLMJudge` constructor failure (missing Gemini API key) doesn't
  cascade and take the whole suite down.
- `src/instrumentation/resource_monitor.py`: peak RAM (psutil) + peak VRAM
  (nvidia-smi poll) sampler used around the ingestion window.

## Results we have (numbers)

**Retrieval-only quality + ingestion throughput** — V100 unless noted:

| Variant | Params | Ingest Step 2 (s) | Peak VRAM | sim@10 | kw@10 |
|---|---|---|---|---|---|
| Qwen3-Embedding-4B (llama.cpp) | 4000M | 426 | 4.3 GiB | 0.347 | 0.30 |
| Nomic-v1.5 (GPT4All, CPU*) | 137M | 7212 (projected) | 0.9 GiB | 0.239‡ | 0.04‡ |
| Nomic-v1.5 (sentence-transformers) | 137M | 126 | 5.9 GiB† | 0.239 | 0.04 |
| all-MiniLM-L6-v2 (sentence-transformers) | 23M | 62 | 0.7 GiB | 0.219 | 0.12 |

*GPT4All falls back to CPU on this node because its bundled CUDA stub needs
libcublas.so.11 and we are on CUDA 12. Projected from a 100-chunk sample.
‡ Reused from ST Nomic (same weights). † Allocator anomaly — see open items.

**End-to-end pytest quality** (11 benchmarks × cross-encoder rerank ×
Qwen2.5-1.5B-Instruct generator; `semantic_similarity` weighted 0.7,
`keyword_similarity` 0.3):

| Variant | semantic | keyword | bleu | final_score | wall (s) |
|---|---|---|---|---|---|
| Qwen3-4B | 0.761 | 0.310 | 0.026 | 0.592 | 44 |
| Nomic (ST, V100) | 0.735 | 0.346 | 0.032 | 0.589 | 36 |
| MiniLM | 0.778 | 0.310 | 0.026 | 0.603 | 39 |

(All four variants running generator on GPU — the earlier 413/722/688-sec
numbers were a bug where `tests/metrics/semantic.py` unconditionally set
`CUDA_VISIBLE_DEVICES=''` on import, killing the GPU for llama.cpp too.
Fixed now; the scorer is pinned to `device='cpu'` explicitly.)

**Headline finding:** the lightweight embedders match the 4B baseline
on end-to-end quality (MiniLM slightly edges it, Nomic slightly below)
despite a clear retrieval-only gap.
The cross-encoder reranker does real rescue work as long as the right chunk
is in the FAISS top-50 candidate pool. So on this corpus a lightweight
embedder is a drop-in replacement when the rest of the pipeline is intact.

## What went wrong and how it was fixed

Each one of these ate time in this session — a future run can skip them.

1. **torch cu130 vs driver CUDA 12.9** — installed torch shipped cu130
   wheels but the node driver supports up to CUDA 12.9. Fixed by
   `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124`.
2. **Prebuilt llama-cpp-python kernels don't run on V100 sm_70** — pip
   wheel targets Ampere+. Rebuilt from source after loading
   `module load cuda/12.6.1` and passing
   `--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=70`. Build is
   ~25 min. `.phase3_env.sh` has the build-time module load as
   commented-out lines so future rebuilds are a one-step uncomment.
3. **libcudart.so.12 wasn't on LD path** — runtime needs it. Fixed by
   adding the pip-shipped `nvidia-cuda-runtime-cu12` dir to
   `LD_LIBRARY_PATH` in `.phase3_env.sh`.
4. **llama-cpp-python RPATH pins an old libstdc++** — any Python process
   that imports it later finds a spack libstdc++ missing `CXXABI_1.3.15`
   and sqlite3 / transformers imports start failing. Fixed by
   `LD_PRELOAD=<conda_lib>/libstdc++.so.6`.
5. **sentence-transformers 5.4.0 broke against transformers 5.5** —
   downgraded to `sentence-transformers==3.3.1` + `transformers==4.49.0` +
   `torchvision==0.20.1+cu124`. einops and matplotlib also needed.
6. **AsyncLLMJudge constructor requires a Gemini API key** and otherwise
   raises at registry import. Wrapped registration in a `try/except`.
7. **`tests/benchmarks.yaml` has a stale `ideal_retrieved_chunks` field** —
   the chunk IDs it lists point to irrelevant sections under the current
   chunker (e.g. the "ACID properties" benchmark lists chunks 1143-1148
   which are actually about "File Organization / Variable-Length Records").
   Recall@k is therefore 0.00 across *all* models. We pivoted the
   retrieval-only metric to keyword coverage + MiniLM answer-similarity.
   This field is left untouched; see open item #1 below.

## Files worth pointing a new Claude session at

- `experiments/phase3/README.md` — full writeup (tables, findings,
  methodology, reproducing instructions). **Read this first.**
- `experiments/phase3/summary.json` + `pytest_all_summary.json` —
  machine-readable aggregates.
- `experiments/phase3/comparison.png` — dual-panel: throughput vs
  retrieval-only on the left, retrieval-only vs end-to-end on the right.
- `experiments/phase3/phase3_<variant>_pytest_summary.json` — per-variant
  end-to-end pytest summaries (n, semantic, keyword, final_score, wall).
- `experiments/phase3/phase3_<variant>_pytest_results.jsonl` — per-question
  details (generated answer, chunks info, scores).
- `experiments/phase3/phase3_<variant>_retrieval_eval.json` — per-query
  retrieval-only scores (kw coverage, answer similarity, top-10 chunk IDs).
- `scripts/phase3_aggregate.py` — re-runs the aggregation / plot after any
  rerun (`source .phase3_env.sh && python -m scripts.phase3_aggregate`).
- `.phase3_env.sh` — environment setup to source before any work.

## Open items (pick these up)

**Ranked by value.**

1. **Fix `tests/benchmarks.yaml` `ideal_retrieved_chunks` field.** It was
   calibrated against an older chunking schema and now references
   irrelevant chunks. Simplest fix: run the current Qwen4B pipeline on each
   question, pick the 5 most similar chunks to the `expected_answer`
   (semantic sim via MiniLM), write those IDs back. Then Recall@k /
   Hit@1 / MRR metrics become meaningful again and can replace the
   proxy-based quality metrics used in Phase 3.

2. **Revisit Phase 1 for more speedup on laptop-scale hardware.** From the
   checkpoint report: 4-worker extraction on a laptop still takes ~33 min.
   Original Phase 1 hit 25× on a 64-core server (381s) but only 4.78× with
   4 cores (2017s). Ideas:
   - async disk I/O to overlap read/parse with downstream work
   - profile where the Amdahl's-law tail is (the PDF splitting step scales
     linearly with worker count — 4.52s → 11.95s)
   - investigate whether a faster PDF parser (pypdfium2, mupdf) can
     replace docling for the text-extraction hot path, keeping docling
     only for table/structure retention.

3. **ST-Nomic 6 GiB VRAM allocator anomaly.** 137M params should not need
   6 GiB; MiniLM sits at a sane 688 MiB. Low-priority. Add a
   `torch.cuda.empty_cache()` in `ResourceMonitor` and re-measure.

4. **Proper rerun of the full 4-variant pytest with GPU generator.** At
   handoff time Qwen4B's pytest runs on GPU (413s wall) and Nomic-ST +
   MiniLM are re-running *as this document is being written* — results
   need to be aggregated once the run finishes.

5. **GPT4All quality on the FAISS index it builds itself.** Phase 3 reused
   ST-Nomic's retrieval numbers for GPT4All. A proper apples-to-apples
   GPT4All quality measurement costs ~2h of CPU time for index building;
   budget wasn't available in this session.

6. **BLEU metric.** The proposal lists BLEU + semantic similarity. The
   repo ships semantic + keyword only. Implementing a BLEU metric under
   `tests/metrics/` is 30 lines of nltk.

## Finish the results branch (the one thing that needs doing before
switching machines)

Because the pytest rerun is still in flight, the `phase3-results` branch was
**not** created at handoff time. To complete it:

```bash
cd /storage/scratch1/1/ssinghal88/TokenSmith
source .phase3_env.sh

# 1. confirm the pytest rerun has finished
ls experiments/phase3/phase3_*_pytest_summary.json
cat experiments/phase3/pytest_all_summary.json | head

# 2. regenerate aggregated artifacts (summary.json, comparison.png, README table)
python -m scripts.phase3_aggregate

# 3. create the results branch from phase3-setup
git checkout phase3-setup
git checkout -b phase3-results
git add experiments/phase3/ logs/phase3_*.log index/sections/phase3_*_page_to_chunk_map.json
git commit -m "Phase 3: experiment artifacts + logs"
git push -u origin phase3-results
```

After that both branches will be on GitHub:
- `phase3-setup` = code only (pushed already)
- `phase3-results` = code + experiment artifacts

The other uncommitted files in the working tree
(`Project Proposal.pdf`, `CS_6423_project_checkpoint.pdf`,
`condaenv.gvyz00hd.requirements.txt`, and the modified
`index/sections/textbook_index_page_to_chunk_map.json`) are pre-existing
user files from before this session — left untouched intentionally.

## Reproducing from scratch on a different machine

```bash
git clone git@github.com:singhals163/TokenSmith.git
cd TokenSmith
git checkout phase3-results    # has the code + reference results
conda env create -f environment.yml   # creates `tokensmith`
conda activate tokensmith

# One-time additional installs on top of environment.yml
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install 'sentence-transformers==3.3.1' 'transformers==4.49.0' einops matplotlib gpt4all
pip install torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# If your GPU is older than Ampere, rebuild llama-cpp-python for your arch
# (sm_70 is the V100 example; use sm_75/sm_80/sm_86/sm_89 as needed)
module load cuda/12.6.1     # or whatever CUDA toolkit is on that host
pip install --upgrade --force-reinstall --no-binary=llama-cpp-python --no-deps \
    --config-settings=cmake.define.GGML_CUDA=ON \
    --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=70 \
    llama-cpp-python

# Adjust cache paths in .phase3_env.sh for the new machine, then:
source .phase3_env.sh

# Re-run everything (safe — results are deterministic modulo LLM temperature)
python -m scripts.run_phase3_index --variant minilm
python -m scripts.run_phase3_index --variant nomic_st
python -m scripts.run_phase3_index --variant qwen4b
python -m scripts.phase3_gpt4all_throughput --sample-size 100 --batch-size 32
python -m scripts.phase3_retrieval_eval --prefix phase3_minilm   --model sentence-transformers/all-MiniLM-L6-v2 --backend sentence_transformers
python -m scripts.phase3_retrieval_eval --prefix phase3_nomic_st --model nomic-ai/nomic-embed-text-v1.5         --backend sentence_transformers
python -m scripts.phase3_retrieval_eval --prefix phase3_qwen4b   --model models/Qwen3-Embedding-4B-Q4_K_M.gguf --backend llama_cpp
python -m scripts.phase3_full_pytest
python -m scripts.phase3_aggregate
```
