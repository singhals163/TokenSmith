# TokenSmith — RTX 6000 (sm_75) Experiment Plan

**Written 2026-04-24. Read this top-to-bottom before running anything.**

This document is a self-contained handoff for a fresh Claude Code session that
will run Phases 1, 2, and 3 of TokenSmith on a PACE Phoenix node allocated on
the `gpu-rtx6000` partition. Goal: produce a consumer-class-GPU comparison row
for the final writeup (deadline 2026-04-25). The RTX 6000 (Turing, 24 GB,
sm_75) is the closest PACE proxy for a gaming-laptop-class GPU (RTX 2080/3060
mobile class).

> **Sibling doc:** `HANDOFF.md` — end-of-V100-session state. Reference it for
> background; this file supersedes it for anything RTX-6000-specific.

## 0. TL;DR for a new Claude session

1. You are on a PACE Phoenix compute node, `gpu-rtx6000` partition, allocated
   for 4 hours. 1 × Quadro RTX 6000 (Turing sm_75, 24 GB). 4 CPUs. 32 GB RAM.
2. The repo is at `/storage/scratch1/1/ssinghal88/TokenSmith` (same path as the
   login node — this is scratch, shared across every compute node on this
   user, so **do not `git clone`**; `cd` directly into it). `conda activate
   tokensmith` works. Home partition is 87% full — **all new downloads,
   installs, caches, and tmp files must go to scratch** (see §2.0).
3. There are three experiments to run, one per phase. Run them in order
   **P1 → P1v2 → P2 → P3**; each phase has a dedicated branch.
4. Every experiment logs into `experiments/phase*/` and `logs/`. Commit results
   to the per-phase `*-rtx6000` branch (create from the phase's head). **Do not
   merge across branches.**
5. Before any GPU code runs, you must rebuild `llama-cpp-python` for sm_75
   (once, ~25 min). See §2.2.
6. Memory-restricted runs (simulating an 8 GB laptop GPU) use a helper
   `tools/cap_vram.py` described in §5.
7. If you run short on the 4-hour allocation, prioritize:
   Phase 3 minilm+nomic_st > Phase 1-v2 > Phase 2 > Phase 3 qwen4b > Phase 3 gpt4all > Phase 1 baseline.

## 1. Repository layout and branches

```
/storage/scratch1/1/ssinghal88/TokenSmith
├── src/                    # production code (embedder, index_builder, preprocessing, generator)
├── scripts/                # experiment drivers, one per phase
│   ├── run_phase3_index.py
│   ├── phase3_full_pytest.py
│   ├── phase3_retrieval_eval.py
│   ├── phase3_gpt4all_throughput.py
│   ├── phase3_aggregate.py
│   └── phase3_fix_benchmarks.py
├── tests/                  # pytest harness; tests/benchmarks.yaml drives eval
├── experiments/
│   ├── phase1/             # workers_{1,2,4,8,16,32,62} extraction timing
│   ├── phase1_v2/          # pypdfium2 vs docling (on phase1-v2-setup branch)
│   ├── phase2/
│   └── phase3/             # 4-variant comparison (qwen4b/nomic_st/minilm/gpt4all)
├── index/sections/         # built FAISS/BM25 artifacts, one set per --index-prefix
├── models/                 # GGUFs (Qwen3-Embedding-4B + Qwen2.5-1.5B-Instruct)
├── data/
│   ├── chapters/textbook.pdf                        # 2195-page Silberschatz
│   └── textbook--extracted_markdown.md              # docling-extracted markdown
├── config/config.yaml
├── tokensmith env lives at /storage/home/hcoda1/1/ssinghal88/scratch/.conda/envs/tokensmith
└── .phase3_env.sh          # MUST source before running anything (see §2.1)
```

### Branch map

| Phase | Branch | What's on it | Push status |
|---|---|---|---|
| P1 baseline | `phase1` | docling ProcessPoolExecutor multiproc extractor | origin |
| P1 fast-path | `phase1-v2-setup` | + `process_pdf_range_pypdfium` in `extraction.py`, `--parser pypdfium` CLI flag, `qwen4b_pypdfium` variant in `run_phase3_index.py`, `experiments/phase1_v2/` results | **local only** — push before switching machines if you haven't already |
| P2 | `phase2` | Qwen3-4B GPU/CPU dynamic routing + batch sorting | origin |
| P3 | `phase3-results` | pluggable embedder backends, GPT4All CPU+GPU, BLEU metric, regenerated `ideal_retrieved_chunks`, 4-variant artifacts | origin |

**Working rule**: `git checkout <phase-branch>` before running that phase's
scripts. After the run, create `<phase-branch>-rtx6000`, commit artifacts there,
push it. Keep each branch's result set separate — this is the structure the
writeup expects.

## 2. Environment setup on the RTX 6000 node

### 2.0. CRITICAL: all disk writes go to scratch, NEVER home

PACE home is quota-capped at 20 GB (current use: ~17 GB, 87% full). Any
download, pip install, model cache, HF cache, pytest artifact, or
intermediate file **must land on scratch**. The environment script already
redirects the obvious caches; the rules for anything new are:

- **Model downloads / HF datasets**: controlled by `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `XDG_CACHE_HOME` — all set to `/storage/scratch1/1/ssinghal88/cache/...` by `.phase3_env.sh`. Do not override.
- **pip installs**: always go through the conda env `tokensmith`, which itself lives at `/storage/home/hcoda1/1/ssinghal88/scratch/.conda/envs/tokensmith` — i.e. on scratch. `pip install <x>` without `--user` is safe. `PIP_CACHE_DIR` is already redirected to scratch.
- **GPT4All model cache**: `GPT4ALL_MODEL_PATH=/storage/scratch1/1/ssinghal88/cache/gpt4all` — already set.
- **Experiment artifacts**: written to `experiments/phase*/` inside the repo, which is on scratch (`/storage/scratch1/...`).
- **New tmp files** you create ad-hoc: put them under `/storage/scratch1/1/ssinghal88/tmp/` (`mkdir -p` first), or `$TMPDIR` if set. Never `/tmp/$USER/...` on a compute node, and never `~/tmp`.
- **If a tool writes to `~/.cache/<something>` anyway**, symlink or move it to scratch before the run (`ln -s /storage/scratch1/1/ssinghal88/cache/<x> ~/.cache/<x>`).

Sanity check before any heavy step:

```bash
# Should print <87% free; if home is tight, something is leaking
pace-quota 2>&1 | grep -E "Home|scratch1"
df -h /storage/home/hcoda1/1/ssinghal88 /storage/scratch1/1/ssinghal88 2>&1
```

If home-usage climbs during a run, stop and find the leak before continuing —
hitting the home quota mid-index-build will kill the job halfway through with
a `[Errno 122] Disk quota exceeded`.

### 2.1. One-liner to source — every shell in this allocation

```bash
cd /storage/scratch1/1/ssinghal88/TokenSmith
conda activate tokensmith
source .phase3_env.sh    # HF_HOME, LD_PRELOAD, LD_LIBRARY_PATH, cache redirects
```

The `.phase3_env.sh` you will read on disk is pinned to CUDA 12 runtime libs
from the conda env's `nvidia-cuda-runtime-cu12` / `nvidia-cublas-cu12` pip
wheels. That is correct for Turing (sm_75) too — **do not edit it**.

**No need to clone the repo.** The scratch filesystem
(`/storage/scratch1/1/ssinghal88/`) is shared across every compute node on
this user, so `/storage/scratch1/1/ssinghal88/TokenSmith` is the same
working tree the login-node session used. Just `cd` in and go.

### 2.2. Rebuild llama-cpp-python for sm_75

Pip's prebuilt llama-cpp-python wheel only ships Ampere+ kernels. It runs on
the RTX 6000 CPU-only unless rebuilt. Do this once per allocation:

```bash
module load cuda/12.6.1
export CUDACXX=$(which nvcc)

pip install --upgrade --force-reinstall --no-binary=llama-cpp-python --no-deps --verbose \
    --config-settings=cmake.define.GGML_CUDA=ON \
    --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=75 \
    llama-cpp-python
```

~25 minutes. Verify:

```bash
python -c "from llama_cpp import Llama; m = Llama('models/Qwen3-Embedding-4B-Q4_K_M.gguf', n_gpu_layers=-1, n_ctx=512, embedding=True, verbose=False); print(m.embed(['ok'])[0][:3])"
```

If that prints 3 floats without `CUDA error: no kernel image...`, the rebuild
worked.

After the rebuild, the `LD_PRELOAD=.../libstdc++.so.6` in `.phase3_env.sh`
becomes load-bearing again (libllama.so's RPATH pins an older spack libstdc++;
without the preload, `sqlite3` and `transformers` imports break later).

### 2.3. Sanity-check the GPU

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
# Expect: Quadro RTX 6000, <some CUDA 12 driver>, 24576 MiB, 7.5
```

## 3. Memory restriction — simulating an 8 GB laptop GPU

On a 24 GB RTX 6000, the honest "laptop" comparison forces the process to use
≤ 8 GiB of VRAM. Two mechanisms, applied together:

### 3.1. PyTorch side — `torch.cuda.set_per_process_memory_fraction`

Before importing any sentence-transformers / torch model, set:

```python
import torch
torch.cuda.set_per_process_memory_fraction(8/24, device=0)  # 8 GiB cap
```

A helper module does this consistently across scripts. **Create it** at the
start of the session:

```python
# tools/cap_vram.py
"""Import this before loading any torch model to cap VRAM to ~8 GiB.
Controlled by env var VRAM_CAP_GIB; default 8.0. Set VRAM_CAP_GIB=0 to disable.
"""
import os, torch

def apply():
    cap = float(os.environ.get("VRAM_CAP_GIB", "8"))
    if cap <= 0 or not torch.cuda.is_available():
        return
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    frac = min(1.0, cap / total_gib)
    torch.cuda.set_per_process_memory_fraction(frac, device=0)
    print(f"[cap_vram] fraction={frac:.3f} target={cap:.1f}GiB / total={total_gib:.1f}GiB")

if os.environ.get("VRAM_CAP_GIB_AUTO", "0") == "1":
    apply()
```

Invoke by exporting `VRAM_CAP_GIB=8 VRAM_CAP_GIB_AUTO=1` and adding
`import tools.cap_vram` at the top of each script that loads torch. Or, simpler
— run each experiment twice: once uncapped, once with `VRAM_CAP_GIB=8`, and
report both.

### 3.2. llama.cpp side — `n_gpu_layers`

`llama-cpp-python` does not honor the torch allocator. For Qwen3-Embedding-4B
(Q4_K_M), offloading all layers uses ~4.3 GiB (fits within 8). For Qwen2.5-1.5B
generator (Q5_K_M), all layers ≈ 1.6 GiB. Both fit the cap; leave
`n_gpu_layers=-1` **but** reduce `n_batch` to 256 and `n_ctx` to 4096 (down
from 8192) if memory pressure shows up.

### 3.3. Verifying the cap was honored

`src/instrumentation/resource_monitor.py` already samples peak VRAM via
`nvidia-smi` during the ingestion window and writes it into
`phase3_<variant>_resources.json`. Read `peak_vram_mib` from that JSON — if it
is ≤ 8192 MiB, the cap was honored for that run. If a sentence-transformers
run blows past 8 GiB (the known Nomic-ST allocator anomaly on V100 was 6 GiB
uncapped → should still be fine at 8 GiB, but MiniLM should sit at ~0.7 GiB),
halve the encode `batch_size` to 16 and retry.

## 4. Experiments

### 4.1. Phase 1 — baseline docling multiproc extraction

**Branch:** `phase1` (do not use the phase1-v2 fast path here)

**Purpose:** confirm extraction scaling on the RTX 6000 node's CPU (host has
only 4 cores in this allocation). The original Phase 1 reported 25.28× on a
64-core server, 4.78× on 4 cores. Point of this rerun is to record a fresh
4-core number on the same PDF for the writeup's hardware-agnostic story.

```bash
git checkout phase1
source .phase3_env.sh

# The existing profile-extraction.sh sweeps 1/2/4 workers. 4 cores = 4 max.
# Edit profile_extraction.sh if needed to cap at 4 workers. Run it; each
# setting writes to experiments/phase1/workers_<N>_chunk_<C>/.
bash profile_extraction.sh 2>&1 | tee logs/phase1_rtx6000.log

# Collect
ls experiments/phase1/workers_*_chunk_*/profiling.txt
```

Expected wall for 4 workers on Silberschatz-2195pp: ~30 min (matches the
checkpoint report's 4-core laptop result of ~33 min).

**Result branch:** `git checkout -b phase1-rtx6000`, commit the new
`experiments/phase1/workers_*` directories + log, push.

### 4.2. Phase 1-v2 — pypdfium2 fast-path extraction

**Branch:** `phase1-v2-setup` (push it first if still local-only)

**Purpose:** demonstrate the docling → pypdfium2 swap on the RTX 6000 / 4-core
host. Expect ~12–15 s (vs V100-host 12.9 s). Also re-ingest with Qwen4B on the
RTX 6000 GPU and compute end-to-end quality to show the fast-path markdown
doesn't degrade quality.

```bash
git checkout phase1-v2-setup
source .phase3_env.sh

# 1. Fast extraction (2195 pages, 4 workers via ProcessPoolExecutor)
python -m src.preprocessing.extraction \
    --pdf_dir data/chapters \
    --parser pypdfium \
    --output_markdown data/textbook--pypdfium_markdown.md \
    --workers 4 2>&1 | tee logs/phase1v2_rtx6000_extract.log

# 2. Re-index with the Qwen4B embedder against the pypdfium markdown
python -m scripts.run_phase3_index --variant qwen4b_pypdfium \
    --markdown data/textbook--pypdfium_markdown.md \
    2>&1 | tee logs/phase1v2_rtx6000_index.log

# 3. Retrieval-only eval
python -m scripts.phase3_retrieval_eval \
    --prefix phase1v2_pypdfium_qwen4b \
    --model models/Qwen3-Embedding-4B-Q4_K_M.gguf \
    --backend llama_cpp \
    2>&1 | tee logs/phase1v2_rtx6000_retrieval.log

# 4. End-to-end pytest
python -m scripts.phase3_full_pytest --only qwen4b_pypdfium \
    2>&1 | tee logs/phase1v2_rtx6000_pytest.log
```

**What to report:** extraction wall time, index wall time, `kw@10`, `sim@10`,
end-to-end `semantic_similarity`, `keyword_similarity`, `bleu`, `final_score`.
Compare side-by-side with the V100 phase1-v2 numbers in
`experiments/phase1_v2/summary.json`.

**Result branch:** `phase1-v2-rtx6000`.

### 4.3. Phase 2 — Qwen3-4B GPU/CPU dynamic routing

**Branch:** `phase2`

**Purpose:** re-run the Phase 2 ingestion (FAISS + BM25 build over 2664 chunks
with Qwen3-Embedding-4B on the RTX 6000). Compare to the A100 Phase-2 baseline
of 305 s Step 2 wall and the V100 Phase-3 baseline of 426 s. RTX 6000 is
expected to land between the two (Turing fp16 throughput ≈ 0.5× V100 fp16 but
the batch-sorting step helps). Also demonstrate the dynamic router still
picks GPU on this node.

```bash
git checkout phase2
source .phase3_env.sh

# Re-run the production indexing path on the extracted markdown
python -m src.main index \
    --config config/config.yaml \
    --index_prefix phase2_rtx6000_qwen4b \
    2>&1 | tee logs/phase2_rtx6000_index.log

# Read profiling from src/profiler output
cat experiments/phase2/profiling_indexing.txt   # updated in-place; diff vs git HEAD
```

**Alternative path** if Phase 2's main.py flow is fragile under the current
env: reuse the Phase-3 driver on the `phase2` branch — it exercises the same
`build_index` code path with more instrumentation:

```bash
python -m scripts.run_phase3_index --variant qwen4b \
    --profile-dir experiments/phase2/rtx6000 \
    2>&1 | tee logs/phase2_rtx6000_index.log
```

**Memory-cap variant** (this is the headline data point for Phase 2 on
laptop-class hardware):

```bash
VRAM_CAP_GIB=8 VRAM_CAP_GIB_AUTO=1 \
    python -m scripts.run_phase3_index --variant qwen4b \
    --profile-dir experiments/phase2/rtx6000_8gib \
    2>&1 | tee logs/phase2_rtx6000_8gib.log
```

**What to report:** Step 2 (embedding) wall time, peak VRAM at 8 GiB cap vs
uncapped, routing decisions (from the log).

**Result branch:** `phase2-rtx6000`.

### 4.4. Phase 3 — lightweight embedder comparison (the main event)

**Branch:** `phase3-results`

**Purpose:** reproduce the 4-variant comparison on RTX 6000, both uncapped
(full 24 GB) and capped (8 GiB "laptop" simulation).

```bash
git checkout phase3-results
source .phase3_env.sh
git checkout -b phase3-rtx6000    # working branch for RTX 6000 artifacts
```

#### 4.4.1. Build indexes (4 variants)

Each takes 1–8 min on GPU; minilm < nomic_st < qwen4b ≈ nomic_gpt4all (GPU).

```bash
python -m scripts.run_phase3_index --variant minilm \
    --profile-dir experiments/phase3/rtx6000 \
    2>&1 | tee logs/phase3_rtx6000_index_minilm.log

python -m scripts.run_phase3_index --variant nomic_st \
    --profile-dir experiments/phase3/rtx6000 \
    2>&1 | tee logs/phase3_rtx6000_index_nomic_st.log

python -m scripts.run_phase3_index --variant qwen4b \
    --profile-dir experiments/phase3/rtx6000 \
    2>&1 | tee logs/phase3_rtx6000_index_qwen4b.log

# GPT4All GPU — requires CUDA 11 stub already loaded by the env script on
# rebuild; if libcublas.so.11 is missing on this node, fall back to
# --variant nomic_gpt4all (CPU) with a 100-chunk sample instead of the full corpus.
python -m scripts.run_phase3_index --variant nomic_gpt4all_gpu \
    --profile-dir experiments/phase3/rtx6000 \
    2>&1 | tee logs/phase3_rtx6000_index_gpt4all_gpu.log || \
python -m scripts.phase3_gpt4all_throughput --sample-size 100 --batch-size 32 \
    --device cpu --prefix phase3_nomic_gpt4all_rtx6000 \
    2>&1 | tee logs/phase3_rtx6000_gpt4all_cpu_sample.log
```

#### 4.4.2. Retrieval-only eval (no generator)

```bash
for V in "phase3_minilm:sentence-transformers/all-MiniLM-L6-v2:sentence_transformers" \
         "phase3_nomic_st:nomic-ai/nomic-embed-text-v1.5:sentence_transformers" \
         "phase3_qwen4b:models/Qwen3-Embedding-4B-Q4_K_M.gguf:llama_cpp"; do
    IFS=: read prefix model backend <<< "$V"
    python -m scripts.phase3_retrieval_eval \
        --prefix "$prefix" --model "$model" --backend "$backend" \
        --out-dir experiments/phase3/rtx6000 \
        2>&1 | tee "logs/phase3_rtx6000_retrieval_${prefix}.log"
done
```

#### 4.4.3. End-to-end pytest

Full pipeline: FAISS retrieval → cross-encoder rerank (top 5) → Qwen2.5-1.5B
generation (on GPU) → metrics (`semantic` + `keyword` + `bleu`).

```bash
python -m scripts.phase3_full_pytest 2>&1 | tee logs/phase3_rtx6000_pytest.log
# Edit OUT_DIR inside phase3_full_pytest.py if you want artifacts under
# experiments/phase3/rtx6000/ rather than experiments/phase3/ — simplest is to
# copy the JSONs out after the run:
mkdir -p experiments/phase3/rtx6000
cp experiments/phase3/phase3_*_pytest_*.{json,yaml} experiments/phase3/rtx6000/ 2>/dev/null
cp experiments/phase3/pytest_all_summary.json experiments/phase3/rtx6000/ 2>/dev/null
```

#### 4.4.4. Memory-capped rerun (8 GiB laptop simulation)

```bash
VRAM_CAP_GIB=8 VRAM_CAP_GIB_AUTO=1 \
    python -m scripts.phase3_full_pytest 2>&1 | tee logs/phase3_rtx6000_pytest_8gib.log

# Collect 8 GiB-capped artifacts under rtx6000_8gib/
mkdir -p experiments/phase3/rtx6000_8gib
cp experiments/phase3/phase3_*_pytest_*.{json,yaml} experiments/phase3/rtx6000_8gib/
cp experiments/phase3/pytest_all_summary.json experiments/phase3/rtx6000_8gib/
```

#### 4.4.5. Aggregate

```bash
python -m scripts.phase3_aggregate 2>&1 | tee logs/phase3_rtx6000_aggregate.log
# Writes summary.json + comparison.png at experiments/phase3/
```

**What to report:**

| | uncapped (24 GiB) | capped (8 GiB) | V100 baseline |
|---|---|---|---|
| variant × {ingest_s, peak_vram_mib, kw@10, sim@10, final_score, wall_s} | from `experiments/phase3/rtx6000/` | from `experiments/phase3/rtx6000_8gib/` | from `experiments/phase3/` (already committed, pre-session) |

**Result branch:** `phase3-rtx6000`. Commit the new `rtx6000/` and
`rtx6000_8gib/` subdirectories plus the logs; push.

## 5. Pitfalls — things that will eat your wall clock

These are the sharp edges that cost time on the V100 session. The new
environment is close to identical, so expect the same ones.

1. **sm_75 rebuild step is ~25 min** — kick it off first so you can do other
   prep in parallel (e.g. read this doc, check `nvidia-smi`, check git state).
2. **Do not pip-install anything on top of the conda env.** The version pins
   (torch==2.5.1+cu124, sentence-transformers==3.3.1, transformers==4.49.0,
   einops, matplotlib, gpt4all, torchvision==0.20.1+cu124) are known-good.
3. **`LD_PRELOAD=.../libstdc++.so.6` is load-bearing.** Any subshell that does
   not `source .phase3_env.sh` and then imports llama-cpp-python will crash
   sqlite3 or transformers on the next import.
4. **Stale `ideal_retrieved_chunks`.** On `phase3-results` the field has been
   regenerated and is correct. On `phase2` and earlier branches it is the old
   stale annotation. The Phase-3 retrieval-eval script already uses MiniLM
   cosine similarity proxies as its *primary* metric, so Recall@k can be
   ignored on those branches.
5. **GPU pollution via `tests/metrics/semantic.py`.** Fixed on
   `phase3-results`: the scorer is pinned to `device='cpu'`. If you switch to
   `phase1` or `phase2`, do NOT `import tests.metrics.semantic` in a process
   where you also want the GPU — on those branches it still clobbers
   `CUDA_VISIBLE_DEVICES` on import.
6. **GPT4All CUDA 11 libs** may not be reachable on this node if `module load
   cuda/11.8.0` isn't available. If so, GPT4All silently falls back to CPU
   (~2.7 s/chunk). The fallback script
   `scripts/phase3_gpt4all_throughput.py --device cpu --sample-size 100`
   will still produce a projected full-corpus throughput in ~5 min.
7. **`n_gpu_layers=-1`** on llama-cpp-python means "offload everything". For
   Qwen4B-Q4_K_M this is ~4.3 GiB VRAM on V100; on RTX 6000 expect a similar
   footprint. If the 8-GiB-cap run reports OOM mid-embed, drop n_batch to 128
   in `src/embedder.py` `LlamaCppEmbedder`.
8. **`TOKENIZERS_PARALLELISM`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`** are
   pinned to 1 in `src/index_builder.py` to avoid oversubscribing the 4-core
   host. Do not override.
9. **Cold vs warm wall times differ by ~5 min** on the first Qwen4B load
   (GGUF page-cache). Always throw away the first run's wall-time number.
10. **`conda activate tokensmith`** — this env lives at
    `/storage/home/hcoda1/1/ssinghal88/scratch/.conda/envs/tokensmith`, which
    is the user's scratch conda install. If `conda activate` fails, run
    `source /storage/home/hcoda1/1/ssinghal88/scratch/.conda/etc/profile.d/conda.sh`
    first.
11. **Never write to `~`.** Home is 87% full — a stray model download to
    `~/.cache/huggingface/` or a `pip install --user` will kill the job with
    `Disk quota exceeded`. See §2.0; if you're about to download something
    bulky, double-check where it will land first.

## 6. Final writeup — what to hand back

After all four experiments finish, produce one consolidated file at
`experiments/rtx6000_summary.md` with these sections:

1. **Hardware context:** 1 × RTX 6000 Turing (sm_75, 24 GB), 4 cores, 32 GB.
2. **Phase 1 (docling, 4 workers):** wall-time table.
3. **Phase 1-v2 (pypdfium, 4 workers):** wall-time side-by-side vs docling, end-to-end quality.
4. **Phase 2 (Qwen4B ingest on RTX 6000, uncapped):** Step-2 wall, peak VRAM, comparison to V100 (426 s) and A100 (305 s).
5. **Phase 3 — uncapped, 4 variants:** full results table, matches the V100 structure.
6. **Phase 3 — 8 GiB-capped (laptop proxy), 4 variants:** same table, with peak-VRAM column confirming the cap held.
7. **Headline:** two sentences on whether the "lightweight embedders + 8 GiB GPU" story holds up on consumer silicon.

Commit each phase's results on its own `*-rtx6000` branch. Write the summary
on `phase3-rtx6000` (the most inclusive of the four). Push all four branches
to origin. Do not merge them.

## 7. Reproduction checklist (tick as you go)

- [ ] `nvidia-smi` shows 1 × Quadro RTX 6000, sm_75, 24 GB
- [ ] `conda activate tokensmith` succeeds
- [ ] `source .phase3_env.sh` — no errors
- [ ] `python -c "from llama_cpp import Llama; print('ok')"` — no errors
- [ ] llama-cpp-python rebuilt for sm_75, embedding smoke test returns floats
- [ ] `tools/cap_vram.py` created and importable
- [ ] `git checkout phase1-v2-setup` — pushed to origin (check `git log @{u}..`)
- [ ] Phase 1 baseline — `experiments/phase1/workers_*` populated
- [ ] Phase 1-v2 — `experiments/phase1_v2/` RTX 6000 artifacts committed on `phase1-v2-rtx6000`
- [ ] Phase 2 — `experiments/phase2/rtx6000/` committed on `phase2-rtx6000`
- [ ] Phase 3 uncapped — `experiments/phase3/rtx6000/` populated
- [ ] Phase 3 capped — `experiments/phase3/rtx6000_8gib/` populated, `peak_vram_mib ≤ 8192`
- [ ] `experiments/rtx6000_summary.md` written and committed
- [ ] All `*-rtx6000` branches pushed to origin

## 8. Open questions to decide as you go

1. **GPT4All-GPU on the RTX 6000 node.** On V100 it needed `module load cuda/11.8.0` + `libcublas.so.11`. If that CUDA 11 module isn't available on the RTX 6000 allocation (likely not), document the fallback to CPU sample and move on — do not sink >30 min into fixing it.
2. **`config/config.yaml` defaults.** The `phase2` branch's config has
   `embed_model: "models/Qwen3-Embedding-4B-Q5_K_M.gguf"` (Q5) while the models
   dir ships Q4. Either download Q5 or edit the config to Q4 before running
   phase2's `src.main index`. The Phase-3 `run_phase3_index.py` already
   points at the Q4 GGUF and doesn't need the config edit.
3. **Which pytest run is the "headline" for the deadline writeup.** The
   `rtx6000_8gib` run is the one that supports the laptop claim. If only one
   pytest run finishes, make sure it's the capped one.

---

**Doc author:** Claude Code, 2026-04-24 session, on `phase3-results` branch.
**Next session:** fresh allocation on `gpu-rtx6000`. Open this file first,
then `HANDOFF.md` for additional background context.
