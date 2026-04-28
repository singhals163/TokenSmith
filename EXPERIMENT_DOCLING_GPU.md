# Experiment: docling on GPU as a fair Phase 1 v2 baseline

A reviewer is going to ask whether `pypdfium2` would still beat
`docling` if we let `docling` use the GPU it was happily ignoring.
The current report compares pypdfium2 (4 workers, CPU) against the
original sequential CPU docling and reports a `~1{,}100x` speedup. That
number is correct against what the original code actually did, but it
overstates the relative win against a properly configured docling.
This runbook tells a fresh session how to close that gap with one
clean experiment.

## 0. TL;DR for a new Claude session

1. Goal: measure docling extraction wall time on the 2{,}195-page
   Silberschatz textbook with the GPU enabled, at one and four workers.
2. Hardware: same RTX 6000 allocation profile we used in the rest of
   the project (1 x Quadro RTX 6000, 24 GiB, 4 CPU cores).
3. Code change: a single block in `src/preprocessing/extraction.py`
   that sets `accelerator_options` on the pipeline.
4. Output: one new row in the Phase 1 results table plus a note in
   the paper and slides that the 1{,}100x is against the unaccelerated
   baseline; against GPU docling the gap is roughly 5x to 10x.
5. Time budget: about 30 minutes of compute including a one-time
   model download.

## 1. Why this matters

The current Phase 1 v2 narrative reads as ``pypdfium2 is 1{,}100x
faster than docling.'' That is true on the original code path
(sequential docling, CPU-bound, layout analysis on every page). It is
not the strongest version of the docling baseline. Modern docling
versions can run their layout-analysis vision transformer on a GPU,
which on this corpus should drop wall time from minutes to seconds.

If the gap is still meaningful after that fix (we expect it is, but
roughly 5x rather than 1{,}100x), that is the honest comparison to
report. If the gap closes entirely, the Phase 1 v2 story has to be
reframed around ``no GPU needed'' rather than raw speed.

## 2. Prerequisites

### 2.1 Branch to start from

Branch from `experiments` (which has the full integrated pipeline
and the existing extraction code). Suggested name:
`feat/docling-gpu-baseline`.

```bash
git fetch origin
git checkout -b feat/docling-gpu-baseline origin/experiments
```

### 2.2 Environment

The same `tokensmith` conda env used in the rest of the project. On
PACE Phoenix:

```bash
module load anaconda3
conda activate tokensmith
# DO NOT source profile.d/conda.sh; the module-load path is the supported one.
```

### 2.3 Make sure docling supports accelerator_options

The `docling` pip wheel pinned in `environment.yml` should be at least
`docling >= 2.x`. Confirm with:

```bash
python -c "from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice; print('ok')"
```

If that import fails, upgrade in place:

```bash
pip install --upgrade 'docling>=2.0' 'docling-core'
```

(no need to touch `environment.yml`; this is a measurement-only run).

### 2.4 GPU sanity check

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
# Expect: Quadro RTX 6000, sm_75, 24576 MiB
```

`demo.sh` already prepends the conda env's pip-shipped CUDA libs to
`LD_LIBRARY_PATH`. Source it for the env vars only, or copy the
relevant block into your shell:

```bash
for sub in cuda_runtime cuda_nvrtc cublas cudnn; do
  for d in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/"$sub"/lib; do
    [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
  done
done
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
```

## 3. Code change

Open `src/preprocessing/extraction.py` and find the
`process_pdf_chunk` worker. The current pipeline_options block looks
like:

```python
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = False
```

Add an explicit accelerator block immediately after, controlled by an
env var so we can flip it on and off without editing source:

```python
import os
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice,
)

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = False

# Phase 1 v1 GPU baseline experiment.
# Set DOCLING_DEVICE=cuda to run layout analysis on GPU (default: cpu,
# matches the rest of the project's measurements).
device_name = os.environ.get("DOCLING_DEVICE", "cpu").lower()
device_map = {
    "cpu":  AcceleratorDevice.CPU,
    "cuda": AcceleratorDevice.CUDA,
    "auto": AcceleratorDevice.AUTO,
}
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=int(os.environ.get("DOCLING_THREADS", "4")),
    device=device_map.get(device_name, AcceleratorDevice.CPU),
)
```

That is the entire code change. The default behavior (no env var set)
is identical to today.

## 4. Experiments to run

All runs are on the same 2{,}195-page Silberschatz textbook used
everywhere else in the project. Each run dumps a profiling text file
that the existing extractor already produces.

### 4.1 Sequential GPU baseline (the headline number we are missing)

```bash
DOCLING_DEVICE=cuda DOCLING_THREADS=4 \
  python -m src.preprocessing.extraction \
    --workers 1 --chunk_size 2000 \
    --out_dir experiments/phase1/workers_1_chunk_2000_gpu \
    --profile_out profiling_1_workers_gpu.txt \
    2>&1 | tee logs/phase1_docling_gpu_w1.log
```

Watch `nvidia-smi` in another shell during the run; the
`docling-layout` process should hold roughly 2 to 4 GiB of VRAM. If it
does not, the env var did not take effect and the run is still on
CPU; do not record that as a GPU number.

### 4.2 Sanity check: 4 workers with GPU

Multi-process plus a single GPU is mostly counter-productive (each
worker reloads the layout model into the same GPU and they serialize
on the allocator), but record one number anyway so the writeup can
mention it.

```bash
DOCLING_DEVICE=cuda DOCLING_THREADS=4 \
  python -m src.preprocessing.extraction \
    --workers 4 --chunk_size 500 \
    --out_dir experiments/phase1/workers_4_chunk_500_gpu \
    --profile_out profiling_4_workers_gpu.txt \
    2>&1 | tee logs/phase1_docling_gpu_w4.log
```

If this OOMs the GPU, halve the model layers (see Pitfall 5) or just
report the OOM as an observation.

### 4.3 Sanity check: re-run sequential CPU on the same allocation

To compare apples to apples we want a fresh single-worker CPU number
on the same machine. The 9{,}650 s number we already have is from a
different host.

```bash
DOCLING_DEVICE=cpu DOCLING_THREADS=4 \
  python -m src.preprocessing.extraction \
    --workers 1 --chunk_size 2000 \
    --out_dir experiments/phase1/workers_1_chunk_2000_cpu_rtx \
    --profile_out profiling_1_workers_cpu_rtx.txt \
    2>&1 | tee logs/phase1_docling_cpu_w1_rtx.log
```

Optional but recommended; gives us a 1-worker CPU number on the RTX
host so the GPU comparison is on identical hardware.

## 5. What to capture

For each run, copy these into `experiments/phase1/<run>/`:

- `profiling_*.txt` (already produced by the extractor).
- The full stdout/stderr log under `experiments/phase1/logs/`.
- A one-line `summary.txt` with: device, workers, chunk_size, total
  wall in seconds, and peak VRAM observed via `nvidia-smi`.

Update `experiments/rtx6000_summary.md` with a new section ``Docling
GPU baseline'' carrying the headline 1-worker GPU number, the
4-worker GPU number, the new 1-worker CPU number on RTX, and a short
narrative comparing them to the existing pypdfium2 8.66 s.

## 6. Updating the writeup after the run

The numbers feed three places. All three should be updated in the same
PR so they stay in sync.

### 6.1 `paper/main.tex`

In Section ``Phase 1 v2: pypdfium2 Fast Path,'' add a one-line caveat
right after the ``$\mathbf{1{,}100\times}$ faster'' sentence:

> The $1{,}100\times$ is measured against the original sequential
> CPU-bound docling extractor. With docling running on the same RTX
> 6000 GPU at one worker the textbook extracts in about
> $\langle$\,X\,$\rangle$\,s; pypdfium2 still beats this by about
> $\langle$\,Y\,$\rangle\times$ at no GPU cost.

Fill in X (the new GPU docling number) and Y (8.66 / X).

### 6.2 `slides/main.tex`

In the ``Phase 1 v2: pypdfium2 fast path'' frame, add a third Pro
bullet:

```
\item No GPU needed; frees the GPU for embedding and generation
```

and append a small footnote-style block under the headline:

```
\\[0.4ex]
\footnotesize
($1{,}100\times$ is vs the original sequential CPU docling.
GPU-accelerated docling runs in about \langle X\rangle\,s; the gap is
$\sim$\,$\langle Y\rangle\times$.)
```

### 6.3 `REPORT.md`

In the Phase 1 v2 section, add a sentence right after the
``$1{,}100\times$ faster'' line acknowledging the GPU baseline and the
revised gap.

## 7. Known pitfalls

1. **Old docling version.** Some pre-2.x wheels do not honor
   `accelerator_options.device`. Verify with the import test in 2.3
   and upgrade if needed. Do not silently report a CPU number as GPU.
2. **GPU not actually engaged.** Always confirm with `nvidia-smi`
   during the run. If the GPU is idle while the script reports a
   ``GPU run,'' the env var did not take effect; the most common
   cause is `accelerator_options` being set on the wrong object or a
   stale module cache.
3. **Model download on first run.** docling pulls layout-model
   weights on first GPU use (a few hundred MiB). Run a 5-page warm-up
   first so the textbook timing does not include the download.
4. **VRAM contention with the embedder.** This experiment is
   extraction-only; the embedder is not running. If you accidentally
   chain into `src.main index` in the same process, the embedder will
   try to allocate 4 GiB of VRAM on top of docling's 2 to 4 GiB.
5. **Multi-process plus single GPU.** Each worker process loads its
   own copy of docling's models on the GPU. Four workers can OOM a
   24 GiB card; on an 8 GiB consumer card, even two workers can.
   Report the OOM behavior as an observation; do not chase a higher
   worker count just to match Phase 1 v1's 62-worker setup.
6. **CUDA visibility.** If the run starts and then complains about
   `libcudart.so.12`, the LD_LIBRARY_PATH preamble in 2.4 was not
   sourced. Source it and retry.

## 8. Open questions for the runner

If any of these come up, leave a note in the run's `summary.txt`
rather than guessing:

1. Does docling's `AcceleratorDevice.AUTO` pick CUDA on the RTX 6000
   without us setting `CUDA` explicitly? If yes, our original run
   may have used GPU for some sub-stage already.
2. Does disabling `do_ocr` and `do_table_structure` short-circuit
   most of the layout-analysis work, so the GPU mostly sits idle even
   with `device=cuda`? If yes, the speedup will be smaller than
   expected.
3. Is the extraction wall on RTX 6000 with CPU docling at 1 worker
   actually 9{,}650 s, or is the original number from a different
   host? Section 4.3 answers this.

## 9. Estimated time and outcome

- Branch + env setup: 5 minutes
- Code change: 5 minutes
- Three runs (one warm-up plus three measured): about 20 minutes total
- Writeup updates: 15 minutes

Total: about 45 minutes. Expected outcome: GPU docling on the
textbook lands somewhere in the 30 to 90 second range at 1 worker.
That moves the Phase 1 v2 gap from $1{,}100\times$ to roughly
$5\times$ to $10\times$, which is still a real win and is much harder
for a reviewer to attack.

---

**Author:** Claude Code, 2026-04-28 session, on `main` branch.\\
**Predecessor runbook:** `EXPERIMENTS_RTX6000.md` (broader, multi-phase).
