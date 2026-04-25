# tools/ + scripts/

Utility scripts and experiment drivers built alongside the production
features. They depend on the feature branches (`feat/parallel-extraction`,
`feat/embedder-router`, `feat/pluggable-embedders`) being merged in — the
`experiments` branch is the integration target where they actually run.

## Resource instrumentation

- **`src/profiler.py`** (lives on `feat/parallel-extraction`): lightweight
  `@timeit` decorator + `TimerBlock` context manager + a single
  `print_profile_stats()` writer. Used to attribute wall-time to map/reduce
  stages of the extraction and embedding pipelines.

- **`src/instrumentation/resource_monitor.py`**: background sampler that
  polls process RSS via psutil and GPU memory via `nvidia-smi` at a
  configurable interval. `with ResourceMonitor() as m: …; m.summary()`
  reports peak RSS, peak VRAM, and the delta over the workload's lifetime.

- **`tools/cap_vram.py`**: wrapper around
  `torch.cuda.set_per_process_memory_fraction` that reads `VRAM_CAP_GIB`
  from the environment and applies a hard cap (defaults to 8 GiB). Used
  for the laptop-class-GPU simulations on consumer Turing hardware.

## Experiment drivers

- **`profile_extraction.sh` / `profile_extraction_sequential.sh`**: shell
  loops that re-run the extraction step at several worker counts and dump
  per-config profiling artifacts (Phase 1 scaling studies).

- **`scripts/run_phase3_index.py`**: build one FAISS+BM25 index per
  embedding-backend variant. Dispatches into the pluggable embedder
  factory (`make_embedder`) with a per-variant model path and backend.

- **`scripts/phase3_retrieval_eval.py`**: retrieval-only eval — runs each
  query against a built index and reports keyword coverage, embedding
  similarity, and per-query embedding latency.

- **`scripts/phase3_full_pytest.py`**: end-to-end pytest harness that
  wires retrieval + cross-encoder rerank + Qwen2.5 generation + scoring,
  one variant at a time.

- **`scripts/phase3_gpt4all_throughput.py`**: standalone throughput probe
  for the GPT4All backend on a small sample of chunks; used to project
  full-corpus wall-time without paying for it.

- **`scripts/phase3_aggregate.py`**: reads each variant's per-run JSON
  artifacts and produces `experiments/phase3/summary.json` plus a
  matplotlib comparison plot.

- **`scripts/phase3_fix_benchmarks.py`**: regenerates the
  `ideal_retrieved_chunks` annotation in `tests/benchmarks.yaml` against
  the current chunker schema (the original annotation drifted).
