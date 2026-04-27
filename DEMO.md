# TokenSmith Demo

A live walk-through of the optimized cold path: drop a PDF, extract,
embed, ask the first question. Target wall on a 4-core / 8 GiB consumer
GPU is roughly **45 seconds** end to end.

The configuration used here is the most efficient one we measured in the
report:

- `pypdfium2` extractor (4 worker processes)
- MiniLM-L6-v2 (23 M parameters) embedder via sentence-transformers
- Qwen2.5-1.5B-Instruct GGUF generator on `llama.cpp`

## 1. One-time setup

```shell
# 1. Conda environment + llama.cpp build
make build
conda activate tokensmith

# 2. Download the generator model into models/
mkdir -p models
# Place qwen2.5-1.5b-instruct-q5_k_m.gguf in models/
# (https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
```

The MiniLM embedder downloads automatically from HuggingFace on first
use, so you do not need to fetch it manually.

## 2. Drop the PDF

```shell
mkdir -p data/chapters
cp /path/to/your-textbook.pdf data/chapters/
```

The default expected path is `data/chapters/textbook.pdf`. The demo
accepts an explicit path as the first argument.

## 3. Run the demo

```shell
./demo.sh                                  # uses data/chapters/textbook.pdf
./demo.sh data/chapters/your_book.pdf      # or pass a different PDF
```

The script runs three steps and prints one bash-measured wall time per
step (the end-to-end number, with Python startup included). What you
should see:

```
================================================================
Step 1 / 3: Extract PDF (pypdfium2, 4 workers)
================================================================
Pages              : 2195
Ranges             : 5 (workers=4)
Map-phase wall     : 3.26s
Total wall         : 8.66s
Output             : data/textbook--extracted_markdown.md

>>> wall: 9.01 s   (end-to-end, includes Python startup)

================================================================
Step 2 / 3: Build MiniLM index over the extracted markdown
================================================================
Embedding 2,664 chunks with all-MiniLM-L6-v2 ...
Hardware Router: GPU detected; using sequential high-throughput path.
...

>>> wall: 30.41 s   (end-to-end, includes Python startup)

================================================================
Cold ingestion total: 39.42 s   (extract 9.01 s + index 30.41 s)
================================================================

Step 3 / 3: Chat. Ask your first question. Ctrl+C to exit.
You: Explain the ACID properties of database transactions.
TokenSmith: ...                                            (~3 s)
```

Total cold time-to-first-answer should land near 45 seconds on a
warm consumer machine. The Python tool's own internal wall print
(`Total wall: 8.66s`) is the work after imports; the `>>> wall:` line
underneath is the bash-measured number that includes Python startup.
Take the bash number as the honest end-to-end wall.

## 4. Try these questions

The benchmark suite ships with 11 questions (see
`tests/benchmarks.yaml`). Good ones for a live demo, since they hit
clear sections of the textbook:

- `Explain the ACID properties of database transactions.`
- `What is a B+ tree index and why is it useful?`
- `How does the recovery manager use ARIES to ensure atomicity?`
- `Contrast the goals of OLTP and data analytics.`

## 5. Reset between runs

The demo writes:

- `data/textbook--extracted_markdown.md`  (extraction output)
- `index/sections/demo.faiss` and friends  (FAISS + BM25 + chunks)

Delete those if you want to re-time a clean cold start:

```shell
rm -f data/textbook--extracted_markdown.md
rm -f index/sections/demo.*
```

## Troubleshooting

### `OSError: libcudart.so.12: cannot open shared object file`

This means `llama-cpp-python` cannot find the CUDA 12 runtime libraries.
Those libraries ship inside the conda env as pip wheels under
`$CONDA_PREFIX/lib/python*/site-packages/nvidia/*/lib`, and they are not
on the default `LD_LIBRARY_PATH`.

`demo.sh` already prepends those directories automatically based on
`$CONDA_PREFIX` and pins the conda env's `libstdc++.so.6` via
`LD_PRELOAD` for good measure (some HPC images have an older system
`libstdc++` that breaks `sqlite3` / `transformers` once
`llama-cpp-python` is imported). You should not need to edit anything.

If you still see the error, confirm:

1. The conda env is activated (`echo $CONDA_PREFIX` should print the
   env path, not be empty).
2. The pip-shipped libs exist:
   `ls $CONDA_PREFIX/lib/python*/site-packages/nvidia/cuda_runtime/lib`.
   If that is empty, the env is missing the CUDA wheels; reinstall via
   `pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12`.

### Slow cold imports on shared filesystems

On PACE Lustre (or any cold network filesystem) the Python interpreter
plus its imports can take 20-30 s before any user code runs, which
inflates step 1's wall to ~30-40 s on a fresh allocation. This is
normal and reflects filesystem latency, not the project's overhead.
Re-running the demo (`./demo.sh ...`) lands closer to the warm numbers
in the report once the OS page cache is populated.

## What the timing means

The script measures wall time per step using `date +%s.%N` before and
after each command, then subtracts via `awk`. This gives one prominent
number per step that you can read out during a presentation. The number
includes Python interpreter startup, which on a cold filesystem can be
the dominant cost of step 1.

For deeper context on what each phase is doing and why, see
`REPORT.md` at the repo root or `paper/main.tex` for the paper version.
