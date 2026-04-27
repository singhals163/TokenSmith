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

The script runs three steps and times each one. What you should see:

```
================================================================
Step 1 / 3: Extract PDF (pypdfium2, 4 workers)
================================================================
Pages              : 2195
Ranges             : 5 (workers=4)
Map-phase wall     : 3.26s
Total wall         : 8.66s
...
real    0m9.012s

================================================================
Step 2 / 3: Build MiniLM index over the extracted markdown
================================================================
Embedding 2,664 chunks with all-MiniLM-L6-v2 ...
Hardware Router: GPU detected; using sequential high-throughput path.
...
real    0m30.412s

================================================================
Step 3 / 3: Chat. Ask your first question. Ctrl+C to exit.
================================================================
You: Explain the ACID properties of database transactions.
TokenSmith: ...                                            (~3 s)
```

Total cold time-to-first-answer should land near 45 seconds.

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

## What the timing means

The script uses bash's `time` builtin, which reports `real` (wall),
`user` (CPU summed across cores), and `sys`. The number to call out
during a demo is `real`. If `real` is meaningfully larger than the
target on your machine, the most likely cause is the Qwen2.5-1.5B GGUF
not being page-cached yet (cold disk read). Re-running the script
(`./demo.sh ...`) lands closer to the warm numbers in the report.

For deeper context on what each phase is doing and why, see
`REPORT.md` at the repo root or `paper/main.tex` for the paper version.
