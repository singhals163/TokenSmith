#!/usr/bin/env bash
# TokenSmith live demo: extract a fresh PDF, build the MiniLM index, ask one
# question. Each step is timed with bash's built-in `time` so you can read out
# the numbers as they finish.
#
# Expected total cold wall on a 4-core / 8 GiB consumer GPU profile:
#   extract  ~  9 s   (pypdfium2, 4 workers)
#   index    ~ 30 s   (MiniLM 23M on the GPU, 2.6k chunks)
#   answer   ~  3 s   (cross-encoder rerank + Qwen2.5-1.5B generation)
#   ----------
#   total    ~ 42 s
#
# Usage:
#   ./demo.sh                                     # uses data/chapters/textbook.pdf
#   ./demo.sh data/chapters/your_textbook.pdf

set -e

PDF="${1:-data/chapters/textbook.pdf}"
INDEX_PREFIX="demo"
DEMO_CFG="config/config.demo.yaml"
MD_OUT="data/textbook--extracted_markdown.md"

if [ ! -f "$PDF" ]; then
  echo "ERROR: PDF not found at $PDF"
  echo "Drop your PDF into data/chapters/, or pass a path to demo.sh."
  exit 1
fi

echo "================================================================"
echo "Step 1 / 3: Extract PDF (pypdfium2, 4 workers)"
echo "================================================================"
time python -m src.preprocessing.pypdfium_extractor \
  --pdf "$PDF" \
  --out "$MD_OUT" \
  --workers 4

echo
echo "================================================================"
echo "Step 2 / 3: Build MiniLM index over the extracted markdown"
echo "================================================================"
time python -m src.main index \
  --config "$DEMO_CFG" \
  --index_prefix "$INDEX_PREFIX"

echo
echo "================================================================"
echo "Step 3 / 3: Chat. Ask your question. Ctrl+C to exit."
echo "================================================================"
python -m src.main chat \
  --config "$DEMO_CFG" \
  --index_prefix "$INDEX_PREFIX"
