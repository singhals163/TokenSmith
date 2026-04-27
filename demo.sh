#!/usr/bin/env bash
# TokenSmith live demo: extract a fresh PDF, build the MiniLM index, ask one
# question. Each step is timed with bash so the wall clock you see is the
# end-to-end number (Python startup included), not just the work after imports.
#
# Expected total cold wall on a 4-core / 8 GiB consumer GPU profile:
#   extract  ~  9 s
#   index    ~ 30 s
#   answer   ~  3 s
#   ----------
#   total    ~ 42 s
#
# On a slow shared filesystem (e.g. PACE Lustre on a cold node) the Python
# import phase alone can add 20-30 s to step 1, so the wall on this script
# may be larger than the headline numbers in the report. The numbers in the
# report are the warm-cache numbers; this script reports the honest cold wall.
#
# Usage:
#   ./demo.sh                                     # uses data/chapters/textbook.pdf
#   ./demo.sh data/chapters/your_textbook.pdf

set -e

PDF="${1:-data/chapters/textbook.pdf}"
INDEX_PREFIX="demo"
DEMO_CFG="config/config.demo.yaml"
MD_OUT="data/textbook--extracted_markdown.md"

# ------------------------------------------------------------------
# Runtime fixups for llama-cpp-python.
#
# The wheel is built against CUDA 12 runtime libs that ship as pip wheels
# inside the conda env (nvidia-cuda-runtime-cu12 and friends). The libs are
# not on the default LD_LIBRARY_PATH, so we prepend them here. Auto-detected
# from $CONDA_PREFIX so the script does not hardcode any user-specific path.
#
# Also pin the conda env's libstdc++ ahead of any system one. On older HPC
# images the system libstdc++ is missing CXXABI symbols that sqlite3 and
# transformers need after llama-cpp-python is imported.
# ------------------------------------------------------------------
if [ -n "${CONDA_PREFIX:-}" ]; then
  for sub in cuda_runtime cuda_nvrtc cublas cudnn cufft cusparse cusolver; do
    for d in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/"$sub"/lib; do
      [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
    done
  done
  if [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
else
  echo "WARN: CONDA_PREFIX is not set. Activate the tokensmith env first:"
  echo "      conda activate tokensmith"
  echo
fi

if [ ! -f "$PDF" ]; then
  echo "ERROR: PDF not found at $PDF"
  echo "Drop your PDF into data/chapters/, or pass a path to demo.sh."
  exit 1
fi

# ------------------------------------------------------------------
# Run a labeled step and capture its wall time in seconds (LAST_T).
# Uses date with nanosecond precision and awk for the subtraction so we do
# not depend on bc or any other package.
# ------------------------------------------------------------------
LAST_T=""
run_step() {
  local label="$1"; shift
  echo
  echo "================================================================"
  echo "$label"
  echo "================================================================"
  local t0=$(date +%s.%N)
  "$@"
  local t1=$(date +%s.%N)
  LAST_T=$(awk -v a="$t1" -v b="$t0" 'BEGIN { printf "%.2f", a - b }')
  echo
  printf ">>> wall: %s s   (end-to-end, includes Python startup)\n" "$LAST_T"
}

run_step "Step 1 / 3: Extract PDF (pypdfium2, 4 workers)" \
  python -m src.preprocessing.pypdfium_extractor \
    --pdf "$PDF" --out "$MD_OUT" --workers 4
T_EXTRACT="$LAST_T"

run_step "Step 2 / 3: Build MiniLM index over the extracted markdown" \
  python -m src.main index --config "$DEMO_CFG" --index_prefix "$INDEX_PREFIX"
T_INDEX="$LAST_T"

echo
echo "================================================================"
TOTAL=$(awk -v a="$T_EXTRACT" -v b="$T_INDEX" 'BEGIN { printf "%.2f", a + b }')
printf "Cold ingestion total: %s s   (extract %s s + index %s s)\n" \
  "$TOTAL" "$T_EXTRACT" "$T_INDEX"
echo "================================================================"
echo
echo "Step 3 / 3: Chat. Ask your first question. Ctrl+C to exit."
python -m src.main chat --config "$DEMO_CFG" --index_prefix "$INDEX_PREFIX"
