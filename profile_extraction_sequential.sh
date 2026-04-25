#!/bin/bash

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tokensmith

# Ensure we are in the project root
cd "$(dirname "$0")"

# Initialize the overall timing log file
TIMING_FILE="profile_experiment_time.txt"
echo "Experiment Timing Results" > "$TIMING_FILE"
echo "=============================================" >> "$TIMING_FILE"

# Define experiments: format is "WORKERS CHUNK_SIZE"
# Assuming a ~2000 page textbook
declare -a EXPERIMENTS=(
    # "32 63"
    # "16 125"
    # "8 250"
    # "4 500"
    # "2 1000"
    "1 2000"
)

echo "Starting Ablation Study for PDF Extraction..."
echo "============================================="

for exp in "${EXPERIMENTS[@]}"; do
    # Read variables from the array string
    read -r WORKERS CHUNK_SIZE <<< "$exp"
    
    # Define outputs
    OUT_DIR="data/experiments_sequential/workers_${WORKERS}_chunk_${CHUNK_SIZE}"
    PROFILE_FILE="profiling_${WORKERS}_workers.txt"
    
    # Calculate CPU cores to pin starting at core 10
    # If 1 worker, it pins to CPU 10. If 4 workers, it pins to CPUs 10,11,12,13.
    CORE_START=10
    CORE_END=$((CORE_START + WORKERS - 1))
    CORE_RANGE="${CORE_START}-${CORE_END}"
    
    echo -e "\n\n>>> RUNNING EXPERIMENT: ${WORKERS} Workers | Chunk Size: ${CHUNK_SIZE} <<<"
    echo "Pinning to CPU Cores: ${CORE_RANGE}"
    
    # Reset bash's internal SECONDS counter to 0 for this experiment
    SECONDS=0
    
    # Run the Python script using taskset to lock it to the specific CPUs
    taskset -c "$CORE_RANGE" python -m src.preprocessing.extraction \
        --workers "$WORKERS" \
        --chunk_size "$CHUNK_SIZE" \
        --out_dir "$OUT_DIR" \
        --profile_out "$PROFILE_FILE"

    # Capture the elapsed time
    ELAPSED_TIME=$SECONDS
    
    # Log the time to the tracking file
    echo "Workers: $(printf '%-2s' "$WORKERS") | Chunk Size: $(printf '%-4s' "$CHUNK_SIZE") | Time: ${ELAPSED_TIME} seconds" >> "$TIMING_FILE"

    echo ">>> EXPERIMENT COMPLETE. Took ${ELAPSED_TIME} seconds. Output saved to ${OUT_DIR} <<<"
done

echo -e "\nAll experiments finished! Check ${TIMING_FILE} for the overall duration summary."