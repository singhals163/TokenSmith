#!/bin/bash

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tokensmith

# Ensure we are in the project root
cd "$(dirname "$0")"

echo "============================================="
echo "Starting CONCURRENT Ablation Study..."
echo "Running experiments for 1, 2, 4, 8, and 16 workers."
echo "============================================="

# Helper function to launch experiments in the background
launch_experiment() {
    local WORKERS=$1
    local CHUNK_SIZE=$2
    local CORE_START=$3
    local CORE_END=$4
    
    local OUT_DIR="data/experiments/workers_${WORKERS}_chunk_${CHUNK_SIZE}"
    local PROFILE_FILE="profiling_${WORKERS}_workers.txt"
    local CORE_RANGE="${CORE_START}-${CORE_END}"
    
    echo ">>> Launching ${WORKERS} Workers on Cores [${CORE_RANGE}] | Chunk: ${CHUNK_SIZE} <<<"
    
    # Use taskset to lock to specific CPUs, and '&' to push to background
    taskset -c "$CORE_RANGE" python -m src.preprocessing.extraction \
        --workers "$WORKERS" \
        --chunk_size "$CHUNK_SIZE" \
        --out_dir "$OUT_DIR" \
        --profile_out "$PROFILE_FILE" &
}

# Launch experiments simultaneously on STRICTLY isolated cores
# Format: launch_experiment <workers> <chunk_size> <start_core> <end_core>

launch_experiment 1 2000 0 0        # Uses 1 core  (Core 0)
launch_experiment 2 1000 1 2        # Uses 2 cores (Cores 1-2)
launch_experiment 4 500 3 6         # Uses 4 cores (Cores 3-6)
launch_experiment 8 250 7 14        # Uses 8 cores (Cores 7-14)
launch_experiment 16 125 15 30      # Uses 16 cores (Cores 15-30)
launch_experiment 32 63 31 62      # Uses 32 cores (Cores 31-62)

echo -e "\nAll isolated workers have been dispatched!"
echo "Waiting for all background processes to finish..."

# The 'wait' command pauses the script until all background '&' jobs finish
wait 

echo -e "\nAll concurrent experiments complete! Check the data/experiments/ folder."