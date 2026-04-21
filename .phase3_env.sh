# Phase 3 cache redirects — source this before running experiments
export HF_HOME=/storage/scratch1/1/ssinghal88/cache/hf
export HUGGINGFACE_HUB_CACHE=/storage/scratch1/1/ssinghal88/cache/hf
export TRANSFORMERS_CACHE=/storage/scratch1/1/ssinghal88/cache/hf
export XDG_CACHE_HOME=/storage/scratch1/1/ssinghal88/cache
export PIP_CACHE_DIR=/storage/scratch1/1/ssinghal88/cache/pip
export GPT4ALL_MODEL_PATH=/storage/scratch1/1/ssinghal88/cache/gpt4all

# Put conda env libs first — the module-loaded libstdc++ is older than conda's and breaks sqlite3
_CONDA_LIB=/storage/home/hcoda1/1/ssinghal88/scratch/.conda/envs/tokensmith/lib
# CUDA 12 runtime shipped as pip wheels (needed by llama-cpp-python CUDA build)
_CUDA_LIB=/storage/home/hcoda1/1/ssinghal88/scratch/.conda/envs/tokensmith/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
_NVRTC_LIB=/storage/home/hcoda1/1/ssinghal88/scratch/.conda/envs/tokensmith/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib
_CUBLAS_LIB=/storage/home/hcoda1/1/ssinghal88/scratch/.conda/envs/tokensmith/lib/python3.12/site-packages/nvidia/cublas/lib
export LD_LIBRARY_PATH="${_CONDA_LIB}:${_CUDA_LIB}:${_NVRTC_LIB}:${_CUBLAS_LIB}:${LD_LIBRARY_PATH}"

# llama-cpp-python's libllama.so was built with an RPATH pinned to an older
# spack libstdc++; preload the conda one so sqlite3/icu still find CXXABI_1.3.15.
export LD_PRELOAD="${_CONDA_LIB}/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

# CUDA 12.6 toolkit (nvcc) — only needed if you rebuild llama-cpp-python from
# source for sm_70. At runtime the conda-packaged libcudart.so.12 is enough.
# Uncomment the next two lines before doing a CMAKE_CUDA_ARCHITECTURES=70 build.
# module load cuda/12.6.1 2>/dev/null || true
# export CUDACXX=$(which nvcc 2>/dev/null)
