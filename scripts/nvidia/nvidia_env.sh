#!/usr/bin/env bash

# Shared NVIDIA CUDA environment for the offline inference entrypoints.
# The calling entrypoint must set repo_path, run_group and CUDA_VISIBLE_DEVICES.

if [[ -z "${repo_path:-}" || -z "${run_group:-}" ]]; then
    echo "[ERROR] nvidia_env.sh requires repo_path and run_group" >&2
    return 64
fi

export PLATFORM=cuda
export DEVICE_TYPE=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TOKENIZERS_PARALLELISM=false
export INFER_PROFILE_LEVEL="${INFER_PROFILE_LEVEL:-2}"
export PROFILING_DEBUG_LEVEL="${INFER_PROFILE_LEVEL}"

# All checkpoints referenced by these entrypoints are local.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

log_root="${repo_path}/logs/nvidia/infer/${run_group}"
result_root="${repo_path}/results/nvidia/infer/${run_group}"
