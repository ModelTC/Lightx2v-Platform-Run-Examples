#!/usr/bin/env bash

# Shared MetaX C500 environment for the offline inference entrypoints.
# The calling entrypoint must set repo_path, run_group and CUDA_VISIBLE_DEVICES.

if [[ -z "${repo_path:-}" || -z "${run_group:-}" ]]; then
    echo "[ERROR] metax_env.sh requires repo_path and run_group" >&2
    return 64
fi

export PLATFORM=metax_cuda
export DEVICE_TYPE=cuda
export DEVICE_MEMORY_GIB=64
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Keep the CUDA-compatible MetaX toolchain ahead of any host CUDA installation.
export MACA_PATH=/opt/maca-3.7.1
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TOKENIZERS_PARALLELISM=false
# Level 2 is part of the benchmark record contract: it keeps both the
# synchronized per-step Level1 timings and the pipeline-level Level2 timings.
# Use a separate knob because LightX2V's base.sh also writes
# PROFILING_DEBUG_LEVEL while it is sourced.
export INFER_PROFILE_LEVEL="${INFER_PROFILE_LEVEL:-2}"
export PROFILING_DEBUG_LEVEL="${INFER_PROFILE_LEVEL}"

# All checkpoints in this matrix are local, so detached runs must never wait
# for a Hugging Face network request.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Keep one canonical visibility value for the common runtime and run record.
export VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

log_root="${repo_path}/logs/metax/infer/${run_group}"
result_root="${repo_path}/results/metax/infer/${run_group}"
