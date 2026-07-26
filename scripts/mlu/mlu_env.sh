#!/usr/bin/env bash

# Shared Cambricon MLU590 environment for the offline inference entrypoints.
# The calling entrypoint must set repo_path, run_group and MLU_VISIBLE_DEVICES.

if [[ -z "${repo_path:-}" || -z "${run_group:-}" ]]; then
    echo "[ERROR] mlu_env.sh requires repo_path and run_group" >&2
    return 64
fi

export PLATFORM=cambricon_mlu
export DEVICE_TYPE=mlu
export DEVICE_MEMORY_GIB=80
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0}"

# torch_mlu gives CN_VISIBLE_DEVICES precedence on some releases. Keep one
# unambiguous source of device visibility for all entrypoints.
unset CN_VISIBLE_DEVICES

export PYTORCH_MLU_ALLOC_CONF="${PYTORCH_MLU_ALLOC_CONF:-expandable_segments:True}"
export LD_LIBRARY_PATH="/usr/local/neuware/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TOKENIZERS_PARALLELISM=false
export PROFILING_DEBUG_LEVEL=0
# This matrix is single-node and uses the direct MLU-Link fabric.
export CNCL_IB_DISABLE="${CNCL_IB_DISABLE:-1}"

# Every referenced checkpoint is local. These prevent a detached suite from
# waiting on a network connection after the interactive session disappears.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

log_root="${repo_path}/logs/mlu/infer/${run_group}"
result_root="${repo_path}/results/mlu/infer/${run_group}"
