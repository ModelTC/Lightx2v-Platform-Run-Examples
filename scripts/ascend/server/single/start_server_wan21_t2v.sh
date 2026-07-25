#!/bin/bash
set -eo pipefail

# Wan2.1 T2V server for Ascend NPU
# System management interface: npu-smi info

lightx2v_path=${LIGHTX2V_PATH:-/data/wushuo1/LightX2V}
model_path=${MODEL_PATH:-/data/wushuo1/models/Wan2.1-T2V-1.3B}
host=${HOST:-0.0.0.0}
port=${PORT:-8000}
max_queue_size=${MAX_QUEUE_SIZE:-10}

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.server \
    --model_cls wan2.1 \
    --task t2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/ascend_npu/single/wan_t2v.json" \
    --host "${host}" \
    --port "${port}" \
    --max_queue_size "${max_queue_size}"

echo "Service stopped"
