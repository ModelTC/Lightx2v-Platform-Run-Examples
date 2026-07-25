#!/bin/bash
set -eo pipefail

# Wan2.2 MoE T2V 8-NPU server for Ascend NPU
# System management interface: npu-smi

lightx2v_path=${LIGHTX2V_PATH:-/data/wushuo1/LightX2V}
model_path=${MODEL_PATH:-/data/wushuo1/models/Wan2.2-T2V-A14B}
host=${HOST:-0.0.0.0}
port=${PORT:-8000}
max_queue_size=${MAX_QUEUE_SIZE:-10}
npus=${NPUS:-8}
master_port=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

source "${lightx2v_path}/scripts/platforms/ascend_npu/dist/logging.sh"
source "${lightx2v_path}/scripts/base/base.sh"
source "$(dirname "$0")/cleanup_lightx2v_server.sh"

cleanup_lightx2v_server_by_port "${port}"

torchrun --master_port="${master_port}" --nproc_per_node="${npus}" -m lightx2v.server \
    --model_cls wan2.2_moe \
    --task t2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/ascend_npu/dist/wan_moe_t2v.json" \
    --host "${host}" \
    --port "${port}" \
    --max_queue_size "${max_queue_size}"

echo "Service stopped"
