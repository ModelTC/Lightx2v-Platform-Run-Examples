#!/bin/bash

server_required_vars=(
    repo_path
    lightx2v_path
    model_path
    config_path
    case_id
    model_cls
    task
    world_size
    parallel_strategy
    visible_devices
)

for server_var in "${server_required_vars[@]}"; do
    if [ -z "${!server_var:-}" ]; then
        echo "Error: ${server_var} is not set." >&2
        return 1
    fi
done

if [ ! -d "${lightx2v_path}" ]; then
    echo "Error: LightX2V path does not exist: ${lightx2v_path}" >&2
    return 1
fi
if [ ! -e "${model_path}" ]; then
    echo "Error: model path does not exist: ${model_path}" >&2
    return 1
fi
if [ ! -f "${config_path}" ]; then
    echo "Error: config does not exist: ${config_path}" >&2
    return 1
fi

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${visible_devices}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
# torch_npu reads its own allocator variable. The generic PyTorch setting
# exported by LightX2V's base.sh is not consumed by the NPU caching allocator.
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

host="${HOST:-0.0.0.0}"
port="${PORT:-8000}"
metric_port="${METRIC_PORT:-8001}"
max_queue_size="${MAX_QUEUE_SIZE:-10}"

IFS=',' read -r -a server_devices <<< "${ASCEND_RT_VISIBLE_DEVICES}"
if [ "${#server_devices[@]}" -ne "${world_size}" ]; then
    echo "Error: ${case_id} requires ${world_size} NPUs, but ASCEND_RT_VISIBLE_DEVICES selects ${#server_devices[@]}: ${ASCEND_RT_VISIBLE_DEVICES}" >&2
    return 1
fi
if [ "${port}" = "${metric_port}" ]; then
    echo "Error: PORT and METRIC_PORT must be different." >&2
    return 1
fi

source "${lightx2v_path}/scripts/base/base.sh"

run_lightx2v_server() {
    echo "==============================================================================="
    echo "Starting LightX2V distributed server"
    echo "case_id: ${case_id}"
    echo "model_cls: ${model_cls}"
    echo "task: ${task}"
    echo "parallel_strategy: ${parallel_strategy}"
    echo "world_size: ${world_size}"
    echo "devices: ${ASCEND_RT_VISIBLE_DEVICES}"
    echo "config: ${config_path}"
    echo "API: http://${host}:${port}"
    echo "metrics: http://${host}:${metric_port}"
    echo "==============================================================================="

    exec torchrun \
        --nnodes=1 \
        --nproc_per_node="${world_size}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        -m lightx2v.server \
        --model_cls "${model_cls}" \
        --task "${task}" \
        --model_path "${model_path}" \
        --config_json "${config_path}" \
        --host "${host}" \
        --port "${port}" \
        --metric_port "${metric_port}" \
        --max_queue_size "${max_queue_size}"
}
