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

server_platform="${server_platform:-ascend_npu}"
case "${server_platform}" in
    ascend_npu)
        export PLATFORM=ascend_npu
        export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${visible_devices}}"
        export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
        # torch_npu reads its own allocator variable. The generic PyTorch
        # setting from base.sh is not consumed by the NPU caching allocator.
        export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
        server_visible_devices="${ASCEND_RT_VISIBLE_DEVICES}"
        server_device_name=NPUs
        ;;
    cambricon_mlu)
        export PLATFORM=cambricon_mlu
        export DEVICE_TYPE=mlu
        export DEVICE_MEMORY_GIB=80
        export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-${visible_devices}}"
        # torch_mlu gives CN_VISIBLE_DEVICES precedence on some releases.
        # Keep MLU_VISIBLE_DEVICES as the single source of device visibility.
        unset CN_VISIBLE_DEVICES
        export PYTORCH_MLU_ALLOC_CONF="${PYTORCH_MLU_ALLOC_CONF:-expandable_segments:True}"
        export LD_LIBRARY_PATH="/usr/local/neuware/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
        export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
        export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
        export CNCL_IB_DISABLE="${CNCL_IB_DISABLE:-1}"
        export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
        export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
        # Server speed tests default to Level 0 to avoid synchronization and
        # logging overhead. INFER_PROFILE_LEVEL is the public override;
        # PROFILING_DEBUG_LEVEL remains compatible with direct launches.
        export INFER_PROFILE_LEVEL="${INFER_PROFILE_LEVEL:-${PROFILING_DEBUG_LEVEL:-0}}"
        case "${INFER_PROFILE_LEVEL}" in
            0|1|2) ;;
            *)
                echo "Error: INFER_PROFILE_LEVEL must be 0, 1, or 2; got: ${INFER_PROFILE_LEVEL}" >&2
                return 1
                ;;
        esac
        export PROFILING_DEBUG_LEVEL="${INFER_PROFILE_LEVEL}"
        server_visible_devices="${MLU_VISIBLE_DEVICES}"
        server_device_name=MLUs
        ;;
    *)
        echo "Error: unsupported server platform: ${server_platform}" >&2
        return 1
        ;;
esac

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

host="${HOST:-0.0.0.0}"
port="${PORT:-8000}"
metric_port="${METRIC_PORT:-8001}"
max_queue_size="${MAX_QUEUE_SIZE:-10}"

IFS=',' read -r -a server_devices <<< "${server_visible_devices}"
if [ "${#server_devices[@]}" -ne "${world_size}" ]; then
    echo "Error: ${case_id} requires ${world_size} ${server_device_name}, but the visibility setting selects ${#server_devices[@]}: ${server_visible_devices}" >&2
    return 1
fi
if [ "${port}" = "${metric_port}" ]; then
    echo "Error: PORT and METRIC_PORT must be different." >&2
    return 1
fi

source "${lightx2v_path}/scripts/base/base.sh"
if [ "${server_platform}" = "cambricon_mlu" ]; then
    # LightX2V's base.sh currently assigns its own profiling default.
    # Restore the caller-selected server profile level after sourcing it.
    export PROFILING_DEBUG_LEVEL="${INFER_PROFILE_LEVEL}"
fi

run_lightx2v_server() {
    echo "==============================================================================="
    echo "Starting LightX2V distributed server"
    echo "case_id: ${case_id}"
    echo "model_cls: ${model_cls}"
    echo "task: ${task}"
    echo "platform: ${PLATFORM}"
    echo "parallel_strategy: ${parallel_strategy}"
    echo "world_size: ${world_size}"
    echo "devices: ${server_visible_devices}"
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
