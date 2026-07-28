#!/usr/bin/env bash

# MetaX C500-only LightX2V server runtime.
# The caller declares one case and this file owns environment setup and
# torchrun. Server performance runs intentionally disable profiling.

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
    if [[ -z "${!server_var:-}" ]]; then
        echo "[ERROR] ${server_var} is not set" >&2
        return 64
    fi
done

if [[ ! -d "${lightx2v_path}" ]]; then
    echo "[ERROR] LightX2V path does not exist: ${lightx2v_path}" >&2
    return 66
fi
if [[ ! -e "${model_path}" ]]; then
    echo "[ERROR] model path does not exist: ${model_path}" >&2
    return 66
fi
if [[ ! -f "${config_path}" ]]; then
    echo "[ERROR] config does not exist: ${config_path}" >&2
    return 66
fi

export PLATFORM=metax_cuda
export DEVICE_TYPE=cuda
export DEVICE_MEMORY_GIB=64
unset ASCEND_RT_VISIBLE_DEVICES
unset MLU_VISIBLE_DEVICES
unset CN_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${visible_devices}}"
export VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

export MACA_PATH="${MACA_PATH:-/opt/maca-3.7.1}"
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# LightX2V's base.sh appends to PYTHONPATH without an unset-variable guard,
# while these entrypoints intentionally use `set -u`.
export PYTHONPATH="${PYTHONPATH:-}"

# The formal service benchmark measures client-observed E2E P50/P90. Level 2
# inserts detailed profiling work and is therefore forbidden for this suite.
export INFER_PROFILE_LEVEL=0
export PROFILING_DEBUG_LEVEL=0

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
host="${HOST:-0.0.0.0}"
port="${PORT:-8000}"
metric_port="${METRIC_PORT:-8001}"
max_queue_size="${MAX_QUEUE_SIZE:-10}"

IFS=',' read -r -a server_devices <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#server_devices[@]} != world_size )); then
    echo "[ERROR] ${case_id} requires ${world_size} GPUs, but CUDA_VISIBLE_DEVICES selects ${#server_devices[@]}: ${CUDA_VISIBLE_DEVICES}" >&2
    return 64
fi
if [[ "${port}" == "${metric_port}" || "${port}" == "${MASTER_PORT}" \
    || "${metric_port}" == "${MASTER_PORT}" ]]; then
    echo "[ERROR] PORT, METRIC_PORT and MASTER_PORT must be distinct" >&2
    return 64
fi

source "${lightx2v_path}/scripts/base/base.sh"
# base.sh has its own default; restore the service benchmark contract.
export INFER_PROFILE_LEVEL=0
export PROFILING_DEBUG_LEVEL=0

run_lightx2v_server() {
    echo "==============================================================================="
    echo "Starting LightX2V MetaX distributed server"
    echo "case_id: ${case_id}"
    echo "model_cls: ${model_cls}"
    echo "task: ${task}"
    echo "platform: ${PLATFORM}"
    echo "parallel_strategy: ${parallel_strategy}"
    echo "world_size: ${world_size}"
    echo "devices: ${CUDA_VISIBLE_DEVICES}"
    echo "profiling_debug_level: ${PROFILING_DEBUG_LEVEL}"
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
