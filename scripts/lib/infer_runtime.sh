#!/usr/bin/env bash

# Shared lifecycle for readable accelerator inference entrypoints.
#
# The entrypoint owns the literal LightX2V command:
#
#   lightx2v_infer() {
#       exec python -m lightx2v.infer ... --save_result_path "${result_path}"
#   }
#   run_infer lightx2v_infer
#
# This runtime owns one run directory, logging, preflight, run.json, artifact
# validation, child supervision and signal forwarding. It deliberately never
# evaluates command text.

_INFER_RUNTIME_ENTRY_SCRIPT="${BASH_SOURCE[1]:-$0}"
_INFER_RUNTIME_INITIALIZED=0
_INFER_RUNTIME_ACTIVE=0
_INFER_RUNTIME_FINALIZED=0
_INFER_RUNTIME_RECORD_STARTED=0
_INFER_RUNTIME_CHILD_PID=""
_INFER_RUNTIME_SIGNAL_NAME=""
_INFER_RUNTIME_SIGNAL_CODE=""
_INFER_RUNTIME_RECORD_TOOL=""

_infer_runtime_flush_log() {
    local marker
    local attempt

    if [[ -z "${RUN_LOG_PATH:-}" || ! -f "${RUN_LOG_PATH}" ]]; then
        return 0
    fi
    marker="[Runtime] log barrier ${RUN_ID:-unknown} $$ ${RANDOM}${RANDOM}"
    printf '%s\n' "${marker}"
    for ((attempt = 0; attempt < 500; attempt++)); do
        if grep -Fqx -- "${marker}" "${RUN_LOG_PATH}" 2>/dev/null; then
            return 0
        fi
        sleep 0.01
    done
    echo "[RunRecord] warning: timed out while flushing run.log" >&2
    return 0
}

_infer_runtime_finish_record() {
    local record_rc=0

    if [[ "${_INFER_RUNTIME_FINALIZED}" == "1" ]]; then
        return 0
    fi
    _INFER_RUNTIME_FINALIZED=1

    if [[ "${_INFER_RUNTIME_RECORD_STARTED}" == "1" ]]; then
        _infer_runtime_flush_log
        python "${_INFER_RUNTIME_RECORD_TOOL}" finish || record_rc=$?
    fi
    return "${record_rc}"
}

_infer_runtime_update_record_command() {
    python -c '
import json
import os
import sys
from pathlib import Path

record_path = Path(sys.argv[1])
command = sys.argv[2]
with record_path.open("r", encoding="utf-8") as record_file:
    record = json.load(record_file)
record.setdefault("execution", {})["command"] = command
temporary = record_path.with_name(f".{record_path.name}.{os.getpid()}.tmp")
try:
    with temporary.open("w", encoding="utf-8") as temporary_file:
        json.dump(record, temporary_file, ensure_ascii=False, indent=2)
        temporary_file.write("\n")
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    os.replace(temporary, record_path)
finally:
    temporary.unlink(missing_ok=True)
' "${RUN_RECORD_PATH}" "${INFER_COMMAND}"
}

_infer_runtime_disable_traps() {
    trap - EXIT INT TERM
}

_infer_runtime_on_exit() {
    local shell_rc=$?
    local record_rc=0
    local final_rc="${shell_rc}"

    if [[ "${_INFER_RUNTIME_FINALIZED}" == "1" ]]; then
        return
    fi

    if [[ -z "${WRAPPER_EXIT_CODE:-}" ]]; then
        export WRAPPER_EXIT_CODE="${shell_rc}"
    fi
    if [[ -z "${RUN_ERROR:-}" ]]; then
        export RUN_ERROR="inference runtime exited before normal finalization"
    fi
    _infer_runtime_finish_record || record_rc=$?

    if (( final_rc == 0 && record_rc != 0 )); then
        final_rc="${record_rc}"
    fi
    _infer_runtime_disable_traps
    exit "${final_rc}"
}

_infer_runtime_on_signal() {
    local signal_name=$1
    local signal_code=$2

    if [[ -z "${_INFER_RUNTIME_SIGNAL_CODE}" ]]; then
        _INFER_RUNTIME_SIGNAL_NAME="${signal_name}"
        _INFER_RUNTIME_SIGNAL_CODE="${signal_code}"
    fi

    if [[ -n "${_INFER_RUNTIME_CHILD_PID}" ]] \
        && kill -0 "${_INFER_RUNTIME_CHILD_PID}" 2>/dev/null; then
        kill -s "${signal_name}" "${_INFER_RUNTIME_CHILD_PID}" 2>/dev/null || true
        return
    fi

    export WRAPPER_EXIT_CODE="${signal_code}"
    export RUN_ERROR="received ${signal_name}"
    exit "${signal_code}"
}

_infer_runtime_fail() {
    local failure_rc=$1
    shift
    local record_rc=0

    export WRAPPER_EXIT_CODE="${failure_rc}"
    export RUN_ERROR="$*"
    echo "[ERROR] ${RUN_ERROR}" >&2
    _infer_runtime_finish_record || record_rc=$?
    _infer_runtime_disable_traps

    if (( failure_rc == 0 && record_rc != 0 )); then
        return "${record_rc}"
    fi
    return "${failure_rc}"
}

_infer_runtime_absolute_path() {
    local path_value=$1
    local path_dir
    local path_base

    if [[ "${path_value}" == /* ]]; then
        printf '%s\n' "${path_value}"
        return 0
    fi
    path_dir="$(dirname -- "${path_value}")"
    path_base="$(basename -- "${path_value}")"
    printf '%s/%s\n' "$(cd -- "${path_dir}" && pwd)" "${path_base}"
}

_infer_runtime_require_inputs() {
    local variable_name
    local -a nonempty_names=(
        repo_path lightx2v_path model_path config_path model_cls task prompt
        seed result_ext output_width output_height output_frames infer_steps
        offload_strategy reference_script
    )

    for variable_name in "${nonempty_names[@]}"; do
        if [[ -z "${!variable_name:-}" ]]; then
            echo "[ERROR] required entrypoint variable is empty: ${variable_name}" >&2
            return 64
        fi
    done
    if ! [[ -v negative_prompt ]]; then
        echo "[ERROR] required entrypoint variable is not set: negative_prompt" >&2
        return 64
    fi
}

_infer_runtime_validate_integer() {
    local name=$1
    local value=$2
    local allow_zero=${3:-0}

    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${name} must be an integer; got: ${value}" >&2
        return 64
    fi
    if [[ "${allow_zero}" != "1" ]] && (( value < 1 )); then
        echo "[ERROR] ${name} must be positive; got: ${value}" >&2
        return 64
    fi
}

_infer_runtime_export_contract() {
    local derived_case
    local device_count_value
    local device_family
    local device_type_value
    local platform_value
    local source_value
    local visibility_environment
    local visible_devices_value

    world_size="${world_size:-${WORLD_SIZE:-1}}"
    parallel_strategy="${parallel_strategy:-${PARALLEL_STRATEGY:-single}}"
    source_value="${source_script:-${_INFER_RUNTIME_ENTRY_SCRIPT}}"
    source_script="$(_infer_runtime_absolute_path "${source_value}")" || return 64
    derived_case="$(basename -- "${source_script}" .sh)"
    derived_case="${derived_case#run_}"
    case_id="${case_id:-${CASE_ID:-${derived_case}}}"
    model_id="${model_id:-${MODEL_ID:-${model_cls}}}"

    _infer_runtime_validate_integer world_size "${world_size}" || return $?
    _infer_runtime_validate_integer output_width "${output_width}" || return $?
    _infer_runtime_validate_integer output_height "${output_height}" || return $?
    _infer_runtime_validate_integer output_frames "${output_frames}" || return $?
    _infer_runtime_validate_integer infer_steps "${infer_steps}" || return $?
    if ! [[ "${seed}" =~ ^-?[0-9]+$ ]]; then
        echo "[ERROR] seed must be an integer; got: ${seed}" >&2
        return 64
    fi
    if ! [[ "${case_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "[ERROR] invalid case_id: ${case_id}" >&2
        return 64
    fi

    platform_value="${platform:-${PLATFORM:-ascend_npu}}"
    if [[ -n "${device_type:-}" ]]; then
        device_type_value="${device_type}"
    elif [[ "${platform_value,,}" == *mlu* ]]; then
        device_type_value="mlu"
    elif [[ "${platform_value,,}" == *metax* || "${platform_value,,}" == *cuda* ]]; then
        device_type_value="cuda"
    elif [[ "${platform_value,,}" == *ascend* || "${platform_value,,}" == *npu* ]]; then
        device_type_value="npu"
    else
        device_type_value="${DEVICE_TYPE:-npu}"
    fi
    if ! [[ "${platform_value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "[ERROR] invalid platform: ${platform_value}" >&2
        return 64
    fi
    if ! [[ "${device_type_value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "[ERROR] invalid device_type: ${device_type_value}" >&2
        return 64
    fi

    if [[ "${device_type_value,,}:${platform_value,,}" == *mlu* ]]; then
        device_family="mlu"
        visibility_environment="MLU_VISIBLE_DEVICES"
        if [[ -n "${visible_devices:-}" ]]; then
            visible_devices_value="${visible_devices}"
        elif [[ -n "${mlu_visible_devices:-}" ]]; then
            visible_devices_value="${mlu_visible_devices}"
        elif [[ -n "${MLU_VISIBLE_DEVICES:-}" ]]; then
            visible_devices_value="${MLU_VISIBLE_DEVICES:-}"
        else
            visible_devices_value="${VISIBLE_DEVICES:-}"
        fi
    elif [[ "${device_type_value,,}" == "cuda" \
        || "${device_type_value,,}" == "gpu" \
        || "${platform_value,,}" == *metax* \
        || "${platform_value,,}" == *cuda* ]]; then
        device_family="cuda"
        visibility_environment="CUDA_VISIBLE_DEVICES"
        if [[ -n "${visible_devices:-}" ]]; then
            visible_devices_value="${visible_devices}"
        elif [[ -n "${cuda_visible_devices:-}" ]]; then
            visible_devices_value="${cuda_visible_devices}"
        elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            visible_devices_value="${CUDA_VISIBLE_DEVICES:-}"
        else
            visible_devices_value="${VISIBLE_DEVICES:-}"
        fi
    else
        device_family="npu"
        visibility_environment="ASCEND_RT_VISIBLE_DEVICES"
        if [[ -n "${visible_devices:-}" ]]; then
            visible_devices_value="${visible_devices}"
        elif [[ -n "${ascend_rt_visible_devices:-}" ]]; then
            visible_devices_value="${ascend_rt_visible_devices}"
        elif [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
            visible_devices_value="${ASCEND_RT_VISIBLE_DEVICES:-}"
        else
            visible_devices_value="${VISIBLE_DEVICES:-}"
        fi
    fi
    if [[ -z "${visible_devices_value}" && "${world_size}" == "1" ]]; then
        visible_devices_value="0"
    fi

    device_count_value="${device_count:-${DEVICE_COUNT:-${world_size}}}"
    _infer_runtime_validate_integer device_count "${device_count_value}" || return $?

    export REPO_ROOT="${repo_path}"
    export LIGHTX2V_PATH="${lightx2v_path}"
    export MODEL_PATH="${model_path}"
    export CONFIG_PATH="${config_path}"
    export CASE_ID="${case_id}"
    export MODEL_ID="${model_id}"
    export TASK_TYPE="${task}"
    export PROMPT="${prompt}"
    export NEGATIVE_PROMPT="${negative_prompt}"
    export SEED="${seed}"
    export RESULT_EXT="${result_ext,,}"
    export OUTPUT_WIDTH="${output_width}"
    export OUTPUT_HEIGHT="${output_height}"
    export OUTPUT_FRAMES="${output_frames}"
    export INFER_STEPS="${infer_steps}"
    export OFFLOAD_STRATEGY="${offload_strategy}"
    export REFERENCE_SCRIPT="${reference_script}"
    export SOURCE_SCRIPT="${source_script}"
    export WORLD_SIZE="${world_size}"
    export PARALLEL_STRATEGY="${parallel_strategy}"
    export PLATFORM="${platform_value}"
    export DEVICE_TYPE="${device_type_value}"
    export DEVICE_FAMILY="${device_family}"
    export VISIBLE_DEVICES="${visible_devices_value}"
    export DEVICE_VISIBILITY_ENV="${visibility_environment}"
    export DEVICE_COUNT="${device_count_value}"
    if [[ "${device_family}" == "mlu" ]]; then
        export MLU_VISIBLE_DEVICES="${visible_devices_value}"
        export MLU_COUNT="${device_count_value}"
    elif [[ "${device_family}" == "cuda" ]]; then
        export CUDA_VISIBLE_DEVICES="${visible_devices_value}"
        export GPU_COUNT="${device_count_value}"
    else
        export ASCEND_RT_VISIBLE_DEVICES="${visible_devices_value}"
        export NPU_COUNT="${device_count_value}"
    fi
    export TASK="infer"
    if [[ -z "${DEVICE_MEMORY_GIB:-}" && "${device_family}" == "npu" ]]; then
        DEVICE_MEMORY_GIB=64
    fi
    export DEVICE_MEMORY_GIB="${DEVICE_MEMORY_GIB:-}"
    export DEVICE_ID="${visible_devices_value}"
    export RUN_MODE="single_sample_single_card"
    if (( world_size > 1 )); then
        export RUN_MODE="single_sample_multi_card"
    fi
    export DTYPE="${DTYPE:-BF16}"
    export SENSITIVE_LAYER_DTYPE="${SENSITIVE_LAYER_DTYPE:-None}"
    # Platform entrypoints set INFER_PROFILE_LEVEL before sourcing LightX2V's
    # base.sh. Prefer that stable value because base.sh and some historical
    # case scripts also assign PROFILING_DEBUG_LEVEL while being sourced.
    export PROFILING_DEBUG_LEVEL="${INFER_PROFILE_LEVEL:-${PROFILING_DEBUG_LEVEL:-2}}"

    if [[ -n "${audio_path:-}" ]]; then
        export AUDIO_PATH="${audio_path}"
    fi
}

_infer_runtime_create_run_paths() {
    local device_tag
    local run_log_dir
    local run_result_dir

    device_tag="${DEVICE_TYPE}${VISIBLE_DEVICES//,/-}"
    run_id="${run_id:-${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_${device_tag}_p$$}}"
    if ! [[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "[ERROR] invalid run_id: ${run_id}" >&2
        return 64
    fi
    export RUN_ID="${run_id}"

    export LOG_ROOT="${log_root:-${LOG_ROOT:-${repo_path}/logs/${PLATFORM}/infer}}"
    export RESULT_ROOT="${result_root:-${RESULT_ROOT:-${repo_path}/results/${PLATFORM}/infer}}"
    run_log_dir="${LOG_ROOT}/${case_id}/${run_id}"
    run_result_dir="${RESULT_ROOT}/${case_id}/${run_id}"
    if [[ -e "${run_log_dir}" || -e "${run_result_dir}" ]]; then
        echo "[ERROR] run_id already exists for ${case_id}: ${run_id}" >&2
        return 73
    fi

    mkdir -p "${LOG_ROOT}/${case_id}" "${RESULT_ROOT}/${case_id}" || return 73
    mkdir "${run_log_dir}" "${run_result_dir}" || return 73

    run_log_path="${run_log_dir}/run.log"
    run_record_path="${run_log_dir}/run.json"
    result_path="${run_result_dir}/output.${RESULT_EXT}"
    export RUN_LOG_PATH="${run_log_path}"
    export RUN_RECORD_PATH="${run_record_path}"
    export RESULT_PATH="${result_path}"
    export STARTED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    export START_EPOCH="$(date +%s)"
    export CHILD_EXIT_CODE=""
    export WRAPPER_EXIT_CODE=""
    export RUN_ERROR=""
}

_infer_runtime_run_preflight() {
    local -a preflight_args=(
        --repo-path "${repo_path}"
        --lightx2v-path "${lightx2v_path}"
        --model-path "${model_path}"
        --config-path "${config_path}"
        --source-script "${source_script}"
        --reference-script "${reference_script}"
        --world-size "${world_size}"
        --platform "${PLATFORM}"
        --device-type "${DEVICE_TYPE}"
        --device-count "${DEVICE_COUNT}"
        --visible-devices "${VISIBLE_DEVICES}"
        --visible-devices-env "${DEVICE_VISIBILITY_ENV}"
        --parallel-strategy "${parallel_strategy}"
        --result-ext "${RESULT_EXT}"
    )
    local preflight_rc=0

    if [[ -n "${audio_path:-}" ]]; then
        preflight_args+=(--audio-path "${audio_path}")
    fi
    python "${repo_path}/scripts/preflight_infer.py" "${preflight_args[@]}" || preflight_rc=$?
    if (( preflight_rc != 0 )); then
        return "${preflight_rc}"
    fi

    return "${preflight_rc}"
}

_infer_runtime_print_header() {
    echo "==============================================================================="
    echo "LightX2V ${PLATFORM} inference"
    echo "case_id: ${CASE_ID}"
    echo "run_id: ${RUN_ID}"
    echo "mode: ${RUN_MODE}"
    echo "parallel_strategy: ${PARALLEL_STRATEGY}"
    echo "world_size: ${WORLD_SIZE}"
    echo "device_type: ${DEVICE_TYPE}"
    echo "devices: ${VISIBLE_DEVICES}"
    echo "log: ${RUN_LOG_PATH}"
    echo "record: ${RUN_RECORD_PATH}"
    echo "result: ${RESULT_PATH}"
    echo "started_at_utc: ${STARTED_AT_UTC}"
    echo "==============================================================================="
}

_infer_runtime_wait_for_child() {
    local child_rc=0
    local had_errexit=0

    [[ $- == *e* ]] && had_errexit=1
    set +e
    while true; do
        wait "${_INFER_RUNTIME_CHILD_PID}"
        child_rc=$?
        if ! kill -0 "${_INFER_RUNTIME_CHILD_PID}" 2>/dev/null; then
            break
        fi
    done
    (( had_errexit == 1 )) && set -e
    return "${child_rc}"
}

_infer_runtime_initialize() {
    local command_name
    local preflight_rc=0

    if [[ "${_INFER_RUNTIME_INITIALIZED}" == "1" ]]; then
        echo "[ERROR] infer_runtime.sh may only be sourced once per entrypoint" >&2
        return 64
    fi
    _INFER_RUNTIME_INITIALIZED=1

    _infer_runtime_require_inputs || return $?
    _infer_runtime_export_contract || return $?

    for command_name in python tee; do
        if ! command -v "${command_name}" >/dev/null 2>&1; then
            echo "[ERROR] required command is unavailable: ${command_name}" >&2
            return 69
        fi
    done
    if (( world_size > 1 )) && ! command -v torchrun >/dev/null 2>&1; then
        echo "[ERROR] required command is unavailable: torchrun" >&2
        return 69
    fi

    _INFER_RUNTIME_RECORD_TOOL="${repo_path}/scripts/run_record.py"
    if [[ ! -f "${_INFER_RUNTIME_RECORD_TOOL}" ]]; then
        echo "[ERROR] run record tool does not exist: ${_INFER_RUNTIME_RECORD_TOOL}" >&2
        return 66
    fi
    if [[ ! -f "${repo_path}/scripts/preflight_infer.py" ]]; then
        echo "[ERROR] preflight tool does not exist: ${repo_path}/scripts/preflight_infer.py" >&2
        return 66
    fi

    _infer_runtime_create_run_paths || return $?
    touch "${RUN_LOG_PATH}"
    exec > >(tee -a "${RUN_LOG_PATH}") 2>&1
    trap _infer_runtime_on_exit EXIT
    trap '_infer_runtime_on_signal INT 130' INT
    trap '_infer_runtime_on_signal TERM 143' TERM

    export INFER_COMMAND="pending lightx2v_infer definition in ${SOURCE_SCRIPT}"
    _infer_runtime_print_header
    if ! python "${_INFER_RUNTIME_RECORD_TOOL}" start; then
        _infer_runtime_disable_traps
        return 70
    fi
    _INFER_RUNTIME_RECORD_STARTED=1

    _infer_runtime_run_preflight || preflight_rc=$?
    if (( preflight_rc != 0 )); then
        _infer_runtime_fail 65 "common inference preflight failed"
        return $?
    fi
}

run_infer() {
    local entry_function=${1:-}
    local function_definition
    local expected_launcher="python"
    local case_preflight_rc=0
    local child_rc=0
    local record_rc=0
    local final_rc=0

    if [[ "${_INFER_RUNTIME_ACTIVE}" == "1" ]]; then
        _infer_runtime_fail 64 "run_infer may only be called once per entrypoint"
        return $?
    fi
    _INFER_RUNTIME_ACTIVE=1

    if [[ -z "${entry_function}" ]] || ! declare -F "${entry_function}" >/dev/null 2>&1; then
        _infer_runtime_fail 64 "run_infer requires the name of a defined shell function"
        return $?
    fi
    function_definition="$(declare -f "${entry_function}")"

    if (( world_size > 1 )); then
        expected_launcher="torchrun"
    fi
    if ! grep -Eq \
        "(^|[[:space:]])exec[[:space:]]+${expected_launcher}([[:space:]]|$)" \
        <<<"${function_definition}"; then
        _infer_runtime_fail 64 "${entry_function} must invoke 'exec ${expected_launcher} ...'"
        return $?
    fi
    if ! grep -q -- "--save_result_path" <<<"${function_definition}" \
        || ! grep -q -- "result_path" <<<"${function_definition}"; then
        _infer_runtime_fail 64 \
            "${entry_function} must save to --save_result_path \"\${result_path}\""
        return $?
    fi

    export INFER_COMMAND="${function_definition}"
    if ! _infer_runtime_update_record_command; then
        _infer_runtime_fail 70 "failed to store the inference command in ${RUN_RECORD_PATH}"
        return $?
    fi

    if declare -F case_preflight >/dev/null 2>&1; then
        case_preflight || case_preflight_rc=$?
        if (( case_preflight_rc != 0 )); then
            _infer_runtime_fail 65 "case-specific inference preflight failed"
            return $?
        fi
    fi

    echo
    echo "[Preflight] model_id: ${MODEL_ID}"
    echo "[Preflight] model_cls: ${model_cls}"
    echo "[Preflight] task: ${TASK_TYPE}"
    echo "[Preflight] model_path: ${MODEL_PATH}"
    echo "[Preflight] config_path: ${CONFIG_PATH}"
    echo "[Preflight] offload_strategy: ${OFFLOAD_STRATEGY}"
    echo "[Preflight] target: ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT}, frames=${OUTPUT_FRAMES}, steps=${INFER_STEPS}"
    echo "[Preflight] seed: ${SEED}"
    echo "[Preflight] reference_script: ${REFERENCE_SCRIPT}"
    echo
    echo "[Command function]"
    echo "${INFER_COMMAND}"
    echo

    "${entry_function}" &
    _INFER_RUNTIME_CHILD_PID=$!
    _infer_runtime_wait_for_child || child_rc=$?
    _INFER_RUNTIME_CHILD_PID=""

    export CHILD_EXIT_CODE="${child_rc}"
    export WRAPPER_EXIT_CODE="0"
    if [[ -n "${_INFER_RUNTIME_SIGNAL_CODE}" ]]; then
        export WRAPPER_EXIT_CODE="${_INFER_RUNTIME_SIGNAL_CODE}"
        export RUN_ERROR="received ${_INFER_RUNTIME_SIGNAL_NAME}"
    elif (( child_rc != 0 )); then
        export RUN_ERROR="LightX2V inference exited with code ${child_rc}"
    fi

    _infer_runtime_finish_record || record_rc=$?

    if [[ -n "${_INFER_RUNTIME_SIGNAL_CODE}" ]]; then
        final_rc="${_INFER_RUNTIME_SIGNAL_CODE}"
    elif (( child_rc != 0 )); then
        final_rc="${child_rc}"
    elif (( record_rc != 0 )); then
        final_rc="${record_rc}"
    fi

    echo
    echo "==============================================================================="
    echo "finished_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "child_exit_code: ${child_rc}"
    echo "record_exit_code: ${record_rc}"
    echo "final_exit_code: ${final_rc}"
    echo "run.log: ${RUN_LOG_PATH}"
    echo "run.json: ${RUN_RECORD_PATH}"
    echo "result: ${RESULT_PATH}"
    echo "==============================================================================="

    _infer_runtime_disable_traps
    return "${final_rc}"
}

_infer_runtime_initialize
