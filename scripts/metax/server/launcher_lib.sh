#!/usr/bin/env bash

metax_server_repo=/data/Lightx2v-Platform-Run-Examples
metax_server_controller="${metax_server_repo}/scripts/metax/server/run_service_suite.py"
metax_server_log_root="${metax_server_repo}/logs/metax/server"
metax_server_result_root="${metax_server_repo}/results/metax/server"

metax_server_validate_suite_id() {
    local suite_id=$1
    if ! [[ "${suite_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
        echo "[ERROR] invalid suite id: ${suite_id}" >&2
        return 64
    fi
}

metax_server_pid_matches() {
    local candidate_pid=$1
    local expected_suite=$2
    local -a command_line=()
    local argument next_argument
    local index
    local has_controller=0
    local has_suite=0

    [[ "${candidate_pid}" =~ ^[0-9]+$ ]] || return 1
    [[ -r "/proc/${candidate_pid}/cmdline" ]] || return 1
    mapfile -d '' -t command_line <"/proc/${candidate_pid}/cmdline" \
        || return 1
    for ((index = 0; index < ${#command_line[@]}; index++)); do
        argument="${command_line[index]}"
        next_argument=""
        if (( index + 1 < ${#command_line[@]} )); then
            next_argument="${command_line[index + 1]}"
        fi
        case "${argument}" in
            "${metax_server_controller}")
                has_controller=1
                ;;
            "--suite-id=${expected_suite}"|"--resume=${expected_suite}")
                has_suite=1
                ;;
            --suite-id|--resume)
                [[ "${next_argument}" == "${expected_suite}" ]] \
                    && has_suite=1
                ;;
        esac
    done
    (( has_controller == 1 && has_suite == 1 ))
}

metax_server_wait_for_identity() {
    local candidate_pid=$1
    local expected_suite=$2
    local attempt
    for ((attempt = 0; attempt < 50; attempt++)); do
        if metax_server_pid_matches "${candidate_pid}" "${expected_suite}"; then
            return 0
        fi
        [[ -d "/proc/${candidate_pid}" ]] || return 2
        sleep 0.1
    done
    return 1
}

metax_server_require_commands() {
    local required
    for required in python nohup setsid; do
        if ! command -v "${required}" >/dev/null 2>&1; then
            echo "[ERROR] required command is unavailable: ${required}" >&2
            return 69
        fi
    done
    if [[ ! -f "${metax_server_controller}" ]]; then
        echo "[ERROR] controller is missing: ${metax_server_controller}" >&2
        return 66
    fi
}

metax_server_reject_managed_arguments() {
    local argument
    for argument in "$@"; do
        case "${argument}" in
            --platform|--platform=*|--suite-id|--suite-id=*|\
            --resume|--resume=*|--only|--only=*|\
            --sample-count|--sample-count=*|\
            --warmup-count|--warmup-count=*|\
            --concurrency|--concurrency=*)
                echo "[ERROR] ${argument%%=*} is managed by the launcher" >&2
                return 64
                ;;
        esac
    done
}

metax_server_launch_new() {
    local mode=$1
    local suite_id=$2
    local selected_cases=$3
    shift 3
    local suite_log_dir="${metax_server_log_root}/${suite_id}"
    local suite_result_dir="${metax_server_result_root}/${suite_id}"
    local controller_log="${suite_log_dir}/controller.log"
    local pid_file="${suite_log_dir}/controller.pid"
    local sample_count suite_kind
    local -a only_arguments=()
    local controller_pid identity_status=0

    metax_server_validate_suite_id "${suite_id}"
    metax_server_require_commands
    metax_server_reject_managed_arguments "$@"
    if [[ -e "${suite_log_dir}" || -e "${suite_result_dir}" ]]; then
        echo "[ERROR] suite already exists: ${suite_id}" >&2
        return 73
    fi
    case "${mode}" in
        formal)
            sample_count=10
            suite_kind=formal
            ;;
        diagnostic)
            sample_count=1
            suite_kind=diagnostic
            [[ -n "${selected_cases}" ]] \
                && only_arguments=(--only "${selected_cases}")
            ;;
        *)
            echo "[ERROR] unsupported launch mode: ${mode}" >&2
            return 64
            ;;
    esac

    mkdir -p "${suite_log_dir}"
    if [[ "${suite_kind}" == "diagnostic" ]]; then
        printf '%s\n' \
            "DIAGNOSTIC ONLY: excluded from every formal report." \
            "Cases: ${selected_cases}" \
            "Load: 1 warm-up + 1 measured request, concurrency 1." \
            >"${suite_log_dir}/DIAGNOSTIC_ONLY"
    fi

    LIGHTX2V_SERVICE_SUITE_KIND="${suite_kind}" \
    INFER_PROFILE_LEVEL=0 \
    PROFILING_DEBUG_LEVEL=0 \
    nohup setsid python "${metax_server_controller}" \
        --suite-id "${suite_id}" \
        "${only_arguments[@]}" \
        --warmup-count 1 \
        --sample-count "${sample_count}" \
        --concurrency 1 \
        "$@" >>"${controller_log}" 2>&1 </dev/null &
    controller_pid=$!
    printf '%s\n' "${controller_pid}" >"${pid_file}"

    metax_server_wait_for_identity "${controller_pid}" "${suite_id}" \
        || identity_status=$?
    if (( identity_status != 0 )); then
        echo "[ERROR] controller identity validation failed; no signal sent" >&2
        echo "[ERROR] inspect ${controller_log}" >&2
        return 70
    fi
    echo "mode: ${mode}"
    echo "profiling_debug_level: 0"
    echo "suite_id: ${suite_id}"
    echo "pid: ${controller_pid}"
    echo "log: ${controller_log}"
    echo "state: ${suite_result_dir}/summary.json"
    echo "results: ${suite_result_dir}"
}

metax_server_resume() {
    local suite_id=$1
    shift
    local suite_log_dir="${metax_server_log_root}/${suite_id}"
    local suite_result_dir="${metax_server_result_root}/${suite_id}"
    local state_file="${suite_result_dir}/summary.json"
    local controller_log="${suite_log_dir}/controller.log"
    local pid_file="${suite_log_dir}/controller.pid"
    local old_pid controller_pid identity_status=0

    metax_server_validate_suite_id "${suite_id}"
    metax_server_require_commands
    metax_server_reject_managed_arguments "$@"
    if [[ ! -f "${state_file}" ]]; then
        echo "[ERROR] suite state does not exist: ${state_file}" >&2
        return 66
    fi
    if [[ -f "${pid_file}" ]]; then
        old_pid="$(<"${pid_file}")"
        if metax_server_pid_matches "${old_pid}" "${suite_id}"; then
            echo "[ERROR] controller is still running: PID ${old_pid}" >&2
            echo "[ERROR] no signal was sent" >&2
            return 75
        fi
    fi
    printf '\n[Resume] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >>"${controller_log}"
    INFER_PROFILE_LEVEL=0 \
    PROFILING_DEBUG_LEVEL=0 \
    nohup setsid python "${metax_server_controller}" \
        --resume "${suite_id}" \
        "$@" >>"${controller_log}" 2>&1 </dev/null &
    controller_pid=$!
    printf '%s\n' "${controller_pid}" >"${pid_file}"
    metax_server_wait_for_identity "${controller_pid}" "${suite_id}" \
        || identity_status=$?
    if (( identity_status != 0 )); then
        echo "[ERROR] resumed controller identity validation failed" >&2
        echo "[ERROR] no signal was sent; inspect ${controller_log}" >&2
        return 70
    fi
    echo "suite_id: ${suite_id}"
    echo "pid: ${controller_pid}"
    echo "log: ${controller_log}"
    echo "state: ${state_file}"
}
