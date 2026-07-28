#!/usr/bin/env bash
set -euo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
controller_path="${repo_path}/scripts/run_service_suite.py"
suite_id="${1:-}"
if [[ -z "${suite_id}" ]]; then
    echo "Usage: $0 <suite_id> [run_service_suite.py options]" >&2
    exit 64
fi
shift
if ! [[ "${suite_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "[ERROR] invalid suite id: ${suite_id}" >&2
    exit 64
fi

suite_log_dir="${repo_path}/logs/mlu/server/${suite_id}"
suite_result_dir="${repo_path}/results/mlu/server/${suite_id}"
state_file="${suite_result_dir}/summary.json"
controller_log="${suite_log_dir}/controller.log"
pid_file="${suite_log_dir}/controller.pid"

pid_is_suite_controller() {
    local candidate_pid=$1
    local expected_suite=$2
    local -a command_line=()
    local argument
    local next_argument
    local index
    local has_controller=0
    local has_platform=0
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
            "${controller_path}")
                has_controller=1
                ;;
            --platform=mlu)
                has_platform=1
                ;;
            --platform)
                [[ "${next_argument}" == "mlu" ]] && has_platform=1
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
    (( has_controller == 1 && has_platform == 1 && has_suite == 1 ))
}

summary_matches_controller() {
    local candidate_pid=$1
    local summary_pid
    local summary_status

    [[ -f "${state_file}" ]] || return 1
    summary_pid="$(
        awk '
            /^[[:space:]]*"controller_pid"[[:space:]]*:/ {
                value=$0
                sub(/^[^:]*:[[:space:]]*/, "", value)
                sub(/,[[:space:]]*$/, "", value)
                print value
                exit
            }
        ' "${state_file}"
    )"
    summary_status="$(
        awk -F'"' '
            /^[[:space:]]*"status"[[:space:]]*:/ {
                print $4
                exit
            }
        ' "${state_file}"
    )"
    [[ "${summary_pid}" == "${candidate_pid}" ]] || return 1
    case "${summary_status}" in
        running|passed|failed)
            return 0
            ;;
    esac
    return 1
}

wait_for_controller_identity() {
    local candidate_pid=$1
    local expected_suite=$2
    local attempt
    local identity_seen=0
    local published_pid

    for ((attempt = 0; attempt < 600; attempt++)); do
        if pid_is_suite_controller "${candidate_pid}" "${expected_suite}"; then
            identity_seen=1
        fi
        published_pid=""
        if [[ -f "${pid_file}" ]]; then
            published_pid="$(<"${pid_file}")"
        fi
        if [[ "${published_pid}" == "${candidate_pid}" ]] \
            && (( identity_seen == 1 )) \
            && summary_matches_controller "${candidate_pid}"; then
            return 0
        fi
        if [[ ! -d "/proc/${candidate_pid}" ]]; then
            return 2
        fi
        sleep 0.1
    done
    return 1
}

for required_command in python nohup setsid env; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "[ERROR] required command is unavailable: ${required_command}" >&2
        exit 69
    fi
done
if [[ ! -f "${controller_path}" ]]; then
    echo "[ERROR] service controller does not exist: ${controller_path}" >&2
    exit 66
fi
if [[ ! -f "${state_file}" ]]; then
    echo "[ERROR] suite state does not exist: ${state_file}" >&2
    exit 66
fi

for argument in "$@"; do
    case "${argument}" in
        --platform|--platform=*|--suite-id|--suite-id=*|--resume|--resume=*|\
        --only|--only=*|--sample-count|--sample-count=*|\
        --warmup-count|--warmup-count=*|--concurrency|--concurrency=*)
            echo "[ERROR] ${argument%%=*} cannot be changed while resuming" >&2
            exit 64
            ;;
    esac
done

state_pid="$(
    awk '
        /^[[:space:]]*"controller_pid"[[:space:]]*:/ {
            value=$0
            sub(/^[^:]*:[[:space:]]*/, "", value)
            sub(/,[[:space:]]*$/, "", value)
            print value
            exit
        }
    ' "${state_file}"
)"
file_pid=""
if [[ -f "${pid_file}" ]]; then
    file_pid="$(<"${pid_file}")"
fi
for old_pid in "${state_pid}" "${file_pid}"; do
    [[ -n "${old_pid}" ]] || continue
    if pid_is_suite_controller "${old_pid}" "${suite_id}"; then
        echo "[ERROR] suite controller is still running with PID ${old_pid}" >&2
        echo "[ERROR] no signal was sent" >&2
        exit 75
    fi
    if [[ "${old_pid}" =~ ^[0-9]+$ && -r "/proc/${old_pid}/cmdline" ]]; then
        echo "[WARN] stale PID ${old_pid} belongs to another process; it will not be signaled" >&2
    fi
done

mkdir -p "${suite_log_dir}"
printf '\n[Resume] starting detached controller for suite %s\n' \
    "${suite_id}" >>"${controller_log}"
env \
    -u LIGHTX2V_SERVICE_SUITE_KIND \
    -u INFER_PROFILE_LEVEL \
    -u PROFILING_DEBUG_LEVEL \
    nohup setsid python "${controller_path}" \
    --platform mlu \
    --resume "${suite_id}" \
    "$@" >>"${controller_log}" 2>&1 </dev/null &
controller_pid=$!

identity_status=0
wait_for_controller_identity "${controller_pid}" "${suite_id}" \
    || identity_status=$?
if (( identity_status != 0 )); then
    if (( identity_status == 2 )); then
        echo "[ERROR] controller PID ${controller_pid} exited before identity validation" >&2
    else
        echo "[ERROR] controller did not acquire the lock, validate provenance and publish validated state for PID ${controller_pid} within 60 seconds" >&2
    fi
    echo "[ERROR] no signal was sent; inspect ${controller_log}" >&2
    exit 70
fi

echo "suite_id: ${suite_id}"
echo "pid: ${controller_pid}"
echo "log: ${controller_log}"
echo "state: ${state_file}"
echo "results: ${suite_result_dir}"
