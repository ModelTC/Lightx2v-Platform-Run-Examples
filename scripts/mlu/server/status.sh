#!/usr/bin/env bash
set -euo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
controller_path="${repo_path}/scripts/run_service_suite.py"
suite_id="${1:-}"
if [[ -z "${suite_id}" ]]; then
    echo "Usage: $0 <suite_id>" >&2
    exit 64
fi
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

controller_state=not_started
controller_pid=unknown
controller_pid_source=none
state_controller_pid=""
if [[ -f "${state_file}" ]]; then
    state_controller_pid="$(
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
fi
file_controller_pid=""
if [[ -f "${pid_file}" ]]; then
    file_controller_pid="$(<"${pid_file}")"
fi
if pid_is_suite_controller "${state_controller_pid}" "${suite_id}"; then
    controller_pid="${state_controller_pid}"
    controller_pid_source=state
    controller_state=running
elif pid_is_suite_controller "${file_controller_pid}" "${suite_id}"; then
    controller_pid="${file_controller_pid}"
    controller_pid_source=pid_file
    controller_state=running
elif [[ "${state_controller_pid}" =~ ^[0-9]+$ \
    && -r "/proc/${state_controller_pid}/cmdline" ]]; then
    controller_pid="${state_controller_pid}"
    controller_pid_source=state
    controller_state=stale_pid_reused
elif [[ "${file_controller_pid}" =~ ^[0-9]+$ \
    && -r "/proc/${file_controller_pid}/cmdline" ]]; then
    controller_pid="${file_controller_pid}"
    controller_pid_source=pid_file
    controller_state=stale_pid_reused
elif [[ -n "${state_controller_pid}" || -n "${file_controller_pid}" ]]; then
    controller_pid="${state_controller_pid:-${file_controller_pid}}"
    controller_pid_source="$(
        [[ -n "${state_controller_pid}" ]] && printf state || printf pid_file
    )"
    controller_state=stopped
fi

suite_state=unavailable
if [[ -f "${state_file}" ]]; then
    suite_state="$(
        awk -F'"' '
            /^[[:space:]]*"status"[[:space:]]*:/ {
                print $4
                exit
            }
        ' "${state_file}"
    )"
    suite_state="${suite_state:-unreadable}"
fi

diagnostic=no
if [[ -f "${suite_log_dir}/DIAGNOSTIC_ONLY" ]]; then
    diagnostic=yes
fi

echo "suite_id: ${suite_id}"
echo "suite_state: ${suite_state}"
echo "controller_state: ${controller_state}"
echo "controller_pid: ${controller_pid}"
echo "controller_pid_source: ${controller_pid_source}"
echo "diagnostic_only: ${diagnostic}"
echo "controller_log: ${controller_log}"
echo "state: ${state_file}"
echo "results: ${suite_result_dir}"
if [[ "${controller_state}" == "stale_pid_reused" ]]; then
    echo "warning: PID identity does not match this suite; no signal was sent"
fi
if [[ -f "${controller_log}" ]]; then
    echo
    echo "last ${TAIL_LINES:-20} controller log lines:"
    tail -n "${TAIL_LINES:-20}" "${controller_log}"
fi

if [[ "${controller_state}" == "running" || "${suite_state}" == "passed" ]]; then
    exit 0
fi
if [[ "${suite_state}" == "failed" ]]; then
    exit 1
fi
exit 3
