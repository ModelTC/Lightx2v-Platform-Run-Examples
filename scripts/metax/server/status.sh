#!/usr/bin/env bash
set -euo pipefail

server_dir=/data/Lightx2v-Platform-Run-Examples/scripts/metax/server
source "${server_dir}/launcher_lib.sh"
suite_id="${1:-}"
if [[ -z "${suite_id}" ]]; then
    echo "Usage: $0 <suite_id>" >&2
    exit 64
fi
metax_server_validate_suite_id "${suite_id}"

suite_log_dir="${metax_server_log_root}/${suite_id}"
suite_result_dir="${metax_server_result_root}/${suite_id}"
state_file="${suite_result_dir}/summary.json"
controller_log="${suite_log_dir}/controller.log"
pid_file="${suite_log_dir}/controller.pid"
controller_state=not_started
controller_pid=unknown
suite_state=unavailable

if [[ -f "${pid_file}" ]]; then
    controller_pid="$(<"${pid_file}")"
    if metax_server_pid_matches "${controller_pid}" "${suite_id}"; then
        controller_state=running
    elif [[ "${controller_pid}" =~ ^[0-9]+$ \
        && -r "/proc/${controller_pid}/cmdline" ]]; then
        controller_state=stale_pid_reused
    else
        controller_state=stopped
    fi
fi
if [[ -f "${state_file}" ]]; then
    suite_state="$(
        python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status","unreadable"))' \
            "${state_file}" 2>/dev/null || printf unreadable
    )"
fi

echo "suite_id: ${suite_id}"
echo "suite_state: ${suite_state}"
echo "controller_state: ${controller_state}"
echo "controller_pid: ${controller_pid}"
echo "diagnostic_only: $([[ -f "${suite_log_dir}/DIAGNOSTIC_ONLY" ]] && echo yes || echo no)"
echo "controller_log: ${controller_log}"
echo "state: ${state_file}"
echo "results: ${suite_result_dir}"
if [[ -f "${controller_log}" ]]; then
    echo
    echo "last ${TAIL_LINES:-20} controller log lines:"
    tail -n "${TAIL_LINES:-20}" "${controller_log}"
fi
[[ "${controller_state}" == "running" || "${suite_state}" == "passed" ]]
