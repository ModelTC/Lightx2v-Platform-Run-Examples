#!/usr/bin/env bash
set -euo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
suite_id="${1:-}"
if [[ -z "${suite_id}" ]]; then
    echo "Usage: $0 <suite_id> [run_infer_suite.py options]" >&2
    exit 64
fi
shift
if ! [[ "${suite_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "[ERROR] invalid suite id: ${suite_id}" >&2
    exit 64
fi

suite_dir="${repo_path}/logs/mlu/infer/suites/${suite_id}"
state_file="${suite_dir}/suite.json"
controller_log="${suite_dir}/controller.log"
pid_file="${suite_dir}/controller.pid"
if [[ ! -f "${state_file}" ]]; then
    echo "[ERROR] suite state does not exist: ${state_file}" >&2
    exit 66
fi

if [[ -f "${pid_file}" ]]; then
    old_pid="$(<"${pid_file}")"
    if [[ "${old_pid}" =~ ^[0-9]+$ && -r "/proc/${old_pid}/cmdline" ]] \
        && tr '\0' ' ' <"/proc/${old_pid}/cmdline" \
            | grep -Fq "run_infer_suite.py"; then
        echo "[ERROR] suite controller is still running with PID ${old_pid}" >&2
        exit 75
    fi
fi

nohup setsid python "${repo_path}/scripts/mlu/run_infer_suite.py" \
    --resume "${suite_id}" "$@" \
    >>"${controller_log}" 2>&1 </dev/null &
controller_pid=$!
printf '%s\n' "${controller_pid}" >"${pid_file}"

echo "suite_id: ${suite_id}"
echo "pid: ${controller_pid}"
echo "log: ${controller_log}"
echo "state: ${state_file}"
