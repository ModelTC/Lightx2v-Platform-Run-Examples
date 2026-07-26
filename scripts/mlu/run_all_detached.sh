#!/usr/bin/env bash
set -euo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
suite_id="${SUITE_ID:-mlu_$(date -u +%Y%m%dT%H%M%SZ)}"
suite_dir="${repo_path}/logs/mlu/infer/suites/${suite_id}"
controller_log="${suite_dir}/controller.log"
pid_file="${suite_dir}/controller.pid"

for required_command in python nohup setsid; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "[ERROR] required command is unavailable: ${required_command}" >&2
        exit 69
    fi
done

if [[ -e "${suite_dir}" ]]; then
    echo "[ERROR] suite directory already exists: ${suite_dir}" >&2
    exit 73
fi
mkdir -p "${suite_dir}"

nohup setsid python "${repo_path}/scripts/mlu/run_infer_suite.py" \
    --suite-id "${suite_id}" "$@" \
    >>"${controller_log}" 2>&1 </dev/null &
controller_pid=$!
printf '%s\n' "${controller_pid}" >"${pid_file}"

echo "suite_id: ${suite_id}"
echo "pid: ${controller_pid}"
echo "log: ${controller_log}"
echo "state: ${suite_dir}/suite.json"
