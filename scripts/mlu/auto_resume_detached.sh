#!/usr/bin/env bash
set -euo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples

if [[ "${1:-}" == "--worker" ]]; then
    suite_id="${2:-}"
    wait_pid="${3:-}"
    suite_dir="${repo_path}/logs/mlu/infer/suites/${suite_id}"
    controller_log="${suite_dir}/controller.log"
    controller_pid_file="${suite_dir}/controller.pid"

    echo "[AutoResume] waiting for controller PID ${wait_pid}, suite=${suite_id}"
    while [[ -r "/proc/${wait_pid}/cmdline" ]]; do
        if ! tr '\0' ' ' <"/proc/${wait_pid}/cmdline" \
            | grep -Fq "run_infer_suite.py"; then
            break
        fi
        if ! tr '\0' ' ' <"/proc/${wait_pid}/cmdline" \
            | grep -Fq "${suite_id}"; then
            break
        fi
        sleep 20
    done

    printf '%s\n' "$$" >"${controller_pid_file}"
    echo "[AutoResume] starting repaired-case resume with PID $$"
    exec python "${repo_path}/scripts/mlu/run_infer_suite.py" \
        --resume "${suite_id}" >>"${controller_log}" 2>&1
fi

suite_id="${1:-}"
wait_pid="${2:-}"
if [[ -z "${suite_id}" ]]; then
    echo "Usage: $0 <suite_id> [controller_pid]" >&2
    exit 64
fi
if ! [[ "${suite_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "[ERROR] invalid suite id: ${suite_id}" >&2
    exit 64
fi

suite_dir="${repo_path}/logs/mlu/infer/suites/${suite_id}"
state_file="${suite_dir}/suite.json"
controller_pid_file="${suite_dir}/controller.pid"
supervisor_pid_file="${suite_dir}/auto_resume.pid"
supervisor_log="${suite_dir}/auto_resume.log"
if [[ ! -f "${state_file}" ]]; then
    echo "[ERROR] suite state does not exist: ${state_file}" >&2
    exit 66
fi
if [[ -z "${wait_pid}" && -f "${controller_pid_file}" ]]; then
    wait_pid="$(<"${controller_pid_file}")"
fi
if ! [[ "${wait_pid}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] controller PID is unavailable or invalid: ${wait_pid}" >&2
    exit 64
fi
if [[ -r "/proc/${wait_pid}/cmdline" ]]; then
    wait_command="$(tr '\0' ' ' <"/proc/${wait_pid}/cmdline")"
    if [[ "${wait_command}" != *"run_infer_suite.py"* \
        || "${wait_command}" != *"${suite_id}"* ]]; then
        echo "[ERROR] PID ${wait_pid} is not the controller for suite ${suite_id}" >&2
        exit 64
    fi
fi

if [[ -f "${supervisor_pid_file}" ]]; then
    old_supervisor_pid="$(<"${supervisor_pid_file}")"
    if [[ "${old_supervisor_pid}" =~ ^[0-9]+$ \
        && -r "/proc/${old_supervisor_pid}/cmdline" ]]; then
        old_supervisor_command="$(
            tr '\0' ' ' <"/proc/${old_supervisor_pid}/cmdline"
        )"
        if [[ "${old_supervisor_command}" == *"auto_resume_detached.sh --worker"* \
            || ( "${old_supervisor_command}" == *"run_infer_suite.py"* \
                && "${old_supervisor_command}" == *"${suite_id}"* ) ]]; then
            echo "[ERROR] auto-resume is already running with PID ${old_supervisor_pid}" >&2
            exit 75
        fi
    fi
fi

nohup setsid bash "$0" --worker "${suite_id}" "${wait_pid}" \
    >>"${supervisor_log}" 2>&1 </dev/null &
supervisor_pid=$!
printf '%s\n' "${supervisor_pid}" >"${supervisor_pid_file}"

echo "suite_id: ${suite_id}"
echo "wait_for_pid: ${wait_pid}"
echo "auto_resume_pid: ${supervisor_pid}"
echo "log: ${supervisor_log}"
