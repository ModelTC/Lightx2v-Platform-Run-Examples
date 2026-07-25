#!/usr/bin/env bash

# Compatibility logger for legacy scripts. Current offline inference
# entrypoints use scripts/lib/infer_runtime.sh so that run.log, run.json and
# the generated artifact share one case_id/run_id.

logging_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_path="${LIGHTX2V_PLATFORM_RUN_EXAMPLES_PATH:-$(cd -- "${logging_script_dir}/.." && pwd)}"
platform_name="${PLATFORM:-ascend_npu}"
task_name="${TASK:-infer}"
task_name="${task_name,,}"

LOG_DIR="${LOG_DIR:-${repo_path}/logs/${platform_name}/${task_name}}"
mkdir -p "${LOG_DIR}"

script_name="$(basename "$0" .sh)"
timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${script_name}_${timestamp}.log}"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to ${LOG_FILE}"
