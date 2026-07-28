#!/usr/bin/env bash
set -euo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
report_tool="${repo_path}/scripts/metax/server/aggregate_service_reports.py"
if (( $# == 0 )); then
    echo "Usage: $0 --source-suite <formal_suite_id> [--source-suite ...] [--diagnostic-suite ...] [--output-dir ...]" >&2
    exit 64
fi

has_output=0
for argument in "$@"; do
    case "${argument}" in
        --output-dir|--output-dir=*) has_output=1 ;;
    esac
done
if (( has_output == 0 )); then
    output_dir="${repo_path}/results/metax/server/final_service_benchmark_$(date -u +%Y%m%d)"
    exec python "${report_tool}" "$@" --output-dir "${output_dir}"
fi
exec python "${report_tool}" "$@"
