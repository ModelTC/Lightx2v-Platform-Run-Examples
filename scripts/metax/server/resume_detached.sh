#!/usr/bin/env bash
set -euo pipefail

server_dir=/data/Lightx2v-Platform-Run-Examples/scripts/metax/server
source "${server_dir}/launcher_lib.sh"
suite_id="${1:-}"
if [[ -z "${suite_id}" ]]; then
    echo "Usage: $0 <suite_id> [controller options]" >&2
    exit 64
fi
shift
metax_server_resume "${suite_id}" "$@"
