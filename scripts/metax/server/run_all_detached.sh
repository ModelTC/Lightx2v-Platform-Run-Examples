#!/usr/bin/env bash
set -euo pipefail

server_dir=/data/Lightx2v-Platform-Run-Examples/scripts/metax/server
source "${server_dir}/launcher_lib.sh"
suite_id="${SUITE_ID:-metax_server_$(date -u +%Y%m%dT%H%M%SZ)}"
metax_server_launch_new formal "${suite_id}" "" "$@"
