#!/usr/bin/env bash
set -euo pipefail

server_dir=/data/Lightx2v-Platform-Run-Examples/scripts/metax/server
source "${server_dir}/launcher_lib.sh"
suite_id="${SUITE_ID:-metax_server_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
smoke_cases="z_image_turbo_t2i_1664x928_sp2"
metax_server_launch_new diagnostic "${suite_id}" "${smoke_cases}" "$@"
