#!/usr/bin/env bash
set -euo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/LightX2V-metax
model_path=/data/models/Z-Image-Turbo
config_path="${repo_path}/configs/metax/dist_2/z_image_turbo_t2i_1664x928_sp2.json"
case_id=z_image_turbo_t2i_1664x928_sp2
model_cls=z_image
task=t2i
world_size=2
parallel_strategy=sp2
visible_devices=0,1

source "${repo_path}/scripts/metax/server/metax_server_runtime.sh"
run_lightx2v_server
