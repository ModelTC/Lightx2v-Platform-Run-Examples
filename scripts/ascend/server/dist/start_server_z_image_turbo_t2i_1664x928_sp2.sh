#!/bin/bash
set -eo pipefail

repo_path=/data/wushuo1/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/wushuo1/LightX2V
model_path=/data/wushuo1/models/Z-Image-Turbo
config_path="${repo_path}/configs/ascend_npu/dist_2/z_image_turbo_t2i_1664x928_sp2.json"

case_id=z_image_turbo_t2i_1664x928_sp2
model_cls=z_image
task=t2i
world_size=2
parallel_strategy=sp2
visible_devices=0,1

source "${repo_path}/scripts/lib/server_runtime.sh"
run_lightx2v_server
