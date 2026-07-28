#!/bin/bash
set -eo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/LightX2V-mlu
model_path=/data/models/Qwen-Image-2512
config_path="${repo_path}/configs/mlu/dist_8/qwen_image_2512_t2i_1664x928_cfg2_sp4.json"

server_platform=cambricon_mlu
case_id=qwen_image_2512_t2i_1664x928_cfg2_sp4
model_cls=qwen_image
task=t2i
world_size=8
parallel_strategy=cfg2_sp4
visible_devices=0,1,2,3,4,5,6,7

source "${repo_path}/scripts/lib/server_runtime.sh"
run_lightx2v_server
