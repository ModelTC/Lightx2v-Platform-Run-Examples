#!/bin/bash
set -eo pipefail

repo_path=/data/wushuo1/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/wushuo1/LightX2V
model_path=/data/wushuo1/models/HunyuanVideo-1.5
config_path="${repo_path}/configs/ascend_npu/dist_8/hunyuan_video_15_t2v_720p_121f_cfg2_sp4.json"

case_id=hunyuan_video_15_t2v_720p_121f_cfg2_sp4
model_cls=hunyuan_video_1.5
task=t2v
world_size=8
parallel_strategy=cfg2_sp4
visible_devices=0,1,2,3,4,5,6,7

source "${repo_path}/scripts/lib/server_runtime.sh"
run_lightx2v_server
