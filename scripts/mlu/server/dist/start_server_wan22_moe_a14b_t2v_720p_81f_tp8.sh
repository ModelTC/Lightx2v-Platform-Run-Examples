#!/bin/bash
set -eo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/LightX2V-mlu
model_path=/data/models/Wan2.2-T2V-A14B
config_path="${repo_path}/configs/mlu/dist_8/wan22_moe_a14b_t2v_720p_81f_tp8.json"

server_platform=cambricon_mlu
case_id=wan22_moe_a14b_t2v_720p_81f_tp8
model_cls=wan2.2_moe
task=t2v
world_size=8
parallel_strategy=tp8
visible_devices=0,1,2,3,4,5,6,7

source "${repo_path}/scripts/lib/server_runtime.sh"
run_lightx2v_server
