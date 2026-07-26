#!/bin/bash
set -eo pipefail

repo_path=/data/wushuo1/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/wushuo1/LightX2V
model_path=/data/wushuo1/models/Wan2.1-T2V-1.3B
config_path="${repo_path}/configs/ascend_npu/dist_8/wan21_1_3b_t2v_480p_81f_cfg2_sp4.json"

case_id=wan21_1_3b_t2v_480p_81f_cfg2_sp4
model_cls=wan2.1
task=t2v
world_size=8
parallel_strategy=cfg2_sp4
visible_devices=0,1,2,3,4,5,6,7

source "${repo_path}/scripts/lib/server_runtime.sh"
run_lightx2v_server
