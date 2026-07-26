#!/bin/bash
set -eo pipefail

repo_path=/data/wushuo1/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/wushuo1/LightX2V
model_path=/data/wushuo1/models/LTX-2
config_path="${repo_path}/configs/ascend_npu/dist_8/ltx2_3_22b_dev_s2v_768x512_241f_sp8.json"

case_id=ltx2_3_22b_dev_s2v_768x512_241f_sp8
model_cls=ltx2
task=ltx2_s2v
world_size=8
parallel_strategy=sp8
visible_devices=0,1,2,3,4,5,6,7

source "${repo_path}/scripts/lib/server_runtime.sh"
run_lightx2v_server
