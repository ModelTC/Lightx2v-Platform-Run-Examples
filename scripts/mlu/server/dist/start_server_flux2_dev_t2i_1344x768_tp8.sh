#!/bin/bash
set -eo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/LightX2V-mlu
model_path=/data/models/FLUX.2-dev
config_path="${repo_path}/configs/mlu/dist_8/flux2_dev_t2i_1344x768_tp8.json"

server_platform=cambricon_mlu
case_id=flux2_dev_t2i_1344x768_tp8
model_cls=flux2_dev
task=t2i
world_size=8
parallel_strategy=tp8
visible_devices=0,1,2,3,4,5,6,7

source "${repo_path}/scripts/lib/server_runtime.sh"
run_lightx2v_server
