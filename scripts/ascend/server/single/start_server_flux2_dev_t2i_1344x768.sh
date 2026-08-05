#!/bin/bash
set -eo pipefail

repo_path=/data/wushuo1/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/wushuo1/LightX2V
model_path=/data/wushuo1/models/FLUX.2-dev
config_path="${repo_path}/configs/ascend_npu/single/flux2_dev_t2i_1344x768.json"

case_id=flux2_dev_t2i_1344x768
model_cls=flux2_dev
task=t2i
world_size=1
parallel_strategy=single
visible_devices=0

source "${repo_path}/scripts/lib/server_runtime.sh"
run_lightx2v_server
