#!/bin/bash
set -eo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/LightX2V
model_path=/data/models/LongCat-Image
config_path="${repo_path}/configs/metax/dist_8/longcat_image_t2i_1344x768_cfg2_sp4.json"

case_id=longcat_image_t2i_1344x768_cfg2_sp4
model_id=LongCat-Image
model_cls=longcat_image
task=t2i
prompt="一只小猫躺在沙发上"
negative_prompt=""
seed=42
result_ext=png
output_width=1344
output_height=768
output_frames=1
infer_steps=50
offload_strategy=none
reference_script="${lightx2v_path}/scripts/longcat/longcat_image_t2i_cfg_parallel.sh"
world_size=8
parallel_strategy=cfg2_sp4
save_output=0

run_group=dist_8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
source "${repo_path}/scripts/metax/metax_env.sh"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

source "${repo_path}/scripts/lib/infer_runtime.sh"
source "${lightx2v_path}/scripts/base/base.sh"
export PROFILING_DEBUG_LEVEL="${INFER_PROFILE_LEVEL:-2}"

lightx2v_infer() {
    exec torchrun --nnodes=1 \
        --nproc_per_node="${world_size}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        "${repo_path}/scripts/metax/longcat_profiled_infer.py" \
        --model_cls "${model_cls}" \
        --task "${task}" \
        --model_path "${model_path}" \
        --config_json "${config_path}" \
        --prompt "${prompt}" \
        --negative_prompt "${negative_prompt}" \
        --seed "${seed}" \
        --aspect_ratio "16:9" \
        --target_shape "${output_height}" "${output_width}" \
        --return_result_tensor
}

run_infer lightx2v_infer
