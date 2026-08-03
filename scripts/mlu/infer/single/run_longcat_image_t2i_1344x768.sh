#!/bin/bash
set -eo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/LightX2V
model_path=/data/models/LongCat-Image
config_path="${repo_path}/configs/mlu/single/longcat_image_t2i_1344x768.json"

case_id=longcat_image_t2i_1344x768
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
reference_script="${lightx2v_path}/scripts/longcat/longcat_image_t2i.sh"
world_size=1
parallel_strategy=single
save_output=0

run_group=single
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0}"
source "${repo_path}/scripts/mlu/mlu_env.sh"

source "${repo_path}/scripts/lib/infer_runtime.sh"
source "${lightx2v_path}/scripts/base/base.sh"
export PROFILING_DEBUG_LEVEL="${INFER_PROFILE_LEVEL:-2}"

lightx2v_infer() {
    exec python -m lightx2v.infer \
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
