#!/bin/bash
set -e

repo_path=/data/wushuo1/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/wushuo1/LightX2V
model_path=/data/wushuo1/models/FLUX.2-dev
config_path="${repo_path}/configs/ascend_npu/single/flux2_dev_t2i_1344x768.json"

case_id=flux2_dev_t2i_1344x768
model_id=FLUX.2-dev
model_cls=flux2_dev
task=t2i
prompt="Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text 'BFL Diffusers' on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."
negative_prompt=""
seed=42
result_ext=png
output_width=1344
output_height=768
output_frames=1
infer_steps=50
offload_strategy=block
reference_script="${lightx2v_path}/scripts/flux2/infer_flux2_dev_offload.sh"
world_size=1
parallel_strategy=single

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES=0

source "${repo_path}/scripts/lib/infer_runtime.sh"
source "${lightx2v_path}/scripts/base/base.sh"

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
        --save_result_path "${result_path}"
}

run_infer lightx2v_infer
