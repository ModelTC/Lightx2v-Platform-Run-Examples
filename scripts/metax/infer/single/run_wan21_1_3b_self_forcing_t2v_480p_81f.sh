#!/bin/bash
set -eo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/LightX2V-metax
model_path=/data/models/Wan2.1-T2V-1.3B
config_path="${repo_path}/configs/metax/single/wan21_1_3b_self_forcing_t2v_480p_81f.json"

case_id=wan21_1_3b_self_forcing_t2v_480p_81f
model_id=Wan2.1-T2V-1.3B-Self-Forcing
model_cls=wan2.1_sf
task=t2v
prompt="A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides."
negative_prompt=""
seed=42
result_ext=mp4
output_width=832
output_height=480
output_frames=81
infer_steps=4
offload_strategy=none
reference_script="${lightx2v_path}/scripts/self_forcing/run_wan_t2v_sf.sh"
world_size=1
parallel_strategy=single

run_group=single
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "${repo_path}/scripts/metax/metax_env.sh"

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
        --save_result_path "${result_path}"
}

run_infer lightx2v_infer
