#!/bin/bash
set -eo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/LightX2V-mlu
model_path=/data/models/HunyuanVideo-1.5
config_path="${repo_path}/configs/mlu/single/hunyuan_video_15_t2v_480p_121f.json"

case_id=hunyuan_video_15_t2v_480p_121f
model_id=HunyuanVideo-1.5
model_cls=hunyuan_video_1.5
task=t2v
prompt="A close-up shot captures a scene on a polished, light-colored granite kitchen counter, illuminated by soft natural light from an unseen window. Initially, the frame focuses on a tall, clear glass filled with golden, translucent apple juice standing next to a single, shiny red apple with a green leaf still attached to its stem. The camera moves horizontally to the right. As the shot progresses, a white ceramic plate smoothly enters the frame, revealing a fresh arrangement of about seven or eight more apples, a mix of vibrant reds and greens, piled neatly upon it. A shallow depth of field keeps the focus sharply on the fruit and glass, while the kitchen backsplash in the background remains softly blurred. The scene is in a realistic style."
negative_prompt=""
seed=123
result_ext=mp4
output_width=848
output_height=480
output_frames=121
infer_steps=50
offload_strategy=none
reference_script="${lightx2v_path}/scripts/hunyuan_video_15/run_hy15_t2v_480p.sh"
world_size=1
parallel_strategy=single

run_group=single
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0}"
source "${repo_path}/scripts/mlu/mlu_env.sh"

source "${repo_path}/scripts/lib/infer_runtime.sh"
source "${lightx2v_path}/scripts/base/base.sh"
export PROFILING_DEBUG_LEVEL=0

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
