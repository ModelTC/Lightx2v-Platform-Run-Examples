#!/bin/bash
set -eo pipefail

repo_path=/data/nvme1/wushuo/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/nvme1/wushuo/LightX2V
model_path=/data/nvme1/models/HunyuanVideo-1.5
config_path="${repo_path}/configs/nvidia/dist_8/hunyuan_video_15_t2v_720p_121f_cfg2_sp4.json"

case_id=hunyuan_video_15_t2v_720p_121f_cfg2_sp4
model_id=HunyuanVideo-1.5
model_cls=hunyuan_video_1.5
task=t2v
prompt="A close-up shot captures a scene on a polished, light-colored granite kitchen counter, illuminated by soft natural light from an unseen window. Initially, the frame focuses on a tall, clear glass filled with golden, translucent apple juice standing next to a single, shiny red apple with a green leaf still attached to its stem. The camera moves horizontally to the right. As the shot progresses, a white ceramic plate smoothly enters the frame, revealing a fresh arrangement of about seven or eight more apples, a mix of vibrant reds and greens, piled neatly upon it. A shallow depth of field keeps the focus sharply on the fruit and glass, while the kitchen backsplash in the background remains softly blurred. The scene is in a realistic style."
negative_prompt=""
seed=123
result_ext=mp4
output_width=1264
output_height=720
output_frames=121
infer_steps=50
offload_strategy=component:qwen25vl
reference_script="${lightx2v_path}/scripts/hunyuan_video_15/run_hy15_t2v_720p.sh"
world_size=8
parallel_strategy=cfg2_sp4
save_output=0

run_group=dist_8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
source "${repo_path}/scripts/nvidia/nvidia_env.sh"
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
        -m lightx2v.infer \
        --model_cls "${model_cls}" \
        --task "${task}" \
        --model_path "${model_path}" \
        --config_json "${config_path}" \
        --prompt "${prompt}" \
        --negative_prompt "${negative_prompt}" \
        --seed "${seed}"
}

run_infer lightx2v_infer
