#!/bin/bash
set -eo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/LightX2V
model_path=/data/models/Wan2.2-T2V-A14B
config_path="${repo_path}/configs/mlu/dist_8/wan22_moe_a14b_t2v_720p_81f_tp8.json"

case_id=wan22_moe_a14b_t2v_720p_81f_tp8
model_id=Wan2.2-T2V-A14B
model_cls=wan2.2_moe
task=t2v
prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
seed=42
result_ext=mp4
output_width=1280
output_height=720
output_frames=81
infer_steps=40
offload_strategy=none
reference_script="${lightx2v_path}/scripts/wan22/run_wan22_moe_t2v_tp.sh"
world_size=8
parallel_strategy=tp8
save_output=0

run_group=dist_8
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
source "${repo_path}/scripts/mlu/mlu_env.sh"
export MASTER_ADDR=127.0.0.1
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
