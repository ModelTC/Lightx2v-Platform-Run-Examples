#!/bin/bash
set -e

repo_path=/data/wushuo1/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/wushuo1/LightX2V
model_path=/data/wushuo1/models/LTX-2
config_path="${repo_path}/configs/ascend_npu/single/ltx2_3_22b_dev_s2v_768x512_241f.json"
audio_path="${lightx2v_path}/assets/inputs/audio/seko_input.mp3"

case_id=ltx2_3_22b_dev_s2v_768x512_241f
model_id=LTX-2.3-22B-dev
model_cls=ltx2
task=ltx2_s2v
prompt="A person speaks clearly in a quiet room, natural lighting, cinematic medium shot."
negative_prompt="blurry, out of focus, overexposed, underexposed, low contrast, excessive noise, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, disfigured hands, artifacts, inconsistent perspective, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, off-sync audio, AI artifacts."
seed=42
result_ext=mp4
output_width=768
output_height=512
output_frames=241
infer_steps=30
offload_strategy=model+component:gemma
reference_script="${lightx2v_path}/scripts/platforms/ascend_npu/run_ltx2_3_s2v.sh"
world_size=1
parallel_strategy=single

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES=0

source "${repo_path}/scripts/lib/infer_runtime.sh"
source "${lightx2v_path}/scripts/base/base.sh"

lightx2v_infer() {
    exec python -m lightx2v.infer \
        --seed "${seed}" \
        --model_cls "${model_cls}" \
        --task "${task}" \
        --model_path "${model_path}" \
        --config_json "${config_path}" \
        --audio_path "${audio_path}" \
        --prompt "${prompt}" \
        --negative_prompt "${negative_prompt}" \
        --target_video_length "${output_frames}" \
        --save_result_path "${result_path}"
}

run_infer lightx2v_infer
