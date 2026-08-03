#!/bin/bash
set -eo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/LightX2V
model_path=/data/models/LTX-2
config_path="${repo_path}/configs/metax/dist_8/ltx2_3_22b_dev_s2v_768x512_241f_sp8.json"
audio_path="${lightx2v_path}/assets/inputs/audio/seko_input.mp3"

case_id=ltx2_3_22b_dev_s2v_768x512_241f_sp8
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
offload_strategy=model+component:gemma+rank0-io
reference_script="${lightx2v_path}/scripts/ltx2/ltx2_3/run_ltx2_3_s2v.sh"
world_size=8
parallel_strategy=sp8
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
        "${repo_path}/scripts/metax/ltx2_no_save_infer.py" \
        --seed "${seed}" \
        --model_cls "${model_cls}" \
        --task "${task}" \
        --model_path "${model_path}" \
        --config_json "${config_path}" \
        --audio_path "${audio_path}" \
        --prompt "${prompt}" \
        --negative_prompt "${negative_prompt}" \
        --return_result_tensor
}

run_infer lightx2v_infer
