#!/bin/bash
set -eo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/LightX2V-metax
model_path=/data/models/Qwen-Image-2512
config_path="${repo_path}/configs/metax/dist_8/qwen_image_2512_t2i_1664x928_cfg2_sp4.json"

case_id=qwen_image_2512_t2i_1664x928_cfg2_sp4
model_id=Qwen-Image-2512
model_cls=qwen_image
task=t2i
prompt='A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition, Ultra HD, 4K, cinematic composition.'
negative_prompt=" "
seed=42
result_ext=png
output_width=1664
output_height=928
output_frames=1
infer_steps=50
offload_strategy=model+component:qwen25vl
reference_script="${lightx2v_path}/scripts/platforms/metax/qwen_image_t2i_2512.sh"
world_size=8
parallel_strategy=cfg2_sp4

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
        -m lightx2v.infer \
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
