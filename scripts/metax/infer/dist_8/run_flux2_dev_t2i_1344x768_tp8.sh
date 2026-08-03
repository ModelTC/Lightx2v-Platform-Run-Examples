#!/bin/bash
set -eo pipefail

repo_path=/data/Lightx2v-Platform-Run-Examples
lightx2v_path=/data/LightX2V
model_path=/data/models/FLUX.2-dev
config_path="${repo_path}/configs/metax/dist_8/flux2_dev_t2i_1344x768_tp8.json"

case_id=flux2_dev_t2i_1344x768_tp8
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
offload_strategy=component:mistral3
reference_script="${lightx2v_path}/scripts/flux2/infer_flux2_dev_dist.sh"
world_size=8
parallel_strategy=tp8
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
        --return_result_tensor
}

run_infer lightx2v_infer
