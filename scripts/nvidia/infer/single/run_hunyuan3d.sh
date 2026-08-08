#!/bin/bash
set -eo pipefail

repo_path="${EXAMPLES_REPO_PATH:-/data/nvme1/wushuo/Lightx2v-Platform-Run-Examples}"
lightx2v_path="${LIGHTX2V_PATH:-/data/nvme1/wushuo/LightX2V}"
model_path="${HUNYUAN3D_MODEL_PATH:-/data/nvme1/wushuo/hf_models/Hunyuan3D-2.1}"
hy_repo="${HUNYUAN3D_REPO_PATH:-/data/nvme1/wushuo/Hunyuan3D-2.1}"
config_path="${repo_path}/configs/nvidia/single/hunyuan3d_shape.json"
paint_python="${HUNYUAN3D_PAINT_PYTHON:-/data/nvme0/conda/bin/python3.11}"
paint_site_packages="${HUNYUAN3D_PAINT_SITE_PACKAGES:-/data/nvme1/wushuo/Codes_Bak/infer_codes/tencent/Hunyuan3D-2.1/.venv/lib/python3.11/site-packages}"
paint_extra_packages="${HUNYUAN3D_PAINT_EXTRA_PACKAGES:-${repo_path}/.deps/hunyuan3d-paint}"
dino_path="${DINOV2_MODEL_PATH:-/data/nvme1/wushuo/hf_models/dinov2-giant}"

# Hunyuan3D-2.1 full pipeline: image -> shape mesh (.glb) -> textured mesh (.glb)
# Install the Hunyuan3D-2.1 source repository before running this example:
#   pip install --no-cache-dir --no-build-isolation -v -e /path/to/Hunyuan3D-2.1
# The custom extensions require pybind11==2.13.4.

run_group=single
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
source "${repo_path}/scripts/nvidia/nvidia_env.sh"

case_id=hunyuan3d
run_id="${RUN_ID:-$(date -u +"%Y%m%dT%H%M%SZ")_$$}"
export LOG_DIR="${LOG_DIR:-${log_root}/${case_id}/${run_id}}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/run.log}"
source "${repo_path}/scripts/logging.sh"

echo "case_id: ${case_id}"
echo "run_id: ${run_id}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=FP16

image_path="${hy_repo}/assets/demo.png"
output_dir="${repo_path}/results/nvidia/infer/single/hunyuan3d"
mesh_path="${output_dir}/demo.glb"
textured_path="${output_dir}/demo_textured.glb"

mkdir -p "${output_dir}"

echo "=== Step 1/2: shape generation ==="
python -m lightx2v.infer \
    --model_cls hunyuan3d \
    --task i23d \
    --model_path "${model_path}" \
    --config_json "${config_path}" \
    --image_path "${image_path}" \
    --save_result_path "${mesh_path}" \
    --seed 42

echo "Saved mesh: ${mesh_path}"

echo "=== Step 2/2: mesh texture (paint) ==="
PYTHONPATH="${paint_extra_packages}:${paint_site_packages}:${hy_repo}:${hy_repo}/hy3dpaint/custom_rasterizer:${hy_repo}/hy3dpaint/DifferentiableRenderer:${lightx2v_path}:${PYTHONPATH}" \
"${paint_python}" "${lightx2v_path}/tools/postprocess/postprocess_paint.py" \
    --hy_repo "${hy_repo}" \
    --model_path "${model_path}" \
    --mesh_path "${mesh_path}" \
    --image_path "${image_path}" \
    --save_path "${textured_path}" \
    --dino_ckpt_path "${dino_path}" \
    --max_num_view 6 \
    --resolution 512

echo "Saved textured mesh: ${textured_path}"
echo "All outputs in: ${output_dir}"
echo "Log saved to: ${LOG_FILE}"
