#!/bin/bash
set -euo pipefail

repo_path="${EXAMPLES_REPO_PATH:-/data/Lightx2v-Platform-Run-Examples}"
lightx2v_path="${LIGHTX2V_PATH:-/data/LightX2V}"
hy_repo="${HUNYUAN3D_REPO:-/data/Hunyuan3D-2.1-platform-metax}"
model_path="${HUNYUAN3D_MODEL_PATH:-/data/models/Hunyuan3D-2.1}"
python_bin="${HUNYUAN3D_PYTHON:-/opt/conda/bin/python}"
dino_path="${HUNYUAN3D_DINO_PATH:-/data/models/dinov2-giant}"
config_path="${CONFIG_PATH:-${repo_path}/configs/metax/single/hunyuan3d_shape.json}"
image_path="${HUNYUAN3D_IMAGE_PATH:-${hy_repo}/assets/demo.png}"
realesrgan_path="${HUNYUAN3D_REALESRGAN_PATH:-${hy_repo}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth}"
shape_cache_dir="${HUNYUAN3D_SHAPE_CACHE:-/data/.cache/hunyuan3d_metax/shape}"

case_id=hunyuan3d
run_id="${RUN_ID:-$(date -u +"%Y%m%dT%H%M%SZ")_$$}"
log_root="${repo_path}/logs/metax/infer/single"
result_root="${repo_path}/results/metax/infer/single"
output_dir="${HUNYUAN3D_OUTPUT_DIR:-${result_root}/${case_id}/${run_id}}"
mesh_path="${output_dir}/shape.glb"
textured_path="${output_dir}/textured.glb"

run_group=single
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "${repo_path}/scripts/metax/metax_env.sh"

export AI_DEVICE=cuda
export TASK=infer
export PYTORCH_DEFAULT_NCHW="${PYTORCH_DEFAULT_NCHW:-1}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/data/.cache/huggingface/hunyuan3d_metax_modules}"
export LIGHTX2V_PLATFORM_RUN_EXAMPLES_PATH="${repo_path}"
export LOG_DIR="${LOG_DIR:-${log_root}/${case_id}/${run_id}}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/run.log}"
export PATH="$(dirname "${python_bin}"):${PATH}"

torch_lib=$("${python_bin}" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
export LD_LIBRARY_PATH="${torch_lib}:${LD_LIBRARY_PATH}"
export PYTHONPATH="${hy_repo}:${hy_repo}/hy3dpaint:${hy_repo}/hy3dpaint/custom_rasterizer:${hy_repo}/hy3dpaint/DifferentiableRenderer${PYTHONPATH:+:${PYTHONPATH}}"

source "${repo_path}/scripts/logging.sh"

if [[ ! -s "${realesrgan_path}" ]]; then
    echo "Missing RealESRGAN checkpoint: ${realesrgan_path}" >&2
    echo "Download it as documented in ${hy_repo}/README.md." >&2
    exit 1
fi

extension_suffix=$("${python_bin}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
rasterizer_extension="${hy_repo}/hy3dpaint/custom_rasterizer/custom_rasterizer_kernel${extension_suffix}"
inpaint_extension="${hy_repo}/hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor${extension_suffix}"

if [[ "${HUNYUAN3D_FORCE_REBUILD:-0}" == "1" || ! -s "${rasterizer_extension}" || ! -s "${inpaint_extension}" ]]; then
    HUNYUAN3D_PYTHON="${python_bin}" MACA_PATH="${MACA_PATH}" "${hy_repo}/build_metax.sh"
fi

source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE="${HUNYUAN3D_DTYPE:-BF16}"
export PROFILING_DEBUG_LEVEL="${INFER_PROFILE_LEVEL:-2}"

mkdir -p "${output_dir}" "${shape_cache_dir}" "${HF_MODULES_CACHE}"

echo "case_id: ${case_id}"
echo "run_id: ${run_id}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "config_path: ${config_path}"
echo "output_dir: ${output_dir}"

echo "=== Step 1/2: shape generation ==="
(
    cd "${shape_cache_dir}"
    "${python_bin}" -m lightx2v.infer \
        --model_cls hunyuan3d \
        --task i23d \
        --model_path "${model_path}" \
        --config_json "${config_path}" \
        --image_path "${image_path}" \
        --save_result_path "${mesh_path}" \
        --seed 42
)

test -s "${mesh_path}"
echo "Saved mesh: ${mesh_path}"

paint_args=()
if [[ "${HUNYUAN3D_NO_REMESH:-0}" == "1" ]]; then
    paint_args+=(--no_remesh)
fi

echo "=== Step 2/2: mesh texture generation ==="
"${python_bin}" "${lightx2v_path}/tools/postprocess/postprocess_paint.py" \
    --hy_repo "${hy_repo}" \
    --model_path "${model_path}" \
    --mesh_path "${mesh_path}" \
    --image_path "${image_path}" \
    --save_path "${textured_path}" \
    --device cuda \
    --dino_ckpt_path "${dino_path}" \
    --realesrgan_ckpt_path "${realesrgan_path}" \
    --render_size 2048 \
    --texture_size 4096 \
    --max_num_view 6 \
    --resolution 512 \
    "${paint_args[@]}"

test -s "${textured_path}"
echo "Saved textured mesh: ${textured_path}"
echo "All outputs in: ${output_dir}"
echo "Log saved to: ${LOG_FILE}"
