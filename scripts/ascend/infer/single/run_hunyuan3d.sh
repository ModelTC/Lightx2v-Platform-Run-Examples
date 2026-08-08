#!/bin/bash
set -euo pipefail

repo_path="${EXAMPLES_REPO_PATH:-/data/wushuo1/Lightx2v-Platform-Run-Examples}"
lightx2v_path="${LIGHTX2V_PATH:-/data/wushuo1/LightX2V}"
hy_repo="${HUNYUAN3D_REPO:-/data/wushuo1/Hunyuan3D-2.1-platform}"
model_path="${HUNYUAN3D_MODEL_PATH:-/data/wushuo1/models/Hunyuan3D-2.1}"
python_bin="${HUNYUAN3D_PYTHON:-/data/wushuo1/envs/hunyuan3d-ascend/bin/python}"
dino_path="${DINOV2_MODEL_PATH:-/data/wushuo1/models/dinov2-giant}"
config_path="${repo_path}/configs/ascend_npu/single/hunyuan3d_shape.json"
image_path="${HUNYUAN3D_IMAGE_PATH:-${hy_repo}/assets/demo.png}"
realesrgan_path="${hy_repo}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
shape_cache_dir="${HUNYUAN3D_SHAPE_CACHE:-/data/wushuo1/cache/hunyuan3d/shape}"

case_id=hunyuan3d
run_id="${RUN_ID:-$(date -u +"%Y%m%dT%H%M%SZ")_$$}"
log_root="${repo_path}/logs/ascend_npu/infer/single"
result_root="${repo_path}/results/ascend_npu/infer/single"
output_dir="${HUNYUAN3D_OUTPUT_DIR:-${result_root}/${case_id}/${run_id}}"
mesh_path="${output_dir}/shape.glb"
textured_path="${output_dir}/textured.glb"

export PLATFORM=ascend_npu
export TASK=infer
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export AI_DEVICE=npu
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/data/wushuo1/cache/huggingface/hunyuan3d_ascend_modules}"
export PYTHONPATH="${PYTHONPATH:-}"
export LIGHTX2V_PLATFORM_RUN_EXAMPLES_PATH="${repo_path}"
export LOG_DIR="${LOG_DIR:-${log_root}/${case_id}/${run_id}}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/run.log}"

source "${repo_path}/scripts/logging.sh"

if [[ ! -x "${python_bin}" ]]; then
    echo "Hunyuan3D Python is not executable: ${python_bin}" >&2
    exit 1
fi
if [[ ! -f "${config_path}" ]]; then
    echo "Missing Hunyuan3D config: ${config_path}" >&2
    exit 1
fi
if [[ ! -s "${image_path}" ]]; then
    echo "Missing Hunyuan3D input image: ${image_path}" >&2
    exit 1
fi
if [[ ! -d "${model_path}" ]]; then
    echo "Missing Hunyuan3D model directory: ${model_path}" >&2
    exit 1
fi
if [[ ! -d "${dino_path}" ]]; then
    echo "Missing DINO checkpoint directory: ${dino_path}" >&2
    exit 1
fi
if [[ ! -s "${realesrgan_path}" ]]; then
    echo "Missing RealESRGAN checkpoint: ${realesrgan_path}" >&2
    echo "Download it as documented in ${hy_repo}/README.md." >&2
    exit 1
fi

source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=FP16
export PROFILING_DEBUG_LEVEL=1

mkdir -p "${output_dir}" "${shape_cache_dir}" "${HF_MODULES_CACHE}"

echo "case_id: ${case_id}"
echo "run_id: ${run_id}"
echo "ASCEND_RT_VISIBLE_DEVICES: ${ASCEND_RT_VISIBLE_DEVICES}"
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

extra_args=()
if [[ "${HUNYUAN3D_NO_REMESH:-0}" == "1" ]]; then
    extra_args+=(--no_remesh)
fi

echo "=== Step 2/2: mesh texture (paint) ==="
"${python_bin}" "${lightx2v_path}/tools/postprocess/postprocess_paint.py" \
    --hy_repo "${hy_repo}" \
    --model_path "${model_path}" \
    --mesh_path "${mesh_path}" \
    --image_path "${image_path}" \
    --save_path "${textured_path}" \
    --device npu \
    --dino_ckpt_path "${dino_path}" \
    --realesrgan_ckpt_path "${realesrgan_path}" \
    --render_size 2048 \
    --texture_size 4096 \
    --max_num_view 6 \
    --resolution 512 \
    "${extra_args[@]}"

test -s "${textured_path}"
echo "Saved textured mesh: ${textured_path}"
echo "All outputs in: ${output_dir}"
echo "Log saved to: ${LOG_FILE}"
