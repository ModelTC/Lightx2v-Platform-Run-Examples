#!/usr/bin/env python3
"""Create and finalize a structured record for one inference run.

The model-specific shell entrypoints pass metadata through environment
variables.  This utility deliberately avoids importing accelerator-specific
torch extensions so that record keeping does not initialize a device runtime.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
ARTIFACT_VALIDATION_SCHEMA_VERSION = "1.0"
ARTIFACT_VALIDATOR = "run_record.validate_artifact"
ARTIFACT_VALIDATION_EXIT_CODE = 74
PROFILE_RE = re.compile(
    r"\[Profile\]\s+.*?\s+-\s+(?:Level\d+_Log\s+)?"
    r"(?P<label>.*?)\s+cost\s+(?P<seconds>[0-9]+(?:\.[0-9]+)?)\s+seconds"
)


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def env_int(name: str) -> int | None:
    value = env(name).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def env_float(name: str) -> float | None:
    value = env(name).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def first_env(*names: str) -> str:
    for name in names:
        value = env(name)
        if value:
            return value
    return ""


def first_env_int(*names: str) -> int | None:
    for name in names:
        value = env_int(name)
        if value is not None:
            return value
    return None


def device_context(platform_name: str) -> dict[str, Any]:
    platform_folded = platform_name.casefold()
    if "mlu" in platform_folded:
        device_type = "mlu"
    elif "metax" in platform_folded or "cuda" in platform_folded:
        device_type = "cuda"
    elif "ascend" in platform_folded or "npu" in platform_folded:
        device_type = "npu"
    else:
        device_type = env("DEVICE_TYPE", "npu")
    device_identity = f"{device_type}:{platform_name}".casefold()
    if "mlu" in device_identity:
        family = "mlu"
        default_visibility_environment = "MLU_VISIBLE_DEVICES"
        count_environments = ("DEVICE_COUNT", "MLU_COUNT", "WORLD_SIZE")
    elif (
        device_type.casefold() in {"cuda", "gpu"}
        or "metax" in device_identity
        or "cuda" in device_identity
    ):
        family = "cuda"
        default_visibility_environment = "CUDA_VISIBLE_DEVICES"
        count_environments = ("DEVICE_COUNT", "GPU_COUNT", "WORLD_SIZE")
    else:
        family = "npu"
        default_visibility_environment = "ASCEND_RT_VISIBLE_DEVICES"
        count_environments = ("DEVICE_COUNT", "NPU_COUNT", "WORLD_SIZE")
    visibility_environment = env(
        "DEVICE_VISIBILITY_ENV",
        default_visibility_environment,
    )
    visible_devices = first_env(
        visibility_environment,
        default_visibility_environment,
        "VISIBLE_DEVICES",
    )
    count = first_env_int(*count_environments)
    return {
        "type": device_type,
        "family": family,
        "visibility_environment": visibility_environment,
        "visible_devices": visible_devices,
        "selected_device": env("DEVICE_ID", visible_devices),
        "count": count,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_from_epoch(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def file_record(path_value: str) -> dict[str, Any]:
    path = Path(path_value).expanduser() if path_value else Path()
    exists = bool(path_value) and path.is_file()
    result: dict[str, Any] = {
        "path": str(path.resolve(strict=False)) if path_value else "",
        "exists": exists,
        "size_bytes": None,
        "sha256": None,
    }
    if exists:
        try:
            result["size_bytes"] = path.stat().st_size
        except OSError:
            pass
        result["sha256"] = sha256_file(path)
    return result


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def run_command(
    arguments: list[str],
    timeout: int = 10,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "command": arguments,
            "return_code": None,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
        }
    return {
        "command": arguments,
        "return_code": completed.returncode,
        "stdout": completed.stdout[-65536:],
        "stderr": completed.stderr[-65536:],
        "error": "",
    }


def unavailable_command(arguments: list[str], error: str) -> dict[str, Any]:
    return {
        "command": arguments,
        "return_code": None,
        "stdout": "",
        "stderr": "",
        "error": error,
    }


def device_monitor_record(
    platform_name: str,
    device_type: str,
) -> tuple[str, dict[str, Any]]:
    device_identity = f"{device_type}:{platform_name}".casefold()
    if "mlu" in device_identity:
        monitor_name = "cnmon"
        # The table view preserves every card's utilization, memory, firmware,
        # driver and process snapshot without the huge output of ``cnmon info``.
        arguments = ["cnmon", "all"]
    elif "metax" in device_identity:
        monitor_name = "mx-smi"
        arguments = ["mx-smi"]
    elif device_type.casefold() in {"cuda", "gpu"} or "cuda" in device_identity:
        monitor_name = "nvidia-smi"
        arguments = ["nvidia-smi"]
    else:
        monitor_name = "npu-smi"
        arguments = ["npu-smi", "info"]

    monitor_path = shutil.which(monitor_name)
    if not monitor_path:
        return monitor_name, unavailable_command(
            arguments,
            f"{monitor_name} is not available",
        )
    return monitor_name, run_command([monitor_path, *arguments[1:]], timeout=15)


def git_record(repository: str) -> dict[str, Any]:
    path = Path(repository).expanduser()
    record: dict[str, Any] = {
        "path": str(path.resolve(strict=False)),
        "is_git_repository": False,
        "commit": None,
        "branch": None,
        "dirty": None,
    }
    if not path.is_dir():
        return record

    resolved_path = str(path.resolve(strict=False))
    escaped_path = (
        resolved_path.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="lightx2v-run-record-git-",
    ) as git_config:
        git_config.write(f'[safe]\n\tdirectory = "{escaped_path}"\n')
        git_config.flush()
        git_environment = os.environ.copy()
        git_environment["GIT_CONFIG_GLOBAL"] = git_config.name
        git_command = ["git", "-C", str(path)]

        inside = run_command(
            [*git_command, "rev-parse", "--is-inside-work-tree"],
            environment=git_environment,
        )
        if inside["return_code"] != 0 or inside["stdout"].strip() != "true":
            return record
        record["is_git_repository"] = True

        commit = run_command(
            [*git_command, "rev-parse", "HEAD"],
            environment=git_environment,
        )
        if commit["return_code"] == 0:
            record["commit"] = commit["stdout"].strip() or None

        branch = run_command(
            [*git_command, "branch", "--show-current"],
            environment=git_environment,
        )
        if branch["return_code"] == 0:
            record["branch"] = branch["stdout"].strip() or None

        status = run_command(
            [
                *git_command,
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            timeout=30,
            environment=git_environment,
        )
        if status["return_code"] == 0:
            record["dirty"] = bool(status["stdout"].strip())
    return record


def package_versions() -> dict[str, str | None]:
    distributions = {
        "torch": "torch",
        "torch_npu": "torch-npu",
        "torch_mlu": "torch_mlu",
        "torch_mlu_ops": "torch_mlu_ops",
        "flash_attn": "flash-attn",
        "sageattention": "sageattention",
        "xformers": "xformers",
        "transformers": "transformers",
        "diffusers": "diffusers",
        "safetensors": "safetensors",
        "pillow": "Pillow",
        "numpy": "numpy",
        "loguru": "loguru",
        "lightx2v": "lightx2v",
    }
    versions: dict[str, str | None] = {}
    for output_name, distribution_name in distributions.items():
        try:
            versions[output_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            versions[output_name] = None
    return versions


def read_text_prefix(path: Path, limit: int = 16384) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as file_obj:
            return file_obj.read(limit)
    except OSError:
        return ""


def cann_record() -> dict[str, Any]:
    candidate_paths: list[Path] = []
    for variable_name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME"):
        root = env(variable_name)
        if root:
            candidate_paths.extend(
                [
                    Path(root) / "version.cfg",
                    Path(root) / "ascend_toolkit_install.info",
                ]
            )
    candidate_paths.extend(
        [
            Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
            Path("/usr/local/Ascend/ascend-toolkit/latest/ascend_toolkit_install.info"),
            Path("/etc/Ascend/ascend_cann_install.info"),
        ]
    )

    version_files: dict[str, str] = {}
    seen: set[str] = set()
    for path in candidate_paths:
        normalized = str(path.resolve(strict=False))
        if normalized in seen or not path.is_file():
            continue
        seen.add(normalized)
        contents = read_text_prefix(path)
        if contents:
            version_files[normalized] = contents

    return {
        "environment": {
            name: env(name)
            for name in (
                "ASCEND_HOME_PATH",
                "ASCEND_TOOLKIT_HOME",
                "ASCEND_HOME",
                "ASCEND_OPP_PATH",
            )
            if env(name)
        },
        "version_files": version_files,
    }


def resolve_reference_script() -> str:
    reference = env("REFERENCE_SCRIPT")
    if not reference:
        return ""
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = Path(env("LIGHTX2V_PATH")) / path
    return str(path.resolve(strict=False))


def config_snapshot(config_path: str) -> tuple[Any, str]:
    if not config_path:
        return None, "CONFIG_PATH is empty"
    try:
        with Path(config_path).open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj), ""
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"cannot read config snapshot: {exc}"


def build_start_record() -> dict[str, Any]:
    started_epoch_seconds = time.time()
    config, config_error = config_snapshot(env("CONFIG_PATH"))
    reference_script_path = resolve_reference_script()
    prompt = env("PROMPT")
    negative_prompt = env("NEGATIVE_PROMPT")
    errors = [config_error] if config_error else []
    platform_name = env("PLATFORM", "ascend_npu")
    device = device_context(platform_name)
    monitor_name, device_monitor = device_monitor_record(
        platform_name,
        device["type"],
    )
    hardware: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "device_type": device["type"],
        "device_memory_gib": env_float("DEVICE_MEMORY_GIB"),
        "device_monitor": {
            "name": monitor_name,
            **device_monitor,
        },
    }
    if monitor_name == "cnmon":
        hardware["cnmon"] = device_monitor
    elif monitor_name == "mx-smi":
        hardware["mx_smi"] = device_monitor
    elif monitor_name == "nvidia-smi":
        hardware["nvidia_smi"] = device_monitor
    else:
        # Preserve the original Ascend schema for existing report consumers.
        hardware["npu_smi"] = device_monitor

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": env("RUN_ID"),
        "status": "running",
        "benchmark": {
            "platform": platform_name,
            "mode": env("RUN_MODE", "single_sample_single_card"),
            "case_id": env("CASE_ID"),
            "model_id": env("MODEL_ID"),
            "task": env("TASK_TYPE"),
            "parallel_strategy": env("PARALLEL_STRATEGY"),
            "precision": env("DTYPE", "BF16"),
            "sensitive_layer_precision": env("SENSITIVE_LAYER_DTYPE", "None"),
            "offload_strategy": env("OFFLOAD_STRATEGY"),
            "target": {
                "width": env_int("OUTPUT_WIDTH"),
                "height": env_int("OUTPUT_HEIGHT"),
                "frames": env_int("OUTPUT_FRAMES"),
                "infer_steps": env_int("INFER_STEPS"),
            },
        },
        "input": {
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "negative_prompt": negative_prompt,
            "negative_prompt_sha256": hashlib.sha256(negative_prompt.encode("utf-8")).hexdigest(),
            "seed": env_int("SEED"),
        },
        "configuration": {
            "file": file_record(env("CONFIG_PATH")),
            "snapshot": config,
        },
        "execution": {
            "command": env("INFER_COMMAND"),
            "entry_script": file_record(env("SOURCE_SCRIPT")),
            "reference_script": file_record(reference_script_path),
            "lightx2v_path": str(Path(env("LIGHTX2V_PATH")).resolve(strict=False)),
            "model_path": str(Path(env("MODEL_PATH")).resolve(strict=False)),
            "device": {
                "type": device["type"],
                "family": device["family"],
                "visibility_environment": device["visibility_environment"],
                "visible_devices": device["visible_devices"],
                "selected_device": device["selected_device"],
                "count": device["count"],
            },
            "profiling_debug_level": env_int("PROFILING_DEBUG_LEVEL"),
        },
        "repositories": {
            "examples": git_record(env("REPO_ROOT")),
            "lightx2v": git_record(env("LIGHTX2V_PATH")),
        },
        "hardware": hardware,
        "software": {
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
            },
            "operating_system": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
            },
            "packages": package_versions(),
            "cann": cann_record(),
        },
        "paths": {
            "run_log": str(Path(env("RUN_LOG_PATH")).resolve(strict=False)),
            "run_record": str(Path(env("RUN_RECORD_PATH")).resolve(strict=False)),
            "result": str(Path(env("RESULT_PATH")).resolve(strict=False)),
        },
        "timing": {
            "started_at_utc": env("STARTED_AT_UTC") or utc_now(),
            "started_epoch_seconds": started_epoch_seconds,
            "finished_at_utc": None,
            "duration_seconds": None,
        },
        "exit": {
            "child_exit_code": None,
            "wrapper_exit_code": None,
            "error": "",
        },
        "artifact": {
            "path": str(Path(env("RESULT_PATH")).resolve(strict=False)),
            "exists": False,
            "size_bytes": None,
            "sha256": None,
            "format": env("RESULT_EXT").lower(),
            "valid": False,
            "validation": {
                "checked": False,
                "method": None,
                "errors": [],
            },
        },
        "metrics": {
            "wall_seconds": None,
            "profile_seconds": {
                "load_models": None,
                "text_encoder": None,
                "dit": None,
                "vae_decoder": None,
                "pipeline": None,
                "total": None,
            },
            "profile_seconds_by_label": {},
            "profile_rank_aggregation": None,
            "dit_seconds_per_step": None,
            "generated_frames_per_second": None,
            "images_per_second": None,
            "peak_device_memory_bytes": None,
        },
        "errors": errors,
    }
    return record


def validate_png(path: Path, expected_width: int | None, expected_height: int | None) -> dict[str, Any]:
    details: dict[str, Any] = {
        "schema_version": ARTIFACT_VALIDATION_SCHEMA_VERSION,
        "validator": ARTIFACT_VALIDATOR,
        "checked": True,
        "method": None,
        "errors": [],
        "width": None,
        "height": None,
        "image_format": None,
        "channel_extrema": None,
    }
    try:
        from PIL import Image

        details["method"] = "Pillow.Image.verify + RGB extrema"
        with Image.open(path) as image:
            details["width"], details["height"] = image.size
            details["image_format"] = image.format
            image.verify()
        # ``verify`` only checks the PNG container.  Decode the pixels in a
        # second pass so a numerically failed inference that was quantized to
        # an all-black/all-white image cannot be reported as successful.
        with Image.open(path) as image:
            extrema = image.convert("RGB").getextrema()
        details["channel_extrema"] = [
            [int(channel_min), int(channel_max)]
            for channel_min, channel_max in extrema
        ]
        if all(
            channel_min == 0 and channel_max == 0
            for channel_min, channel_max in extrema
        ):
            details["errors"].append("PNG pixels are uniformly black")
        elif all(
            channel_min == 255 and channel_max == 255
            for channel_min, channel_max in extrema
        ):
            details["errors"].append("PNG pixels are uniformly white")
    except ImportError:
        details["method"] = "PNG signature and IHDR"
        try:
            with path.open("rb") as file_obj:
                signature = file_obj.read(8)
                chunk_length = struct.unpack(">I", file_obj.read(4))[0]
                chunk_type = file_obj.read(4)
                chunk_data = file_obj.read(chunk_length)
            if signature != b"\x89PNG\r\n\x1a\n":
                details["errors"].append("invalid PNG signature")
            elif chunk_type != b"IHDR" or chunk_length != 13:
                details["errors"].append("missing or invalid PNG IHDR")
            else:
                details["width"], details["height"] = struct.unpack(">II", chunk_data[:8])
                details["image_format"] = "PNG"
        except (OSError, struct.error) as exc:
            details["errors"].append(f"cannot inspect PNG: {exc}")
    except Exception as exc:
        details["method"] = details["method"] or "Pillow.Image.verify"
        details["errors"].append(f"PNG verification failed: {exc}")

    if details["image_format"] not in (None, "PNG"):
        details["errors"].append(f"expected PNG, got {details['image_format']}")
    if expected_width is not None and details["width"] != expected_width:
        details["errors"].append(f"expected width {expected_width}, got {details['width']}")
    if expected_height is not None and details["height"] != expected_height:
        details["errors"].append(f"expected height {expected_height}, got {details['height']}")
    return details


def parse_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def validation_record(method: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_VALIDATION_SCHEMA_VERSION,
        "validator": ARTIFACT_VALIDATOR,
        "checked": True,
        "method": method,
        "errors": [],
    }


def is_mp4_container(format_name: Any) -> bool:
    if not isinstance(format_name, str):
        return False
    return "mp4" in {
        token.strip().casefold()
        for token in format_name.split(",")
        if token.strip()
    }


def mp4_pixel_validation_record(
    method: str | None = None,
    *,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "checked": method is not None,
        "method": method,
        "fallback_reason": fallback_reason,
        "decoded_frames": None,
        "channel_extrema": None,
        "errors": [],
    }


def finish_mp4_rgb_extrema(
    details: dict[str, Any],
    decoded_frames: int,
    extrema: list[list[int]],
) -> dict[str, Any]:
    details["decoded_frames"] = decoded_frames
    details["channel_extrema"] = extrema
    if decoded_frames <= 0:
        details["errors"].append("MP4 pixel decode returned no frames")
    elif all(
        channel_min == 0 and channel_max == 0
        for channel_min, channel_max in extrema
    ):
        details["errors"].append("MP4 pixels are uniformly black")
    elif all(
        channel_min == 255 and channel_max == 255
        for channel_min, channel_max in extrema
    ):
        details["errors"].append("MP4 pixels are uniformly white")
    return details


def inspect_mp4_rgb_extrema_pyav(
    path: Path,
    *,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Fully decode MP4 pixels with PyAV without counting plane padding."""
    details: dict[str, Any] = {
        **mp4_pixel_validation_record(
            "PyAV full decode at 64x64 RGB24",
            fallback_reason=fallback_reason,
        ),
    }
    try:
        import av
    except ImportError as exc:
        details["errors"].append(
            f"PyAV is not available for MP4 pixel validation: {exc}"
        )
        return details

    channel_minima = [255, 255, 255]
    channel_maxima = [0, 0, 0]
    decoded_frames = 0
    try:
        with av.open(str(path), mode="r") as container:
            video_streams = [
                stream
                for stream in container.streams
                if stream.type == "video"
            ]
            if not video_streams:
                details["errors"].append("MP4 has no video stream")
                return details
            stream = video_streams[0]
            for frame in container.decode(stream):
                rgb_frame = frame.reformat(
                    width=64,
                    height=64,
                    format="rgb24",
                )
                plane = rgb_frame.planes[0]
                payload = bytes(plane)
                line_size = int(plane.line_size)
                row_size = 64 * 3
                if line_size < row_size or len(payload) < line_size * 64:
                    raise ValueError(
                        "PyAV returned an invalid RGB plane layout"
                    )
                for row_index in range(64):
                    row_start = row_index * line_size
                    row = payload[row_start : row_start + row_size]
                    for channel in range(3):
                        samples = row[channel::3]
                        channel_minima[channel] = min(
                            channel_minima[channel],
                            min(samples),
                        )
                        channel_maxima[channel] = max(
                            channel_maxima[channel],
                            max(samples),
                        )
                decoded_frames += 1
    except Exception as exc:
        details["decoded_frames"] = decoded_frames
        details["errors"].append(
            f"PyAV could not fully decode MP4 pixels: {exc}"
        )
        return details

    extrema = [
        [channel_minima[channel], channel_maxima[channel]]
        for channel in range(3)
    ]
    return finish_mp4_rgb_extrema(details, decoded_frames, extrema)


def inspect_mp4_rgb_extrema(path: Path) -> dict[str, Any]:
    """Decode every frame at 64x64 and report per-channel RGB extrema."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return inspect_mp4_rgb_extrema_pyav(
            path,
            fallback_reason="ffmpeg is not available",
        )

    ffmpeg_check = run_command([ffmpeg, "-version"], timeout=10)
    if ffmpeg_check["return_code"] != 0:
        reason = (
            ffmpeg_check["stderr"].strip()
            or ffmpeg_check["error"]
            or f"exit code {ffmpeg_check['return_code']}"
        )
        return inspect_mp4_rgb_extrema_pyav(
            path,
            fallback_reason=f"ffmpeg is unusable: {reason}",
        )

    details = mp4_pixel_validation_record(
        "ffmpeg full decode at 64x64 RGB24"
    )
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        "scale=64:64:flags=fast_bilinear",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        details["errors"].append(f"cannot decode MP4 pixels: {exc}")
        return details
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        details["errors"].append(
            error or f"ffmpeg pixel decode exited with {completed.returncode}"
        )
        return details

    frame_bytes = 64 * 64 * 3
    payload = completed.stdout
    if not payload:
        details["errors"].append("ffmpeg pixel decode returned no frames")
        return details
    if len(payload) % frame_bytes:
        details["errors"].append(
            "ffmpeg pixel decode returned a partial RGB frame"
        )
        return details

    decoded_frames = len(payload) // frame_bytes
    extrema = [
        [min(payload[channel::3]), max(payload[channel::3])]
        for channel in range(3)
    ]
    return finish_mp4_rgb_extrema(details, decoded_frames, extrema)


def validate_mp4(
    path: Path,
    expected_width: int | None,
    expected_height: int | None,
    expected_frames: int | None,
    expected_audio: bool = False,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        **validation_record("ffprobe"),
        "expected_audio": expected_audio,
        "container_format": None,
        "codec": None,
        "width": None,
        "height": None,
        "frames": None,
        "frame_rate": None,
        "duration_seconds": None,
        "audio_codec": None,
        "audio_frames": None,
        "audio_samples": None,
        "pixel_validation": mp4_pixel_validation_record(),
    }
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        details["method"] = "PyAV full decode"
        try:
            import av
        except ImportError as exc:
            details["errors"].append(
                f"neither ffprobe nor PyAV is available for MP4 validation: {exc}"
            )
            return details
        try:
            with av.open(str(path), mode="r") as container:
                video_streams = [
                    stream
                    for stream in container.streams
                    if stream.type == "video"
                ]
                if not video_streams:
                    details["errors"].append("MP4 has no video stream")
                    return details
                stream = video_streams[0]
                details["container_format"] = getattr(
                    container.format, "name", None
                )
                details["codec"] = stream.codec_context.name
                details["width"] = parse_positive_int(stream.codec_context.width)
                details["height"] = parse_positive_int(stream.codec_context.height)
                rate = stream.average_rate or stream.base_rate or stream.guessed_rate
                details["frame_rate"] = str(rate) if rate is not None else None
                details["frames"] = sum(
                    1 for _frame in container.decode(stream)
                )
                if container.duration is not None:
                    details["duration_seconds"] = float(
                        container.duration / av.time_base
                    )
                elif stream.duration is not None and stream.time_base is not None:
                    details["duration_seconds"] = float(
                        stream.duration * stream.time_base
                    )
            if expected_audio:
                with av.open(str(path), mode="r") as container:
                    audio_streams = [
                        stream
                        for stream in container.streams
                        if stream.type == "audio"
                    ]
                    if not audio_streams:
                        details["errors"].append(
                            "MP4 has no audio stream for an audio-conditioned task"
                        )
                    else:
                        audio_stream = audio_streams[0]
                        details["audio_codec"] = (
                            audio_stream.codec_context.name
                        )
                        audio_frames = 0
                        audio_samples = 0
                        for frame in container.decode(audio_stream):
                            audio_frames += 1
                            audio_samples += frame.samples
                        details["audio_frames"] = audio_frames
                        details["audio_samples"] = audio_samples
        except Exception as exc:
            details["errors"].append(f"PyAV could not fully decode MP4: {exc}")
            return details
    else:
        probe = run_command(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames,nb_read_frames,r_frame_rate:format=format_name,duration",
                "-of",
                "json",
                str(path),
            ],
            timeout=120,
        )
        if probe["return_code"] != 0:
            error = (
                probe["stderr"].strip()
                or probe["error"]
                or f"ffprobe exited with {probe['return_code']}"
            )
            details["errors"].append(error)
            return details

        try:
            payload = json.loads(probe["stdout"])
            if not isinstance(payload, dict):
                raise TypeError("top-level ffprobe value is not an object")
            stream = payload["streams"][0]
            if not isinstance(stream, dict):
                raise TypeError("ffprobe video stream is not an object")
            format_payload = payload["format"]
            if not isinstance(format_payload, dict):
                raise TypeError("ffprobe format is not an object")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            details["errors"].append(f"invalid ffprobe output: {exc}")
            return details

        details["codec"] = stream.get("codec_name")
        details["container_format"] = format_payload.get("format_name")
        details["width"] = parse_positive_int(stream.get("width"))
        details["height"] = parse_positive_int(stream.get("height"))
        details["frames"] = parse_positive_int(
            stream.get("nb_read_frames")
        ) or parse_positive_int(stream.get("nb_frames"))
        details["frame_rate"] = stream.get("r_frame_rate")
        try:
            details["duration_seconds"] = float(
                format_payload.get("duration")
            )
        except (TypeError, ValueError):
            details["duration_seconds"] = None
        if expected_audio:
            audio_probe = run_command(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=codec_name,nb_frames,nb_read_frames",
                    "-of",
                    "json",
                    str(path),
                ],
                timeout=120,
            )
            if audio_probe["return_code"] != 0:
                error = (
                    audio_probe["stderr"].strip()
                    or audio_probe["error"]
                    or f"audio ffprobe exited with {audio_probe['return_code']}"
                )
                details["errors"].append(error)
            else:
                try:
                    audio_payload = json.loads(audio_probe["stdout"])
                    if not isinstance(audio_payload, dict):
                        raise TypeError(
                            "top-level audio ffprobe value is not an object"
                        )
                    audio_stream = audio_payload["streams"][0]
                    if not isinstance(audio_stream, dict):
                        raise TypeError(
                            "ffprobe audio stream is not an object"
                        )
                    details["audio_codec"] = audio_stream.get("codec_name")
                    details["audio_frames"] = parse_positive_int(
                        audio_stream.get("nb_read_frames")
                    ) or parse_positive_int(audio_stream.get("nb_frames"))
                except (
                    json.JSONDecodeError,
                    KeyError,
                    IndexError,
                    TypeError,
                ) as exc:
                    details["errors"].append(
                        f"invalid audio ffprobe output: {exc}"
                    )

    if not is_mp4_container(details["container_format"]):
        details["errors"].append(
            "expected an MP4 container, got "
            f"{details['container_format']!r}"
        )
    if not details["codec"]:
        details["errors"].append("video codec is missing")
    if details["width"] is None or details["height"] is None:
        details["errors"].append("video dimensions are missing")
    if details["frames"] is None or details["frames"] <= 0:
        details["errors"].append("video has no decodable frames")
    if details["duration_seconds"] is not None and details["duration_seconds"] <= 0:
        details["errors"].append("video duration is not positive")
    if expected_width is not None and details["width"] != expected_width:
        details["errors"].append(f"expected width {expected_width}, got {details['width']}")
    if expected_height is not None and details["height"] != expected_height:
        details["errors"].append(f"expected height {expected_height}, got {details['height']}")
    if expected_frames is not None and details["frames"] is not None and details["frames"] != expected_frames:
        details["errors"].append(f"expected {expected_frames} frames, got {details['frames']}")
    if expected_audio:
        if not details["audio_codec"]:
            details["errors"].append("audio codec is missing")
        if (
            details["audio_frames"] is None
            or details["audio_frames"] <= 0
        ):
            details["errors"].append(
                "audio stream has no decodable frames"
            )
        if (
            details["method"] == "PyAV full decode"
            and (
                details["audio_frames"] is None
                or details["audio_frames"] <= 0
                or details["audio_samples"] is None
                or details["audio_samples"] <= 0
            )
        ):
            details["errors"].append("audio stream has no decodable samples")

    pixel_validation = inspect_mp4_rgb_extrema(path)
    pixel_frames = parse_positive_int(
        pixel_validation.get("decoded_frames")
    )
    metadata_frames = parse_positive_int(details["frames"])
    if (
        pixel_frames is not None
        and metadata_frames is not None
        and pixel_frames != metadata_frames
    ):
        pixel_validation["errors"].append(
            "MP4 pixel decoder produced "
            f"{pixel_frames} frames, metadata decoder produced "
            f"{metadata_frames}"
        )
    details["pixel_validation"] = pixel_validation
    details["errors"].extend(pixel_validation["errors"])
    return details


def validate_artifact(
    path: Path,
    result_format: str,
    target: dict[str, Any],
    *,
    expected_audio: bool = False,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "path": str(path.resolve(strict=False)),
        "exists": path.is_file(),
        "size_bytes": None,
        "sha256": None,
        "format": result_format,
        "valid": False,
        "validation": validation_record(),
    }
    if not path.is_file():
        artifact["validation"]["errors"].append("result file does not exist")
        return artifact
    try:
        artifact["size_bytes"] = path.stat().st_size
    except OSError as exc:
        artifact["validation"]["errors"].append(f"cannot stat result file: {exc}")
        return artifact
    if artifact["size_bytes"] <= 0:
        artifact["validation"]["errors"].append("result file is empty")
        return artifact

    artifact["sha256"] = sha256_file(path)
    if artifact["sha256"] is None:
        artifact["validation"]["errors"].append("cannot calculate result SHA-256")
        return artifact

    if result_format == "png":
        validation = validate_png(path, target.get("width"), target.get("height"))
    elif result_format == "mp4":
        validation = validate_mp4(
            path,
            target.get("width"),
            target.get("height"),
            target.get("frames"),
            expected_audio=expected_audio,
        )
    else:
        validation = {
            **validation_record(),
            "errors": [f"unsupported result format: {result_format}"],
        }
    artifact["validation"] = validation
    artifact["valid"] = not validation["errors"]
    return artifact


def parse_profile_metrics(
    log_path: Path,
    *,
    use_slowest_rank: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    aggregates: dict[str, dict[str, float | int]] = {}
    if log_path.is_file():
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as file_obj:
                for line in file_obj:
                    match = PROFILE_RE.search(line)
                    if not match:
                        continue
                    label = match.group("label").strip()
                    seconds = float(match.group("seconds"))
                    aggregate = aggregates.setdefault(
                        label,
                        {
                            "count": 0,
                            "total_seconds": 0.0,
                            "last_seconds": 0.0,
                            "min_seconds": seconds,
                            "max_seconds": seconds,
                        },
                    )
                    aggregate["count"] = int(aggregate["count"]) + 1
                    aggregate["total_seconds"] = float(aggregate["total_seconds"]) + seconds
                    aggregate["last_seconds"] = seconds
                    aggregate["min_seconds"] = min(float(aggregate["min_seconds"]), seconds)
                    aggregate["max_seconds"] = max(float(aggregate["max_seconds"]), seconds)
        except OSError:
            pass

    def selected_for(label: str) -> float | None:
        label_lower = label.casefold()
        for actual_label, aggregate in aggregates.items():
            if actual_label.casefold() == label_lower:
                field = "max_seconds" if use_slowest_rank else "last_seconds"
                return round(float(aggregate[field]), 6)
        return None

    canonical = {
        "load_models": selected_for("Load models"),
        "text_encoder": selected_for("Run Text Encoder"),
        "dit": selected_for("Run DiT"),
        "vae_decoder": selected_for("Run VAE Decoder"),
        "pipeline": selected_for("RUN pipeline"),
        "total": selected_for("Total Cost"),
    }
    rounded_aggregates: dict[str, Any] = {}
    for label, aggregate in aggregates.items():
        rounded_aggregates[label] = {
            "count": aggregate["count"],
            "total_seconds": round(float(aggregate["total_seconds"]), 6),
            "last_seconds": round(float(aggregate["last_seconds"]), 6),
            "min_seconds": round(float(aggregate["min_seconds"]), 6),
            "max_seconds": round(float(aggregate["max_seconds"]), 6),
        }
    return canonical, rounded_aggregates


def finish_record(record: dict[str, Any]) -> tuple[dict[str, Any], int]:
    finished_epoch_seconds = time.time()
    started_epoch_seconds = record.get("timing", {}).get("started_epoch_seconds")
    try:
        duration_seconds = max(0.0, finished_epoch_seconds - float(started_epoch_seconds))
    except (TypeError, ValueError):
        duration_seconds = None

    target = record.get("benchmark", {}).get("target", {})
    result_format = record.get("artifact", {}).get("format") or env("RESULT_EXT").lower()
    result_path = Path(env("RESULT_PATH") or record.get("paths", {}).get("result", ""))
    benchmark_no_save = env("BENCHMARK_NO_SAVE") == "1"
    if benchmark_no_save:
        artifact = {
            "path": None,
            "format": result_format,
            "size_bytes": None,
            "sha256": None,
            "valid": True,
            "validation": {
                **validation_record(),
                "skipped": True,
                "reason": "BENCHMARK_NO_SAVE=1",
            },
        }
    else:
        artifact = validate_artifact(
            result_path,
            result_format,
            target,
            expected_audio=record.get("benchmark", {}).get("task")
            in {"s2v", "ltx2_s2v"},
        )

    log_path = Path(env("RUN_LOG_PATH") or record.get("paths", {}).get("run_log", ""))
    device_count = record.get("execution", {}).get("device", {}).get("count")
    use_slowest_rank = isinstance(device_count, int) and device_count > 1
    profile_seconds, profile_by_label = parse_profile_metrics(
        log_path,
        use_slowest_rank=use_slowest_rank,
    )

    infer_steps = target.get("infer_steps")
    output_frames = target.get("frames")
    dit_seconds = profile_seconds.get("dit")
    pipeline_seconds = profile_seconds.get("pipeline")
    dit_seconds_per_step = None
    if dit_seconds is not None and isinstance(infer_steps, int) and infer_steps > 0:
        dit_seconds_per_step = round(dit_seconds / infer_steps, 6)

    generated_frames_per_second = None
    images_per_second = None
    if pipeline_seconds is not None and pipeline_seconds > 0:
        if result_format == "mp4" and isinstance(output_frames, int) and output_frames > 0:
            generated_frames_per_second = round(output_frames / pipeline_seconds, 6)
        elif result_format == "png":
            images_per_second = round(1.0 / pipeline_seconds, 9)

    child_exit_code = env_int("CHILD_EXIT_CODE")
    wrapper_exit_code = env_int("WRAPPER_EXIT_CODE")
    run_error = env("RUN_ERROR")
    interrupted = (
        child_exit_code in (130, 143)
        or wrapper_exit_code in (130, 143)
        or "SIGINT" in run_error
        or "SIGTERM" in run_error
    )
    succeeded = (
        child_exit_code == 0
        and wrapper_exit_code == 0
        and artifact["valid"]
        and not run_error
    )
    status = "succeeded" if succeeded else ("interrupted" if interrupted else "failed")

    errors = list(record.get("errors") or [])
    if run_error and run_error not in errors:
        errors.append(run_error)
    for validation_error in artifact["validation"].get("errors", []):
        if validation_error not in errors:
            errors.append(validation_error)
    if child_exit_code is None and not run_error:
        errors.append("inference child did not report an exit code")

    record["status"] = status
    record["timing"].update(
        {
            "finished_at_utc": utc_from_epoch(finished_epoch_seconds),
            "finished_epoch_seconds": finished_epoch_seconds,
            "duration_seconds": round(duration_seconds, 6) if duration_seconds is not None else None,
        }
    )
    record["exit"] = {
        "child_exit_code": child_exit_code,
        "wrapper_exit_code": wrapper_exit_code,
        "error": run_error,
    }
    record["artifact"] = artifact
    record["metrics"] = {
        "wall_seconds": round(duration_seconds, 6) if duration_seconds is not None else None,
        "profile_seconds": profile_seconds,
        "profile_seconds_by_label": profile_by_label,
        "profile_rank_aggregation": "max" if use_slowest_rank else "last",
        "dit_seconds_per_step": dit_seconds_per_step,
        "generated_frames_per_second": generated_frames_per_second,
        "images_per_second": images_per_second,
        "peak_device_memory_bytes": None,
    }
    record["errors"] = errors

    exit_code = 0
    if child_exit_code == 0 and wrapper_exit_code == 0 and not artifact["valid"]:
        exit_code = ARTIFACT_VALIDATION_EXIT_CODE
    return record, exit_code


def command_start(record_path: Path) -> int:
    record = build_start_record()
    atomic_write_json(record_path, record)
    print(f"[RunRecord] initialized: {record_path}")
    monitor = record["hardware"]["device_monitor"]
    monitor_name = monitor["name"]
    if monitor["return_code"] == 0 and monitor["stdout"].strip():
        if monitor_name in {"cnmon", "mx-smi"}:
            print(
                f"[Environment] {monitor_name} snapshot captured in run.json "
                f"({len(monitor['stdout'].encode('utf-8'))} bytes)"
            )
        else:
            print(f"[Environment] {monitor_name} info")
            print(monitor["stdout"].rstrip())
    else:
        reason = (
            monitor["stderr"].strip()
            or monitor["error"]
            or f"exit code {monitor['return_code']}"
        )
        print(
            f"[RunRecord] warning: {monitor_name} info unavailable: {reason}"
        )
    return 0


def command_finish(record_path: Path) -> int:
    try:
        with record_path.open("r", encoding="utf-8") as file_obj:
            record = json.load(file_obj)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[RunRecord] cannot load {record_path}: {exc}", file=sys.stderr)
        return 70

    record, exit_code = finish_record(record)
    atomic_write_json(record_path, record)
    artifact = record["artifact"]
    print(
        "[RunRecord] "
        f"status={record['status']} "
        f"duration_seconds={record['timing']['duration_seconds']} "
        f"result={artifact['path']} "
        f"size_bytes={artifact['size_bytes']} "
        f"sha256={artifact['sha256']} "
        f"record={record_path}"
    )
    return exit_code


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"start", "finish"}:
        print(f"usage: {Path(sys.argv[0]).name} start|finish", file=sys.stderr)
        return 64
    record_path_value = env("RUN_RECORD_PATH")
    if not record_path_value:
        print("RUN_RECORD_PATH is required", file=sys.stderr)
        return 64
    record_path = Path(record_path_value).expanduser().resolve(strict=False)
    if sys.argv[1] == "start":
        return command_start(record_path)
    return command_finish(record_path)


if __name__ == "__main__":
    raise SystemExit(main())
