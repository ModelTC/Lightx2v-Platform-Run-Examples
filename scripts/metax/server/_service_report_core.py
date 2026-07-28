#!/usr/bin/env python3
"""Pinned report-validation core for the MetaX Level-0 report adapter."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from service_benchmark_common import sha256_file

REPO_PATH = Path("/data/Lightx2v-Platform-Run-Examples")
PLATFORMS = ("ascend_npu", "mlu", "metax")
PLATFORM_LABELS = {
    "ascend_npu": "Ascend NPU",
    "mlu": "Cambricon MLU",
    "metax": "MetaX C500",
}
MEASURED_REQUESTS = 10
WARMUP_REQUESTS = 1
CONCURRENCY = 1


@dataclass(frozen=True)
class CaseContract:
    task_kind: str
    world_size: int
    parallel_strategy: str
    width: int
    height: int
    frames: int
    infer_steps: int
    extension: str
    audio_required: bool = False


CASE_CONTRACTS: dict[str, CaseContract] = {
    "z_image_turbo_t2i_1664x928_sp2": CaseContract(
        "t2i", 2, "sp2", 1664, 928, 1, 9, ".png"
    ),
    "flux2_dev_t2i_1344x768_tp8": CaseContract(
        "t2i", 8, "tp8", 1344, 768, 1, 50, ".png"
    ),
    "longcat_image_t2i_1344x768_cfg2_sp4": CaseContract(
        "t2i", 8, "cfg2_sp4", 1344, 768, 1, 50, ".png"
    ),
    "qwen_image_2512_t2i_1664x928_cfg2_sp4": CaseContract(
        "t2i", 8, "cfg2_sp4", 1664, 928, 1, 50, ".png"
    ),
    "wan21_1_3b_t2v_480p_81f_cfg2_sp4": CaseContract(
        "t2v", 8, "cfg2_sp4", 832, 480, 81, 50, ".mp4"
    ),
    "wan22_moe_a14b_t2v_480p_81f_cfg2_sp4": CaseContract(
        "t2v", 8, "cfg2_sp4", 832, 480, 81, 40, ".mp4"
    ),
    "wan22_moe_a14b_t2v_480p_81f_tp8": CaseContract(
        "t2v", 8, "tp8", 832, 480, 81, 40, ".mp4"
    ),
    "wan22_moe_a14b_t2v_720p_81f_cfg2_sp4": CaseContract(
        "t2v", 8, "cfg2_sp4", 1280, 720, 81, 40, ".mp4"
    ),
    "wan22_moe_a14b_t2v_720p_81f_tp8": CaseContract(
        "t2v", 8, "tp8", 1280, 720, 81, 40, ".mp4"
    ),
    "hunyuan_video_15_t2v_480p_121f_cfg2_sp4": CaseContract(
        "t2v", 8, "cfg2_sp4", 848, 480, 121, 50, ".mp4"
    ),
    "hunyuan_video_15_t2v_720p_121f_cfg2_sp4": CaseContract(
        "t2v", 8, "cfg2_sp4", 1264, 720, 121, 50, ".mp4"
    ),
    "ltx2_3_22b_dev_s2v_768x512_241f_sp8": CaseContract(
        "s2v",
        8,
        "sp8",
        768,
        512,
        241,
        30,
        ".mp4",
        audio_required=True,
    ),
}
EXPECTED_CASES = tuple(CASE_CONTRACTS)

COMMON_REMEDIATIONS = (
    {
        "scope": "服务测试清理",
        "problem": (
            "失败用例可能残留服务进程，在下一用例开始前继续占用设备显存。"
        ),
        "resolution": (
            "按精确的 suite run ID 追踪并终止该批次拥有的服务子进程，"
            "随后检查设备显存。"
        ),
    },
)
ASCEND_REMEDIATIONS = (
    {
        "scope": "Ascend 64GB 上的 LTX2.3 S2V",
        "problem": (
            "全量权重 241 帧配置会在 Gemma 编码阶段显存不足，"
            "或在 VAE 分块解码时保留计算图。"
        ),
        "resolution": (
            "采用 Gemma 分阶段 CPU offload，移除未使用的视觉与 logits 路径，"
            "在 inference mode 下执行 VAE 分块解码，并启用 torch_npu "
            "expandable-segments 分配器。正式结果仍使用全量权重、241 帧、"
            "30 个扩散步，且 double_precision_rope=false。"
        ),
    },
)

TASK_START_PATTERN = re.compile(r"\bProcessing task (?P<task_id>\S+)")
TASK_END_PATTERN = re.compile(
    r"\bTask (?P<task_id>\S+) completed successfully\b"
)
DIT_COST_PATTERN = re.compile(
    r"\bRank (?P<rank>\d+) - Level1_Log .*?"
    r"\binfer_main cost (?P<seconds>\d+(?:\.\d+)?) seconds\b"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file_obj:
            file_obj.write(text)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def result_root(platform: str) -> Path:
    return REPO_PATH / "results" / platform / "server"


def log_root(platform: str) -> Path:
    return REPO_PATH / "logs" / platform / "server"


def rebase_repo_path(value: str | Path) -> Path:
    """Resolve current and historical absolute paths against this checkout."""
    raw_text = str(value)
    if not raw_text:
        raise ValueError("empty path in suite record")
    raw = Path(raw_text)
    candidates: list[Path] = [raw if raw.is_absolute() else REPO_PATH / raw]
    if REPO_PATH.name in raw.parts:
        repo_index = raw.parts.index(REPO_PATH.name)
        candidates.append(REPO_PATH.joinpath(*raw.parts[repo_index + 1 :]))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def suite_path(value: str, platform: str) -> Path:
    raw = Path(value)
    path = rebase_repo_path(raw) if raw.is_absolute() else result_root(
        platform
    ) / raw
    return path.resolve(strict=True)


def case_result_dir(case: dict[str, Any], source_suite: Path) -> Path:
    recorded = case.get("result_dir")
    if isinstance(recorded, str) and recorded:
        path = rebase_repo_path(recorded)
        if path.is_dir():
            return path
    fallback = source_suite / str(case.get("case_id", ""))
    return fallback.resolve()


def case_log_dir(
    case: dict[str, Any], source_suite: Path, platform: str
) -> Path:
    recorded = case.get("log_dir")
    if isinstance(recorded, str) and recorded:
        path = rebase_repo_path(recorded)
        if path.is_dir():
            return path
    return (log_root(platform) / source_suite.name / case["case_id"]).resolve()


def throughput_per_minute(throughput: dict[str, Any]) -> float:
    for key, value in throughput.items():
        if (
            key.endswith("_per_minute")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            return float(value)
    raise ValueError(f"missing per-minute throughput: {throughput}")


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def distribution(values: Sequence[float]) -> dict[str, float]:
    sorted_values = sorted(float(value) for value in values)
    return {
        "p50_s": percentile(sorted_values, 50),
        "p90_s": percentile(sorted_values, 90),
        "avg_s": statistics.fmean(sorted_values),
        "min_s": min(sorted_values),
        "max_s": max(sorted_values),
    }


def validate_png(path: Path, contract: CaseContract) -> dict[str, Any]:
    errors: list[str] = []
    details: dict[str, Any] = {
        "kind": "png",
        "expected": {
            "width": contract.width,
            "height": contract.height,
            "frames": 1,
        },
    }
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            image_format = image.format
            mode = image.mode
        details.update(
            {
                "decoder": "Pillow.Image.verify + Image.load",
                "format": image_format,
                "mode": mode,
                "width": width,
                "height": height,
                "frames": 1,
            }
        )
        if image_format != "PNG":
            errors.append(f"decoded format {image_format!r} is not PNG")
        if (width, height) != (contract.width, contract.height):
            errors.append(
                f"decoded size {width}x{height} != "
                f"{contract.width}x{contract.height}"
            )
    except (ImportError, OSError, SyntaxError, ValueError) as exception:
        errors.append(f"PNG decode failed: {type(exception).__name__}: {exception}")
    details["status"] = "passed" if not errors else "failed"
    details["errors"] = errors
    return details


def ffprobe(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "format=format_name,duration:"
            "stream=index,codec_type,codec_name,width,height,"
            "nb_frames,nb_read_frames,duration,sample_rate,channels"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exception:
        return None, f"{type(exception).__name__}: {exception}"
    if completed.returncode != 0:
        return None, completed.stderr.strip() or (
            f"ffprobe exited with {completed.returncode}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exception:
        return None, f"invalid ffprobe JSON: {exception}"
    if not isinstance(value, dict):
        return None, "ffprobe JSON is not an object"
    return value, None


def positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def validate_mp4(path: Path, contract: CaseContract) -> dict[str, Any]:
    errors: list[str] = []
    details: dict[str, Any] = {
        "kind": "mp4",
        "expected": {
            "width": contract.width,
            "height": contract.height,
            "frames": contract.frames,
            "audio_required": contract.audio_required,
        },
        "decoder": "ffprobe -count_frames",
    }
    probe, probe_error = ffprobe(path)
    if probe_error:
        errors.append(f"ffprobe failed: {probe_error}")
    elif probe is None:
        errors.append("ffprobe returned no probe data")
    else:
        streams = probe.get("streams")
        if not isinstance(streams, list):
            streams = []
        videos = [
            stream
            for stream in streams
            if isinstance(stream, dict)
            and stream.get("codec_type") == "video"
        ]
        audios = [
            stream
            for stream in streams
            if isinstance(stream, dict)
            and stream.get("codec_type") == "audio"
        ]
        if len(videos) != 1:
            errors.append(f"expected exactly one video stream, found {len(videos)}")
        if videos:
            video = videos[0]
            width = positive_int(video.get("width"))
            height = positive_int(video.get("height"))
            frame_count = positive_int(video.get("nb_read_frames"))
            frame_count_source = "nb_read_frames"
            if frame_count is None:
                frame_count = positive_int(video.get("nb_frames"))
                frame_count_source = "nb_frames"
            details["video"] = {
                "codec": video.get("codec_name"),
                "width": width,
                "height": height,
                "frames": frame_count,
                "frame_count_source": frame_count_source,
                "duration_seconds": video.get("duration"),
            }
            if (width, height) != (contract.width, contract.height):
                errors.append(
                    f"video size {width}x{height} != "
                    f"{contract.width}x{contract.height}"
                )
            if frame_count != contract.frames:
                errors.append(
                    f"video frames {frame_count} != {contract.frames}"
                )
        details["audio_streams"] = [
            {
                "codec": audio.get("codec_name"),
                "sample_rate": positive_int(audio.get("sample_rate")),
                "channels": positive_int(audio.get("channels")),
                "duration_seconds": audio.get("duration"),
            }
            for audio in audios
        ]
        if contract.audio_required and not audios:
            errors.append("required LTX audio stream is missing")
        for index, audio in enumerate(details["audio_streams"]):
            if audio["sample_rate"] is None or audio["channels"] is None:
                errors.append(
                    f"audio stream {index} has invalid sample rate/channels"
                )
        container = probe.get("format")
        details["container"] = container
        format_name = (
            container.get("format_name")
            if isinstance(container, dict)
            else None
        )
        if not isinstance(format_name, str) or "mp4" not in format_name.split(
            ","
        ):
            errors.append(f"container format {format_name!r} is not MP4")
    details["status"] = "passed" if not errors else "failed"
    details["errors"] = errors
    return details


def validate_media(path: Path, contract: CaseContract) -> dict[str, Any]:
    if path.suffix.lower() != contract.extension:
        return {
            "status": "failed",
            "kind": contract.extension.lstrip("."),
            "expected": {
                "extension": contract.extension,
                "width": contract.width,
                "height": contract.height,
                "frames": contract.frames,
                "audio_required": contract.audio_required,
            },
            "errors": [
                f"artifact extension {path.suffix!r} != {contract.extension!r}"
            ],
        }
    if contract.extension == ".png":
        return validate_png(path, contract)
    return validate_mp4(path, contract)


def verify_case_artifacts(
    case: dict[str, Any], contract: CaseContract
) -> dict[str, Any]:
    benchmark = case["benchmark_result"]
    requests = benchmark.get("request_results")
    failures: list[str] = []
    total_bytes = 0
    verified = 0
    media_verified = 0
    request_records: list[dict[str, Any]] = []
    if not isinstance(requests, list):
        requests = []
        failures.append("benchmark request_results is not a list")
    if len(requests) != MEASURED_REQUESTS:
        failures.append(
            f"request_results count {len(requests)} != {MEASURED_REQUESTS}"
        )
    seen_task_ids: set[str] = set()
    for request in requests:
        if not isinstance(request, dict):
            failures.append("request result is not an object")
            continue
        task_id = str(request.get("task_id", ""))
        record: dict[str, Any] = {
            "task_id": task_id,
            "artifact_path_recorded": request.get("artifact_path"),
            "sha256_expected": request.get("artifact_sha256"),
            "bytes_expected": request.get("artifact_bytes"),
            "sha256_verified": False,
            "media_validation": None,
        }
        request_records.append(record)
        if not task_id or task_id in seen_task_ids:
            failures.append(f"invalid or duplicate task_id: {task_id!r}")
        seen_task_ids.add(task_id)
        if request.get("phase") not in (None, "measure"):
            failures.append(f"{task_id}: request phase is not measure")
        if not request.get("ok"):
            failures.append(f"{task_id}: request is not successful")
            continue
        artifact_value = request.get("artifact_path")
        if not isinstance(artifact_value, str) or not artifact_value:
            failures.append(f"{task_id}: artifact_path is missing")
            continue
        artifact_path = rebase_repo_path(artifact_value)
        record["artifact_path"] = str(artifact_path)
        if not artifact_path.is_file():
            failures.append(f"{artifact_path}: missing")
            continue
        expected_bytes = request.get("artifact_bytes")
        actual_bytes = artifact_path.stat().st_size
        record["bytes_actual"] = actual_bytes
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or actual_bytes <= 0
            or actual_bytes != expected_bytes
        ):
            failures.append(
                f"{artifact_path}: size {actual_bytes} != {expected_bytes}"
            )
            continue
        expected_sha = request.get("artifact_sha256")
        actual_sha = sha256_file(artifact_path)
        record["sha256_actual"] = actual_sha
        if not isinstance(expected_sha, str) or actual_sha != expected_sha:
            failures.append(
                f"{artifact_path}: sha256 {actual_sha} != {expected_sha}"
            )
            continue
        record["sha256_verified"] = True
        verified += 1
        total_bytes += actual_bytes
        media_validation = validate_media(artifact_path, contract)
        record["media_validation"] = media_validation
        if media_validation["status"] != "passed":
            failures.extend(
                f"{artifact_path}: {message}"
                for message in media_validation["errors"]
            )
            continue
        media_verified += 1
    expected = MEASURED_REQUESTS
    return {
        "status": (
            "passed"
            if not failures
            and verified == expected
            and media_verified == expected
            else "failed"
        ),
        "artifacts_expected": expected,
        "artifacts_verified": verified,
        "media_verified": media_verified,
        "artifact_bytes": total_bytes,
        "failures": failures,
        "requests": request_records,
    }


def model_config_path(
    case_id: str, benchmark: dict[str, Any], platform: str
) -> Path | None:
    run_config = benchmark.get("run_config")
    if isinstance(run_config, dict):
        model_config = run_config.get("model_config")
        if isinstance(model_config, dict):
            recorded = model_config.get("path")
            if isinstance(recorded, str) and recorded:
                path = rebase_repo_path(recorded)
                if path.is_file():
                    return path
    matches = sorted(
        (REPO_PATH / "configs" / platform).rglob(f"{case_id}.json")
    )
    return matches[0].resolve() if len(matches) == 1 else None


def expected_steps_from_config(
    case_id: str,
    benchmark: dict[str, Any],
    platform: str,
    contract: CaseContract,
) -> tuple[int | None, Path | None, str | None]:
    config_path = model_config_path(case_id, benchmark, platform)
    if config_path is None:
        return None, None, "model config could not be resolved uniquely"
    try:
        config = read_json(config_path)
    except (OSError, ValueError) as exception:
        return None, config_path, (
            f"model config cannot be read: {type(exception).__name__}: "
            f"{exception}"
        )
    infer_steps = positive_int(config.get("infer_steps"))
    if infer_steps is None:
        return None, config_path, "model config infer_steps is missing or invalid"
    if infer_steps != contract.infer_steps:
        return None, config_path, (
            f"model config infer_steps {infer_steps} does not match "
            f"case contract {contract.infer_steps}"
        )
    return infer_steps, config_path, None


def unavailable_dit_profile(
    *,
    source_log: Path,
    expected_requests: int,
    expected_steps: int | None,
    world_size: int,
    reasons: Sequence[str],
    config_path: Path | None = None,
    task_ids_observed: int = 0,
) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "authoritative": False,
        "method": (
            "server.log task boundaries; per-rank infer_main occurrence order; "
            "rank max for each request/step"
        ),
        "warmup_excluded": True,
        "source_log": str(source_log),
        "config_path": str(config_path) if config_path else None,
        "expected_requests": expected_requests,
        "task_ids_observed": task_ids_observed,
        "requests_profiled": 0,
        "expected_steps_per_request": expected_steps,
        "world_size": world_size,
        "rank_samples_expected": (
            expected_requests * expected_steps * world_size
            if expected_steps is not None
            else None
        ),
        "rank_samples_observed": 0,
        "summary": None,
        "all_step_summary": None,
        "requests": [],
        "reasons": list(dict.fromkeys(reasons)),
    }


def parse_dit_step_profile(
    *,
    server_log: Path,
    task_ids: Sequence[str],
    expected_steps: int | None,
    world_size: int,
    config_path: Path | None,
    initial_error: str | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if initial_error:
        reasons.append(initial_error)
    if expected_steps is None:
        return unavailable_dit_profile(
            source_log=server_log,
            expected_requests=MEASURED_REQUESTS,
            expected_steps=None,
            world_size=world_size,
            reasons=reasons,
            config_path=config_path,
            task_ids_observed=len(task_ids),
        )
    if len(task_ids) != MEASURED_REQUESTS or len(set(task_ids)) != len(
        task_ids
    ):
        reasons.append(
            f"measured task IDs are not {MEASURED_REQUESTS} unique requests"
        )
    if not server_log.is_file():
        reasons.append("server.log is missing")
        return unavailable_dit_profile(
            source_log=server_log,
            expected_requests=MEASURED_REQUESTS,
            expected_steps=expected_steps,
            world_size=world_size,
            reasons=reasons,
            config_path=config_path,
            task_ids_observed=len(task_ids),
        )

    records: dict[str, dict[str, Any]] = {
        task_id: {
            "started": 0,
            "completed": 0,
            "rank_seconds": {},
            "boundary_errors": [],
        }
        for task_id in task_ids
    }
    active_task: str | None = None
    try:
        with server_log.open(
            "r", encoding="utf-8", errors="replace"
        ) as file_obj:
            for line_number, line in enumerate(file_obj, 1):
                start_match = TASK_START_PATTERN.search(line)
                if start_match:
                    next_task = start_match.group("task_id")
                    if active_task in records:
                        records[active_task]["boundary_errors"].append(
                            f"line {line_number}: next task started before "
                            "completion"
                        )
                    active_task = next_task
                    if next_task in records:
                        records[next_task]["started"] += 1

                cost_match = DIT_COST_PATTERN.search(line)
                if cost_match and active_task in records:
                    rank = int(cost_match.group("rank"))
                    seconds = float(cost_match.group("seconds"))
                    rank_seconds = records[active_task]["rank_seconds"]
                    rank_seconds.setdefault(rank, []).append(seconds)

                end_match = TASK_END_PATTERN.search(line)
                if end_match:
                    completed_task = end_match.group("task_id")
                    if completed_task in records:
                        records[completed_task]["completed"] += 1
                        if active_task != completed_task:
                            records[completed_task]["boundary_errors"].append(
                                f"line {line_number}: completion did not match "
                                "active task"
                            )
                    if active_task == completed_task:
                        active_task = None
    except OSError as exception:
        reasons.append(f"cannot read server.log: {exception}")
        return unavailable_dit_profile(
            source_log=server_log,
            expected_requests=MEASURED_REQUESTS,
            expected_steps=expected_steps,
            world_size=world_size,
            reasons=reasons,
            config_path=config_path,
            task_ids_observed=len(task_ids),
        )

    request_profiles: list[dict[str, Any]] = []
    all_step_seconds: list[float] = []
    request_average_seconds: list[float] = []
    rank_samples_observed = 0
    expected_ranks = set(range(world_size))
    for task_id in task_ids:
        raw = records[task_id]
        rank_seconds = raw["rank_seconds"]
        rank_samples_observed += sum(len(values) for values in rank_seconds.values())
        request_errors = list(raw["boundary_errors"])
        if raw["started"] != 1:
            request_errors.append(
                f"Processing task boundary count {raw['started']} != 1"
            )
        if raw["completed"] != 1:
            request_errors.append(
                f"successful completion boundary count {raw['completed']} != 1"
            )
        observed_ranks = set(rank_seconds)
        if observed_ranks != expected_ranks:
            request_errors.append(
                f"ranks {sorted(observed_ranks)} != "
                f"{sorted(expected_ranks)}"
            )
        rank_sample_counts = {
            str(rank): len(rank_seconds.get(rank, []))
            for rank in sorted(expected_ranks | observed_ranks)
        }
        for rank in expected_ranks:
            count = len(rank_seconds.get(rank, []))
            if count != expected_steps:
                request_errors.append(
                    f"rank {rank} step samples {count} != {expected_steps}"
                )

        request_profile: dict[str, Any] = {
            "task_id": task_id,
            "status": "unavailable" if request_errors else "available",
            "expected_steps": expected_steps,
            "rank_sample_counts": rank_sample_counts,
            "step_seconds_rank_max": None,
            "step_latency_seconds": None,
            "errors": request_errors,
        }
        if not request_errors:
            step_seconds = [
                max(rank_seconds[rank][step_index] for rank in expected_ranks)
                for step_index in range(expected_steps)
            ]
            request_profile["step_seconds_rank_max"] = step_seconds
            request_profile["step_latency_seconds"] = distribution(step_seconds)
            all_step_seconds.extend(step_seconds)
            request_average_seconds.append(
                request_profile["step_latency_seconds"]["avg_s"]
            )
        else:
            reasons.extend(f"{task_id}: {error}" for error in request_errors)
        request_profiles.append(request_profile)

    available = (
        not reasons
        and len(request_profiles) == MEASURED_REQUESTS
        and len(request_average_seconds) == MEASURED_REQUESTS
    )
    return {
        "status": "available" if available else "unavailable",
        "authoritative": available,
        "method": (
            "server.log Processing/Task-completed boundaries; measure task IDs "
            "only; per-rank infer_main occurrence order defines step index; "
            "each request/step uses max across ranks"
        ),
        "warmup_excluded": True,
        "source_log": str(server_log),
        "config_path": str(config_path) if config_path else None,
        "expected_requests": MEASURED_REQUESTS,
        "task_ids_observed": len(task_ids),
        "requests_profiled": len(request_average_seconds),
        "expected_steps_per_request": expected_steps,
        "world_size": world_size,
        "rank_samples_expected": (
            MEASURED_REQUESTS * expected_steps * world_size
        ),
        "rank_samples_observed": rank_samples_observed,
        "summary": (
            distribution(request_average_seconds) if available else None
        ),
        "all_step_summary": (
            distribution(all_step_seconds) if available else None
        ),
        "requests": request_profiles,
        "reasons": list(dict.fromkeys(reasons)),
    }


def collect_peak_memory(case: dict[str, Any]) -> dict[str, Any]:
    source_field = None
    raw: Any = None
    for field in (
        "peak_device_memory_mb",
        "peak_memory_mb",
        "peak_hbm_mb",
        "peak_mlu_memory_mb",
    ):
        if isinstance(case.get(field), dict):
            source_field = field
            raw = case[field]
            break
    by_device: dict[str, int | float] = {}
    if isinstance(raw, dict):
        for device, value in raw.items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value >= 0
            ):
                by_device[str(device)] = value
    return {
        "unit": "MiB",
        "max": max(by_device.values()) if by_device else None,
        "by_device": by_device,
        "source_field": source_field,
    }


def validate_case_contract(
    case: dict[str, Any], contract: CaseContract
) -> None:
    case_id = case.get("case_id")
    expected = {
        "task_kind": contract.task_kind,
        "world_size": contract.world_size,
        "parallel_strategy": contract.parallel_strategy,
    }
    for field, expected_value in expected.items():
        if case.get(field) != expected_value:
            raise ValueError(
                f"{case_id}: {field} {case.get(field)!r} != "
                f"{expected_value!r}"
            )


def validate_run_config(
    benchmark: dict[str, Any], *, case_id: str, platform: str
) -> None:
    run_config = benchmark.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError(f"{case_id}: benchmark run_config is missing")
    if run_config.get("phase") not in (None, "measure"):
        raise ValueError(f"{case_id}: benchmark phase is not measure")
    if run_config.get("platform") not in (None, platform):
        raise ValueError(
            f"{case_id}: benchmark platform "
            f"{run_config.get('platform')!r} != {platform!r}"
        )
    for field, expected in (
        ("request_count", MEASURED_REQUESTS),
        ("concurrency", CONCURRENCY),
    ):
        if run_config.get(field) not in (None, expected):
            raise ValueError(
                f"{case_id}: benchmark {field} "
                f"{run_config.get(field)!r} != {expected}"
            )


def select_device_samples(log_dir: Path, platform: str) -> Path:
    platform_name = {
        "ascend_npu": "npu_samples.csv",
        "mlu": "mlu_samples.csv",
        "metax": "metax_samples.csv",
    }.get(platform, "device_samples.csv")
    candidates = [
        log_dir / platform_name,
        log_dir / "device_samples.csv",
        log_dir / "npu_samples.csv",
        log_dir / "mlu_samples.csv",
        log_dir / "metax_samples.csv",
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def collect_case(
    case: dict[str, Any],
    *,
    source_suite: Path,
    platform: str,
) -> dict[str, Any]:
    case_id = case.get("case_id")
    contract = CASE_CONTRACTS[case_id]
    validate_case_contract(case, contract)
    benchmark = case.get("benchmark_result")
    if case.get("status") != "passed" or not isinstance(benchmark, dict):
        raise ValueError(f"{case_id}: selected case is not passed")
    if (
        benchmark.get("status") != "passed"
        or benchmark.get("requests_total") != MEASURED_REQUESTS
        or benchmark.get("requests_success") != MEASURED_REQUESTS
        or benchmark.get("requests_failed") != 0
    ):
        raise ValueError(
            f"{case_id}: benchmark is not a clean "
            f"{MEASURED_REQUESTS}/{MEASURED_REQUESTS} pass"
        )
    validate_run_config(benchmark, case_id=case_id, platform=platform)

    result_dir = case_result_dir(case, source_suite)
    log_dir = case_log_dir(case, source_suite, platform)
    warmup = read_json(result_dir / "warmup" / "result.json")
    if (
        warmup.get("status") != "passed"
        or warmup.get("requests_total") not in (None, WARMUP_REQUESTS)
        or warmup.get("requests_success") != WARMUP_REQUESTS
        or warmup.get("requests_failed") != 0
    ):
        raise ValueError(f"{case_id}: warm-up is not a clean pass")

    artifact_validation = verify_case_artifacts(case, contract)

    expected_steps, config_path, config_error = expected_steps_from_config(
        case_id, benchmark, platform, contract
    )
    raw_request_results = benchmark.get("request_results")
    request_results = (
        raw_request_results
        if isinstance(raw_request_results, list)
        else []
    )
    task_ids = [
        str(request.get("task_id", ""))
        for request in request_results
        if isinstance(request, dict)
    ]
    server_log = log_dir / "server.log"
    dit_step_latency = parse_dit_step_profile(
        server_log=server_log,
        task_ids=task_ids,
        expected_steps=expected_steps,
        world_size=contract.world_size,
        config_path=config_path,
        initial_error=config_error,
    )

    latency = benchmark["end_to_end_latency"]
    peak_device_memory = collect_peak_memory(case)
    device_samples = select_device_samples(log_dir, platform)
    source = {
        "suite_id": source_suite.name,
        "suite_summary": str(source_suite / "summary.json"),
        "suite_manifest": str(source_suite / "manifest.json"),
        "result": str(result_dir / "measure" / "result.json"),
        "requests": str(result_dir / "measure" / "requests.jsonl"),
        "outputs": str(result_dir / "measure" / "outputs"),
        "server_log": str(server_log),
        "client_log": str(log_dir / "client.log"),
        "warmup_log": str(log_dir / "warmup.log"),
        "device_samples": str(device_samples),
        "model_config": str(config_path) if config_path else None,
    }
    if platform == "ascend_npu":
        source["npu_samples"] = str(device_samples)
    else:
        source["mlu_samples"] = str(device_samples)
    if artifact_validation["status"] != "passed":
        case_status = "failed_validation"
    elif dit_step_latency["status"] != "available":
        case_status = "unavailable_dit"
    else:
        case_status = "passed"
    collected = {
        "case_id": case_id,
        "task_kind": case["task_kind"],
        "world_size": case["world_size"],
        "parallel_strategy": case["parallel_strategy"],
        "status": case_status,
        "requests": {
            "warmup": WARMUP_REQUESTS,
            "measured": benchmark["requests_total"],
            "success": benchmark["requests_success"],
            "failed": benchmark["requests_failed"],
            "concurrency": CONCURRENCY,
        },
        "startup_seconds": case["startup_seconds"],
        "warmup_seconds": warmup["wall_time_s"],
        "latency_seconds": latency,
        "throughput_per_minute": throughput_per_minute(
            benchmark["throughput"]
        ),
        "peak_device_memory_mb": peak_device_memory,
        "artifact_validation": artifact_validation,
        "dit_step_latency_seconds": dit_step_latency,
        "source": source,
    }
    if platform == "ascend_npu":
        # Kept for consumers of the original Ascend-only schema.
        collected["peak_hbm_mb"] = {
            "max": peak_device_memory["max"],
            "by_device": peak_device_memory["by_device"],
        }
    return collected


def validate_suite_contract(
    summary: dict[str, Any], source_suite: Path, platform: str
) -> None:
    suite_kind = summary.get("suite_kind")
    legacy_ascend = suite_kind is None and platform == "ascend_npu"
    if suite_kind != "formal" and not legacy_ascend:
        raise ValueError(
            f"{source_suite}: --source-suite must be a formal suite; "
            f"suite_kind is {suite_kind!r}"
        )
    recorded_platform = summary.get("platform")
    if recorded_platform not in (None, platform):
        raise ValueError(
            f"{source_suite}: platform {recorded_platform!r} != "
            f"{platform!r}"
        )
    parameters = summary.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{source_suite}: suite parameters are missing")
    expected = {
        "sample_count": MEASURED_REQUESTS,
        "warmup_count": WARMUP_REQUESTS,
        "concurrency": CONCURRENCY,
    }
    for field, expected_value in expected.items():
        if parameters.get(field) != expected_value:
            raise ValueError(
                f"{source_suite}: {field} {parameters.get(field)!r} != "
                f"{expected_value}"
            )


def format_number(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "N/A"
    return f"{value:.{digits}f}"


def dit_summary(case: dict[str, Any]) -> dict[str, Any] | None:
    profile = case.get("dit_step_latency_seconds")
    if not isinstance(profile, dict) or profile.get("status") != "available":
        return None
    summary = profile.get("summary")
    return summary if isinstance(summary, dict) else None


def markdown_report(report: dict[str, Any]) -> str:
    platform_label = PLATFORM_LABELS[report["platform"]]
    unavailable_dit = ", ".join(report["unavailable_dit_cases"]) or "无"
    missing_cases = ", ".join(report["missing_cases"]) or "无"
    lines = [
        f"# {platform_label} 服务化推理性能报告",
        "",
        f"- 生成时间：`{report['generated_at']}`",
        f"- 平台：`{report['platform']}`",
        "- 测试口径：1 次预热（不计入统计）+ 10 个正式样本，并发 1",
        f"- 总体结果：`{report['status']}`，"
        f"{report['cases_passed']}/{report['cases_expected']} 个配置通过，"
        f"{report['artifacts_verified']} 个正式产物通过 SHA256 复核，"
        f"{report['media_verified']} 个产物通过媒体结构校验",
        "- 延迟：客户端端到端时延，P50/P90 仅由 10 个成功正式样本计算",
        (
            "- 单步 DiT：仅解析正式 measure 请求；按请求、step、rank "
            "对齐后取 rank 最大值，再对每请求平均单步时间输出 "
            "P50/P90/Avg。日志边界或步数不完整时明确标为 N/A。"
        ),
        f"- DiT 日志可用：`{report['dit_profiles_available']}`/"
        f"`{report['dit_profiles_expected']}` 个预期配置",
        f"- DiT 不可用配置：`{unavailable_dit}`",
        f"- 缺失配置：`{missing_cases}`",
        "",
        "| 配置 | 任务 | 卡数 | 并行 | 状态 | E2E P50(s) | E2E P90(s) | "
        "E2E Avg(s) | 吞吐/分 | DiT P50(s) | DiT P90(s) | "
        "DiT Avg(s) | 启动(s) | 预热(s) | 峰值设备显存(MiB) | 媒体 |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        latency = case["latency_seconds"]
        dit = dit_summary(case) or {}
        memory = case["peak_device_memory_mb"]["max"]
        validation = case["artifact_validation"]
        lines.append(
            f"| {case['case_id']} | {case['task_kind']} | "
            f"{case['world_size']} | {case['parallel_strategy']} | "
            f"{case['status']} | "
            f"{latency['p50_s']:.4f} | {latency['p90_s']:.4f} | "
            f"{latency['avg_s']:.4f} | "
            f"{case['throughput_per_minute']:.6f} | "
            f"{format_number(dit.get('p50_s'))} | "
            f"{format_number(dit.get('p90_s'))} | "
            f"{format_number(dit.get('avg_s'))} | "
            f"{case['startup_seconds']:.2f} | "
            f"{case['warmup_seconds']:.2f} | "
            f"{format_number(memory, digits=0)} | "
            f"{validation['media_verified']}/"
            f"{validation['artifacts_expected']} |"
        )

    if report["remediations"]:
        lines.extend(["", "## 测试中修复", ""])
        for remediation in report["remediations"]:
            lines.append(
                f"- `{remediation['scope']}`：{remediation['problem']} "
                f"{remediation['resolution']}"
            )

    lines.extend(["", "## 结果追溯", ""])
    for case in report["cases"]:
        source = case["source"]
        profile = case["dit_step_latency_seconds"]
        lines.extend(
            [
                f"### {case['case_id']}",
                "",
                f"- 原始结果：`{source['result']}`",
                f"- 请求明细：`{source['requests']}`",
                f"- 正式产物：`{source['outputs']}`",
                f"- 服务日志：`{source['server_log']}`",
                f"- 客户端日志：`{source['client_log']}`",
                f"- 设备显存采样：`{source['device_samples']}`",
                f"- 媒体结构：`{case['artifact_validation']['media_verified']}`"
                f"/`{case['artifact_validation']['artifacts_expected']}` 通过",
                f"- 单步 DiT：`{profile['status']}`",
            ]
        )
        if profile["status"] == "unavailable":
            reasons = "; ".join(profile["reasons"]) or "日志证据不足"
            lines.append(f"- DiT 不可用原因：{reasons}")
        validation = case["artifact_validation"]
        if validation["status"] != "passed":
            failures = "; ".join(validation["failures"])
            lines.append(f"- 产物/媒体校验失败：{failures}")
        lines.extend(
            [
                f"- 来源批次：`{source['suite_id']}`",
                "",
            ]
        )

    lines.extend(["## 问题诊断批次", ""])
    if report["diagnostic_suites"]:
        for diagnostic in report["diagnostic_suites"]:
            lines.append(
                f"- `{diagnostic['suite_id']}`：`{diagnostic['status']}`，"
                f"`{diagnostic['summary_markdown']}`"
            )
        lines.extend(
            [
                "",
                "这些诊断批次不参与性能统计，但保留了对应日志、配置快照和"
                "环境信息。",
                "",
            ]
        )
    else:
        lines.extend(["- 无。", ""])

    comparison = report.get("platform_comparison")
    if isinstance(comparison, dict):
        lines.extend(
            [
                "## MLU vs Ascend",
                "",
                f"- 对比状态：`{comparison['status']}`",
                f"- 独立对比报告：`{comparison['markdown_path']}`",
                "",
            ]
        )
    return "\n".join(lines)


def infer_report_platform(report: dict[str, Any], path: Path) -> str | None:
    platform = report.get("platform")
    if platform in PLATFORMS:
        return str(platform)
    haystack = " ".join(
        [
            str(path),
            *[
                str(value)
                for value in report.get("source_suites", [])
                if isinstance(value, str)
            ],
        ]
    )
    if "ascend_npu" in haystack:
        return "ascend_npu"
    if re.search(r"(?:^|[/_])mlu(?:[/_]|$)", haystack):
        return "mlu"
    return None


def ratio(numerator: Any, denominator: Any) -> float | None:
    if (
        isinstance(numerator, bool)
        or isinstance(denominator, bool)
        or not isinstance(numerator, (int, float))
        or not isinstance(denominator, (int, float))
        or denominator == 0
    ):
        return None
    return float(numerator) / float(denominator)


def comparison_metric(
    *, mlu_value: Any, ascend_value: Any, lower_is_better: bool
) -> dict[str, Any]:
    result = {
        "mlu": mlu_value,
        "ascend": ascend_value,
        "mlu_vs_ascend_percent": None,
        "mlu_speedup": None,
    }
    value_ratio = ratio(mlu_value, ascend_value)
    if value_ratio is not None:
        result["mlu_vs_ascend_percent"] = (value_ratio - 1.0) * 100.0
    result["mlu_speedup"] = (
        ratio(ascend_value, mlu_value)
        if lower_is_better
        else ratio(mlu_value, ascend_value)
    )
    return result


def build_comparison(
    *,
    current_report: dict[str, Any],
    current_report_path: Path,
    other_report_path: Path,
    markdown_path: Path,
    json_path: Path,
) -> dict[str, Any]:
    other_report = read_json(other_report_path)
    current_platform = current_report["platform"]
    other_platform = infer_report_platform(other_report, other_report_path)
    if other_platform is None:
        raise ValueError(
            f"{other_report_path}: cannot determine report platform"
        )
    if {current_platform, other_platform} != {"ascend_npu", "mlu"}:
        raise ValueError(
            "comparison requires one mlu report and one ascend_npu report"
        )
    reports = {
        current_platform: current_report,
        other_platform: other_report,
    }
    report_paths = {
        current_platform: str(current_report_path),
        other_platform: str(other_report_path),
    }
    by_platform = {
        platform: {
            case["case_id"]: case
            for case in report.get("cases", [])
            if isinstance(case, dict) and isinstance(case.get("case_id"), str)
        }
        for platform, report in reports.items()
    }
    cases: list[dict[str, Any]] = []
    missing: list[str] = []
    for case_id in EXPECTED_CASES:
        mlu_case = by_platform["mlu"].get(case_id)
        ascend_case = by_platform["ascend_npu"].get(case_id)
        if mlu_case is None or ascend_case is None:
            missing.append(case_id)
            continue
        mlu_latency = mlu_case.get("latency_seconds") or {}
        ascend_latency = ascend_case.get("latency_seconds") or {}
        mlu_dit = dit_summary(mlu_case) or {}
        ascend_dit = dit_summary(ascend_case) or {}
        mlu_memory = (
            mlu_case.get("peak_device_memory_mb")
            or mlu_case.get("peak_hbm_mb")
            or {}
        ).get("max")
        ascend_memory = (
            ascend_case.get("peak_device_memory_mb")
            or ascend_case.get("peak_hbm_mb")
            or {}
        ).get("max")
        cases.append(
            {
                "case_id": case_id,
                "end_to_end_p50_seconds": comparison_metric(
                    mlu_value=mlu_latency.get("p50_s"),
                    ascend_value=ascend_latency.get("p50_s"),
                    lower_is_better=True,
                ),
                "end_to_end_p90_seconds": comparison_metric(
                    mlu_value=mlu_latency.get("p90_s"),
                    ascend_value=ascend_latency.get("p90_s"),
                    lower_is_better=True,
                ),
                "end_to_end_avg_seconds": comparison_metric(
                    mlu_value=mlu_latency.get("avg_s"),
                    ascend_value=ascend_latency.get("avg_s"),
                    lower_is_better=True,
                ),
                "throughput_per_minute": comparison_metric(
                    mlu_value=mlu_case.get("throughput_per_minute"),
                    ascend_value=ascend_case.get("throughput_per_minute"),
                    lower_is_better=False,
                ),
                "dit_request_avg_step_seconds": comparison_metric(
                    mlu_value=mlu_dit.get("avg_s"),
                    ascend_value=ascend_dit.get("avg_s"),
                    lower_is_better=True,
                ),
                "peak_device_memory_mb": comparison_metric(
                    mlu_value=mlu_memory,
                    ascend_value=ascend_memory,
                    lower_is_better=True,
                ),
            }
        )
    source_reports_passed = all(
        report.get("status") == "passed" for report in reports.values()
    )
    dit_cases_compared = sum(
        case["dit_request_avg_step_seconds"]["mlu"] is not None
        and case["dit_request_avg_step_seconds"]["ascend"] is not None
        for case in cases
    )
    comparison = {
        "schema_version": "1.0",
        "kind": "mlu_vs_ascend_service_benchmark",
        "generated_at": utc_now(),
        "status": (
            "passed"
            if (
                not missing
                and source_reports_passed
                and dit_cases_compared == len(EXPECTED_CASES)
            )
            else "incomplete"
        ),
        "source_reports_passed": source_reports_passed,
        "dit_cases_compared": dit_cases_compared,
        "cases_expected": len(EXPECTED_CASES),
        "cases_compared": len(cases),
        "missing_cases": missing,
        "reports": report_paths,
        "cases": cases,
        "markdown_path": str(markdown_path),
        "json_path": str(json_path),
    }
    return comparison


def comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# MLU vs Ascend 服务化推理对比报告",
        "",
        f"- 生成时间：`{comparison['generated_at']}`",
        f"- 状态：`{comparison['status']}`，"
        f"{comparison['cases_compared']}/{comparison['cases_expected']} "
        "个配置可对比",
        f"- DiT 双平台可比：`{comparison['dit_cases_compared']}`/"
        f"`{comparison['cases_compared']}` 个配置",
        "- `加速比 > 1` 表示 MLU 更快；DiT 为每请求平均单步时间。",
        "",
        "| 配置 | MLU E2E P50(s) | Ascend E2E P50(s) | MLU加速比 | "
        "MLU吞吐/分 | Ascend吞吐/分 | 吞吐比 | MLU DiT Avg(s) | "
        "Ascend DiT Avg(s) | DiT加速比 | MLU显存(MiB) | "
        "Ascend显存(MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in comparison["cases"]:
        e2e = case["end_to_end_p50_seconds"]
        throughput = case["throughput_per_minute"]
        dit = case["dit_request_avg_step_seconds"]
        memory = case["peak_device_memory_mb"]
        lines.append(
            f"| {case['case_id']} | "
            f"{format_number(e2e['mlu'])} | "
            f"{format_number(e2e['ascend'])} | "
            f"{format_number(e2e['mlu_speedup'])} | "
            f"{format_number(throughput['mlu'], 6)} | "
            f"{format_number(throughput['ascend'], 6)} | "
            f"{format_number(throughput['mlu_speedup'])} | "
            f"{format_number(dit['mlu'])} | "
            f"{format_number(dit['ascend'])} | "
            f"{format_number(dit['mlu_speedup'])} | "
            f"{format_number(memory['mlu'], 0)} | "
            f"{format_number(memory['ascend'], 0)} |"
        )
    if comparison["missing_cases"]:
        lines.extend(
            [
                "",
                "## 缺失配置",
                "",
                *[f"- `{case_id}`" for case_id in comparison["missing_cases"]],
            ]
        )
    lines.extend(
        [
            "",
            "## 来源报告",
            "",
            f"- MLU：`{comparison['reports']['mlu']}`",
            f"- Ascend：`{comparison['reports']['ascend_npu']}`",
            "",
        ]
    )
    return "\n".join(lines)


def comparison_report_path(value: Path) -> Path:
    path = rebase_repo_path(value)
    if path.is_dir():
        path = path / "final_report.json"
    return path.resolve(strict=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate completed service suites without starting a service."
        )
    )
    parser.add_argument(
        "--platform",
        choices=PLATFORMS,
        default="ascend_npu",
        help="Source platform (default: ascend_npu)",
    )
    parser.add_argument("--source-suite", action="append", required=True)
    parser.add_argument("--diagnostic-suite", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compare-report",
        "--compare-with-report",
        dest="compare_report",
        type=Path,
        help=(
            "Other platform final_report.json (or its directory); writes "
            "comparison_report.md/json"
        ),
    )
    parser.add_argument(
        "--comparison-output-dir",
        type=Path,
        help="Comparison output directory (default: --output-dir)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    platform = args.platform
    selected: dict[str, dict[str, Any]] = {}
    source_suites = [suite_path(value, platform) for value in args.source_suite]
    for source_suite in source_suites:
        summary = read_json(source_suite / "summary.json")
        validate_suite_contract(summary, source_suite, platform)
        for raw_case in summary.get("cases", []):
            if (
                isinstance(raw_case, dict)
                and raw_case.get("status") == "passed"
                and raw_case.get("case_id") in EXPECTED_CASES
            ):
                case_id = raw_case["case_id"]
                if case_id in selected:
                    raise ValueError(f"duplicate passed case: {case_id}")
                selected[case_id] = collect_case(
                    raw_case,
                    source_suite=source_suite,
                    platform=platform,
                )

    missing = [case_id for case_id in EXPECTED_CASES if case_id not in selected]
    diagnostics = []
    for value in args.diagnostic_suite:
        path = suite_path(value, platform)
        summary = read_json(path / "summary.json")
        suite_kind = summary.get("suite_kind")
        if suite_kind == "formal":
            raise ValueError(
                f"{path}: formal suite cannot be passed as "
                "--diagnostic-suite"
            )
        legacy_ascend = suite_kind is None and platform == "ascend_npu"
        if suite_kind != "diagnostic" and not legacy_ascend:
            raise ValueError(
                f"{path}: invalid diagnostic suite_kind "
                f"{suite_kind!r}"
            )
        recorded_platform = summary.get("platform")
        if recorded_platform not in (None, platform):
            raise ValueError(
                f"{path}: platform {recorded_platform!r} != "
                f"{platform!r}"
            )
        recorded_log_root = summary.get("log_root")
        diagnostic_log_root = (
            str(rebase_repo_path(recorded_log_root))
            if isinstance(recorded_log_root, str) and recorded_log_root
            else None
        )
        diagnostics.append(
            {
                "suite_id": path.name,
                "suite_kind": suite_kind or "legacy_unspecified",
                "status": summary.get("status"),
                "summary_json": str(path / "summary.json"),
                "summary_markdown": str(path / "summary.md"),
                "log_root": diagnostic_log_root,
            }
        )

    cases = [
        selected[case_id] for case_id in EXPECTED_CASES if case_id in selected
    ]
    artifacts_verified = sum(
        case["artifact_validation"]["artifacts_verified"] for case in cases
    )
    media_verified = sum(
        case["artifact_validation"]["media_verified"] for case in cases
    )
    expected_artifacts = len(EXPECTED_CASES) * MEASURED_REQUESTS
    cases_passed = sum(case["status"] == "passed" for case in cases)
    failed_validation_cases = [
        case["case_id"]
        for case in cases
        if case["status"] == "failed_validation"
    ]
    unavailable_dit_cases = [
        case_id
        for case_id in EXPECTED_CASES
        if case_id not in selected
        or selected[case_id]["dit_step_latency_seconds"]["status"]
        != "available"
    ]
    dit_profiles_available = sum(
        case["dit_step_latency_seconds"]["status"] == "available"
        for case in cases
    )
    complete = (
        not missing
        and not failed_validation_cases
        and not unavailable_dit_cases
        and cases_passed == len(EXPECTED_CASES)
        and artifacts_verified == expected_artifacts
        and media_verified == expected_artifacts
        and dit_profiles_available == len(EXPECTED_CASES)
    )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "platform": platform,
        "platform_label": PLATFORM_LABELS[platform],
        "repository_path": str(REPO_PATH),
        "status": "passed" if complete else "incomplete",
        "test_contract": {
            "warmup_samples": WARMUP_REQUESTS,
            "measured_samples": MEASURED_REQUESTS,
            "concurrency": CONCURRENCY,
            "percentiles": ["p50", "p90"],
            "p99_measured": False,
            "cases": list(EXPECTED_CASES),
            "artifacts_expected": expected_artifacts,
            "dit_profiles_expected": len(EXPECTED_CASES),
        },
        "cases_expected": len(EXPECTED_CASES),
        "cases_passed": cases_passed,
        "missing_cases": missing,
        "failed_validation_cases": failed_validation_cases,
        "unavailable_dit_cases": unavailable_dit_cases,
        "artifacts_expected": expected_artifacts,
        "artifacts_verified": artifacts_verified,
        "media_verified": media_verified,
        "dit_profiles_expected": len(EXPECTED_CASES),
        "dit_profiles_available": dit_profiles_available,
        "remediations": [
            *COMMON_REMEDIATIONS,
            *(ASCEND_REMEDIATIONS if platform == "ascend_npu" else ()),
        ],
        "source_suites": [str(path) for path in source_suites],
        "diagnostic_suites": diagnostics,
        "cases": cases,
    }

    output_dir = args.output_dir.resolve()
    final_json_path = output_dir / "final_report.json"
    final_markdown_path = output_dir / "final_report.md"
    comparison = None
    if args.compare_report is not None:
        other_path = comparison_report_path(args.compare_report)
        comparison_dir = (
            args.comparison_output_dir.resolve()
            if args.comparison_output_dir is not None
            else output_dir
        )
        comparison_json_path = comparison_dir / "comparison_report.json"
        comparison_markdown_path = comparison_dir / "comparison_report.md"
        comparison = build_comparison(
            current_report=report,
            current_report_path=final_json_path,
            other_report_path=other_path,
            markdown_path=comparison_markdown_path,
            json_path=comparison_json_path,
        )
        report["platform_comparison"] = comparison

    atomic_write(
        final_json_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write(final_markdown_path, markdown_report(report))
    if comparison is not None:
        comparison_json_path = Path(comparison["json_path"])
        comparison_markdown_path = Path(comparison["markdown_path"])
        atomic_write(
            comparison_json_path,
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        )
        atomic_write(
            comparison_markdown_path,
            comparison_markdown(comparison),
        )
        print(comparison_markdown_path)
    print(final_markdown_path)
    comparison_passed = (
        comparison is None or comparison["status"] == "passed"
    )
    return 0 if report["status"] == "passed" and comparison_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
