#!/usr/bin/env python3
"""Common, accelerator-free checks for one LightX2V inference entrypoint."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEVICE_RE = re.compile(r"^[0-9]+$")
STRATEGY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class PreflightInputs:
    repo_path: Path
    lightx2v_path: Path
    model_path: Path
    config_path: Path
    source_script: Path
    reference_script: Path
    world_size: int
    visible_devices: str
    parallel_strategy: str
    result_ext: str
    audio_path: Path | None = None


def _positive_parallel_size(parallel: dict[str, Any], key: str, errors: list[str]) -> int:
    value = parallel.get(key, 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        errors.append(f"config parallel.{key} must be a positive integer; got {value!r}")
        return 1
    return value


def effective_parallel_size(config: dict[str, Any], errors: list[str]) -> int:
    parallel = config.get("parallel", {})
    if parallel is None:
        parallel = {}
    if not isinstance(parallel, dict):
        errors.append(f"config parallel must be an object; got {type(parallel).__name__}")
        return 1

    tensor_size = _positive_parallel_size(parallel, "tensor_p_size", errors)
    cfg_size = _positive_parallel_size(parallel, "cfg_p_size", errors)
    sequence_size = _positive_parallel_size(parallel, "seq_p_size", errors)
    if tensor_size > 1 and (cfg_size > 1 or sequence_size > 1):
        errors.append(
            "tensor parallelism cannot be combined with CFG or sequence parallelism "
            f"in this benchmark; got tp={tensor_size}, cfg={cfg_size}, sp={sequence_size}"
        )
    return tensor_size if tensor_size > 1 else cfg_size * sequence_size


def validate(inputs: PreflightInputs) -> list[str]:
    errors: list[str] = []

    if not inputs.repo_path.is_dir():
        errors.append(f"Examples repository does not exist: {inputs.repo_path}")
    if not inputs.lightx2v_path.is_dir():
        errors.append(f"LightX2V repository does not exist: {inputs.lightx2v_path}")
    elif not (inputs.lightx2v_path / "scripts" / "base" / "base.sh").is_file():
        errors.append(
            "LightX2V base script does not exist: "
            f"{inputs.lightx2v_path / 'scripts' / 'base' / 'base.sh'}"
        )
    if not inputs.model_path.is_dir():
        errors.append(f"model directory does not exist: {inputs.model_path}")
    if not inputs.source_script.is_file():
        errors.append(f"entry script does not exist: {inputs.source_script}")
    if not inputs.reference_script.is_file():
        errors.append(f"LightX2V reference script does not exist: {inputs.reference_script}")

    config: dict[str, Any] | None = None
    if not inputs.config_path.is_file():
        errors.append(f"config file does not exist: {inputs.config_path}")
    else:
        try:
            loaded = json.loads(inputs.config_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                errors.append("config JSON root must be an object")
            else:
                config = loaded
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"config is not valid JSON: {inputs.config_path}: {exc}")

    if inputs.world_size < 1:
        errors.append(f"world_size must be a positive integer; got {inputs.world_size}")

    devices = inputs.visible_devices.split(",") if inputs.visible_devices else []
    if len(devices) != inputs.world_size:
        errors.append(
            f"world_size={inputs.world_size} requires exactly {inputs.world_size} visible "
            f"NPU device(s); got {inputs.visible_devices!r}"
        )
    elif any(not DEVICE_RE.fullmatch(device) for device in devices):
        errors.append(
            "ASCEND_RT_VISIBLE_DEVICES must be a comma-separated list of numeric IDs; "
            f"got {inputs.visible_devices!r}"
        )
    elif len(set(devices)) != len(devices):
        errors.append(f"ASCEND_RT_VISIBLE_DEVICES contains duplicate IDs: {inputs.visible_devices!r}")

    if not STRATEGY_RE.fullmatch(inputs.parallel_strategy):
        errors.append(f"invalid parallel_strategy: {inputs.parallel_strategy!r}")
    elif inputs.world_size == 1 and inputs.parallel_strategy != "single":
        errors.append(
            f"world_size=1 requires parallel_strategy='single'; got {inputs.parallel_strategy!r}"
        )
    elif inputs.world_size > 1 and inputs.parallel_strategy == "single":
        errors.append("multi-card inference requires a non-single parallel_strategy")

    if config is not None:
        checkpoint = config.get("dit_original_ckpt")
        if checkpoint is not None:
            if not isinstance(checkpoint, str) or not checkpoint:
                errors.append(
                    "config dit_original_ckpt must be a non-empty local file path"
                )
            elif not Path(checkpoint).is_file():
                errors.append(
                    f"config dit_original_ckpt does not exist: {checkpoint}"
                )
        configured_size = effective_parallel_size(config, errors)
        if configured_size != inputs.world_size:
            errors.append(
                f"config parallel size is {configured_size}, but world_size is {inputs.world_size}"
            )

    if inputs.audio_path is not None and not inputs.audio_path.is_file():
        errors.append(f"input audio does not exist: {inputs.audio_path}")

    if inputs.result_ext not in {"png", "mp4"}:
        errors.append(f"result_ext must be png or mp4; got {inputs.result_ext!r}")

    for command in ("python", "tee"):
        if shutil.which(command) is None:
            errors.append(f"required command is unavailable: {command}")
    if inputs.world_size > 1 and shutil.which("torchrun") is None:
        errors.append("required command is unavailable: torchrun")

    return errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--lightx2v-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--source-script", type=Path, required=True)
    parser.add_argument("--reference-script", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--visible-devices", required=True)
    parser.add_argument("--parallel-strategy", required=True)
    parser.add_argument("--result-ext", required=True)
    parser.add_argument("--audio-path", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    errors = validate(
        PreflightInputs(
            repo_path=arguments.repo_path,
            lightx2v_path=arguments.lightx2v_path,
            model_path=arguments.model_path,
            config_path=arguments.config_path,
            source_script=arguments.source_script,
            reference_script=arguments.reference_script,
            world_size=arguments.world_size,
            visible_devices=arguments.visible_devices,
            parallel_strategy=arguments.parallel_strategy,
            result_ext=arguments.result_ext,
            audio_path=arguments.audio_path,
        )
    )
    for error in errors:
        print(f"[ERROR] {error}")
    if errors:
        return 1
    print(
        "[Preflight] common checks passed: "
        f"world_size={arguments.world_size}, "
        f"parallel_strategy={arguments.parallel_strategy}, "
        f"devices={arguments.visible_devices}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
