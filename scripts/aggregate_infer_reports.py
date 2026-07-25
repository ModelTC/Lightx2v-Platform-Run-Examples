#!/usr/bin/env python3
"""Aggregate inference suite/run records without running inference.

The input side of this utility is deliberately read-only: it reads ``suite.json``
files and the ``run.json`` files referenced by them.  It does not inspect logs,
initialize an accelerator runtime, or operate on processes.  Its only writes are
the requested final Markdown and JSON reports.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_PATH = Path("/data/wushuo1/Lightx2v-Platform-Run-Examples")
DEFAULT_SUITES_ROOT = REPO_PATH / "logs" / "ascend_npu" / "infer" / "suites"
DEFAULT_JSON_OUTPUT = DEFAULT_SUITES_ROOT / "final_report.json"
DEFAULT_MARKDOWN_OUTPUT = DEFAULT_SUITES_ROOT / "final_report.md"
REPORT_SCHEMA_VERSION = "1.0"
ARTIFACT_VALIDATION_SCHEMA_VERSION = "1.0"
STRICT_REVALIDATION_SCHEMA_VERSION = "1.0"
ARTIFACT_VALIDATOR = "run_record.validate_artifact"

# This is the benchmark contract, not a discovery result.  Keeping the expected
# cases explicit ensures that an unstarted or accidentally omitted case remains
# visible in the final report.
CASE_ORDER: tuple[str, ...] = (
    "z_image_turbo_t2i_1664x928",
    "z_image_turbo_t2i_1664x928_sp2",
    "wan21_1_3b_self_forcing_t2v_480p_81f",
    "wan21_1_3b_t2v_480p_81f",
    "wan21_1_3b_t2v_480p_81f_cfg2_sp4",
    "longcat_image_t2i_1344x768",
    "longcat_image_t2i_1344x768_cfg2_sp4",
    "qwen_image_2512_t2i_1664x928",
    "qwen_image_2512_t2i_1664x928_cfg2_sp4",
    "flux2_dev_t2i_1344x768",
    "flux2_dev_t2i_1344x768_tp8",
    "wan22_moe_a14b_t2v_480p_81f",
    "wan22_moe_a14b_t2v_480p_81f_cfg2_sp4",
    "wan22_moe_a14b_t2v_480p_81f_tp8",
    "wan22_moe_a14b_t2v_720p_81f",
    "wan22_moe_a14b_t2v_720p_81f_cfg2_sp4",
    "wan22_moe_a14b_t2v_720p_81f_tp8",
    "hunyuan_video_15_t2v_480p_121f",
    "hunyuan_video_15_t2v_480p_121f_cfg2_sp4",
    "hunyuan_video_15_t2v_720p_121f",
    "hunyuan_video_15_t2v_720p_121f_cfg2_sp4",
    "ltx2_3_22b_dev_s2v_768x512_241f",
    "ltx2_3_22b_dev_s2v_768x512_241f_sp8",
)

TERMINAL_FAILURE_STATUSES = {
    "failed",
    "interrupted",
    "invalid",
    "killed",
    "timeout",
    "timed_out",
}
INCOMPLETE_STATUSES = {"pending", "queued", "running", "unknown"}
STRICT_VALIDATION_METHODS = {
    "png": {"Pillow.Image.verify", "PNG signature and IHDR"},
    "mp4": {"ffprobe", "PyAV full decode"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "top-level JSON value is not an object"
    return payload, None


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file_obj:
            file_obj.write(text)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, text)


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def string_value(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def number_value(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_value(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        return None
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        return None
    return value.casefold()


def has_strict_artifact_validation(
    artifact: dict[str, Any],
    *,
    expected_audio: bool = False,
) -> bool:
    validation = artifact.get("validation")
    if not isinstance(validation, dict):
        return False
    result_format = string_value(artifact.get("format"))
    method = string_value(validation.get("method"))
    validation_errors = validation.get("errors")
    if (
        result_format not in STRICT_VALIDATION_METHODS
        or method not in STRICT_VALIDATION_METHODS[result_format]
        or validation.get("checked") is not True
        or not isinstance(validation_errors, list)
        or validation_errors
    ):
        return False

    artifact_sha256 = sha256_value(artifact.get("sha256"))
    if artifact_sha256 is None:
        return False
    if (
        result_format == "mp4"
        and "strict_revalidation" not in validation
        and validation.get("expected_audio") is not expected_audio
    ):
        return False

    # A strict revalidation marker takes precedence whenever present. This
    # prevents a historical record revalidated with the current decoder from
    # being accepted solely because that decoder also emits the native schema
    # marker.
    if "strict_revalidation" in validation:
        strict = validation.get("strict_revalidation")
        if not isinstance(strict, dict):
            return False
        original_sha256 = sha256_value(strict.get("original_sha256"))
        current_sha256 = sha256_value(strict.get("current_sha256"))
        return (
            strict.get("schema_version")
            == STRICT_REVALIDATION_SCHEMA_VERSION
            and strict.get("checked") is True
            and strict.get("validator") == ARTIFACT_VALIDATOR
            and strict.get("sha256_match_required") is True
            and strict.get("sha256_matches_original") is True
            and strict.get("stable_during_validation") is True
            and strict.get("expected_audio") is expected_audio
            and original_sha256 is not None
            and current_sha256 is not None
            and original_sha256 == current_sha256 == artifact_sha256
        )

    # Records produced at run completion by the current validator have this
    # native marker. Older records have no marker and must pass through the
    # historical strict-revalidation path above before aggregation.
    return (
        validation.get("schema_version")
        == ARTIFACT_VALIDATION_SCHEMA_VERSION
        and validation.get("validator") == ARTIFACT_VALIDATOR
    )


def parse_epoch(value: Any) -> float | None:
    number = number_value(value)
    if number is not None:
        return float(number)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def resolve_json_path(path_value: str | None, suite_dir: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = suite_dir / path
    return path.resolve(strict=False)


def unique_strings(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def metric_summary(record: dict[str, Any]) -> dict[str, Any]:
    metrics = dict_value(record.get("metrics"))
    profile = dict_value(metrics.get("profile_seconds"))
    return {
        "wall_seconds": number_value(metrics.get("wall_seconds")),
        "profile_seconds": {
            "load_models": number_value(profile.get("load_models")),
            "text_encoder": number_value(profile.get("text_encoder")),
            "dit": number_value(profile.get("dit")),
            "vae_decoder": number_value(profile.get("vae_decoder")),
            "pipeline": number_value(profile.get("pipeline")),
            "total": number_value(profile.get("total")),
        },
        "profile_rank_aggregation": string_value(
            metrics.get("profile_rank_aggregation")
        ),
        "dit_seconds_per_step": number_value(
            metrics.get("dit_seconds_per_step")
        ),
        "generated_frames_per_second": number_value(
            metrics.get("generated_frames_per_second")
        ),
        "images_per_second": number_value(metrics.get("images_per_second")),
        "peak_device_memory_bytes": number_value(
            metrics.get("peak_device_memory_bytes")
        ),
    }


def classify_attempt(
    suite_case: dict[str, Any],
    record: dict[str, Any] | None,
    case_id_matches: bool,
) -> tuple[str, bool]:
    record_status = (
        string_value(record.get("status")) if record is not None else None
    )
    suite_status = string_value(suite_case.get("status"))
    artifact_valid = (
        dict_value(record.get("artifact")).get("valid") is True
        if record is not None
        else False
    )
    artifact = dict_value(record.get("artifact")) if record is not None else {}
    benchmark = dict_value(record.get("benchmark")) if record is not None else {}
    expected_audio = benchmark.get("task") in {"s2v", "ltx2_s2v"}
    artifact_strict = has_strict_artifact_validation(
        artifact, expected_audio=expected_audio
    )
    eligible_success = (
        record_status == "succeeded"
        and artifact_valid
        and artifact_strict
        and case_id_matches
        and suite_status == "passed"
    )
    if eligible_success:
        return "succeeded", True
    if record_status == "succeeded" and not artifact_valid:
        return "failed", False
    if record_status == "succeeded" and artifact_valid and not artifact_strict:
        return "failed", False
    if record_status in TERMINAL_FAILURE_STATUSES:
        return record_status, False
    if suite_status in TERMINAL_FAILURE_STATUSES:
        return suite_status, False
    if record_status in INCOMPLETE_STATUSES:
        return record_status, False
    if suite_status in INCOMPLETE_STATUSES:
        return suite_status, False
    if record is None:
        return "missing_run_record", False
    return "invalid", False


def make_attempt(
    suite_dir: Path,
    suite: dict[str, Any],
    suite_case: dict[str, Any],
    discovery_order: int,
    warnings: list[str],
) -> dict[str, Any] | None:
    case_id = string_value(suite_case.get("case_id"))
    if not case_id:
        warnings.append(
            f"{suite_dir / 'suite.json'}: ignored case entry without case_id"
        )
        return None

    suite_id = string_value(suite.get("suite_id")) or suite_dir.name
    run_record_path = resolve_json_path(
        string_value(suite_case.get("run_record")), suite_dir
    )
    record: dict[str, Any] | None = None
    if run_record_path is not None:
        record, error = read_json(run_record_path)
        if error:
            warnings.append(f"{run_record_path}: {error}")

    benchmark = dict_value(record.get("benchmark")) if record else {}
    record_case_id = string_value(benchmark.get("case_id"))
    case_id_matches = record is not None and record_case_id == case_id
    if record is not None and not case_id_matches:
        warnings.append(
            f"{run_record_path}: benchmark.case_id={record_case_id!r} "
            f"does not match suite case_id={case_id!r}"
        )

    run_paths = dict_value(record.get("paths")) if record else {}
    artifact = dict_value(record.get("artifact")) if record else {}
    run_log = string_value(run_paths.get("run_log"))
    if not run_log and run_record_path is not None:
        run_log = str(run_record_path.with_name("run.log"))
    run_record_output = string_value(run_paths.get("run_record"))
    if not run_record_output and run_record_path is not None:
        run_record_output = str(run_record_path)
    result = (
        string_value(run_paths.get("result"))
        or string_value(artifact.get("path"))
        or string_value(suite_case.get("result"))
    )

    attempt_status, eligible_success = classify_attempt(
        suite_case, record, case_id_matches
    )
    timing = dict_value(record.get("timing")) if record else {}
    finished_at = (
        string_value(timing.get("finished_at_utc"))
        or string_value(suite_case.get("finished_at_utc"))
    )
    started_at = (
        string_value(timing.get("started_at_utc"))
        or string_value(suite_case.get("started_at_utc"))
    )
    sort_epoch = next(
        (
            timestamp
            for timestamp in (
                parse_epoch(timing.get("finished_epoch_seconds")),
                parse_epoch(finished_at),
                parse_epoch(timing.get("started_epoch_seconds")),
                parse_epoch(started_at),
                parse_epoch(suite.get("started_at_utc")),
            )
            if timestamp is not None
        ),
        float("-inf"),
    )

    exit_record = dict_value(record.get("exit")) if record else {}
    errors = unique_strings(
        [
            *list_value(record.get("errors") if record else None),
            exit_record.get("error"),
            suite_case.get("error"),
        ]
    )
    if record is None and run_record_path is None:
        errors.append("suite case does not reference a run.json")
    elif record is None:
        errors.append("run.json is unavailable or invalid")
    elif not case_id_matches:
        errors.append("run.json case_id does not match suite case_id")
    elif record.get("status") == "succeeded" and artifact.get("valid") is not True:
        errors.append("run succeeded but artifact.valid is not true")
    elif (
        record.get("status") == "succeeded"
        and artifact.get("valid") is True
        and not has_strict_artifact_validation(
            artifact,
            expected_audio=dict_value(record.get("benchmark")).get("task")
            in {"s2v", "ltx2_s2v"},
        )
    ):
        errors.append(
            "artifact was not verified by an accepted strict decoder"
        )
    elif (
        record.get("status") == "succeeded"
        and artifact.get("valid") is True
        and suite_case.get("status") != "passed"
    ):
        errors.append(
            "run artifact succeeded but suite case status is "
            f"{suite_case.get('status')!r}; cleanup/controller checks did not pass"
        )
    errors = unique_strings(errors)

    run_id = (
        string_value(record.get("run_id") if record else None)
        or string_value(suite_case.get("run_id"))
    )
    return {
        "suite_id": suite_id,
        "suite_directory": str(suite_dir),
        "run_id": run_id,
        "status": attempt_status,
        "eligible_success": eligible_success,
        "suite_status": string_value(suite_case.get("status")),
        "run_record_status": (
            string_value(record.get("status")) if record is not None else None
        ),
        "artifact_valid": (
            artifact.get("valid") is True if record is not None else False
        ),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "duration_seconds": number_value(
            timing.get("duration_seconds")
            if record is not None
            else suite_case.get("duration_seconds")
        ),
        "exit_code": number_value(
            exit_record.get("child_exit_code")
            if record is not None
            else suite_case.get("exit_code")
        ),
        "paths": {
            "run_log": run_log,
            "run_record": run_record_output,
            "result": result,
        },
        "metrics": metric_summary(record) if record is not None else None,
        "errors": errors,
        "_sort_epoch": sort_epoch,
        "_discovery_order": discovery_order,
    }


def attempt_sort_key(attempt: dict[str, Any]) -> tuple[float, str, str, int]:
    return (
        float(attempt["_sort_epoch"]),
        attempt.get("suite_id") or "",
        attempt.get("run_id") or "",
        int(attempt["_discovery_order"]),
    )


def public_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in attempt.items()
        if key not in {"_sort_epoch", "_discovery_order"}
    }


def final_status(attempts: list[dict[str, Any]]) -> str:
    if any(attempt["eligible_success"] for attempt in attempts):
        return "passed"
    if not attempts:
        return "not_run"
    latest = max(attempts, key=attempt_sort_key)
    if latest["status"] in INCOMPLETE_STATUSES:
        return "running" if latest["status"] == "running" else "incomplete"
    return "failed"


def discover_suite_directories(
    explicit_paths: Sequence[str], suites_root: Path
) -> list[Path]:
    candidates: list[Path] = []
    if explicit_paths:
        for raw_path in explicit_paths:
            path = Path(raw_path).expanduser().resolve(strict=False)
            if path.name == "suite.json":
                path = path.parent
            candidates.append(path)
    elif suites_root.is_dir():
        candidates.extend(
            path.resolve(strict=False)
            for path in suites_root.glob("suite_*")
            if path.is_dir() and (path / "suite.json").is_file()
        )

    unique: dict[str, Path] = {}
    for path in candidates:
        unique[str(path)] = path
    return sorted(unique.values(), key=lambda path: str(path))


def aggregate(suite_directories: Sequence[Path]) -> dict[str, Any]:
    warnings: list[str] = []
    suites: list[dict[str, Any]] = []
    attempts_by_case: dict[str, list[dict[str, Any]]] = {
        case_id: [] for case_id in CASE_ORDER
    }
    extra_attempts: list[dict[str, Any]] = []
    discovery_order = 0

    for suite_dir in suite_directories:
        suite_path = suite_dir / "suite.json"
        suite, error = read_json(suite_path)
        if error or suite is None:
            warnings.append(f"{suite_path}: {error or 'invalid suite record'}")
            continue
        suite_id = string_value(suite.get("suite_id")) or suite_dir.name
        suites.append(
            {
                "suite_id": suite_id,
                "directory": str(suite_dir),
                "status": string_value(suite.get("status")),
                "started_at_utc": string_value(suite.get("started_at_utc")),
                "finished_at_utc": string_value(suite.get("finished_at_utc")),
            }
        )
        for suite_case_raw in list_value(suite.get("cases")):
            if not isinstance(suite_case_raw, dict):
                warnings.append(
                    f"{suite_path}: ignored non-object case entry"
                )
                continue
            discovery_order += 1
            attempt = make_attempt(
                suite_dir,
                suite,
                suite_case_raw,
                discovery_order,
                warnings,
            )
            if attempt is None:
                continue
            case_id = string_value(suite_case_raw.get("case_id"))
            if case_id in attempts_by_case:
                attempts_by_case[case_id].append(attempt)
            else:
                extra_attempts.append(
                    {"case_id": case_id, **public_attempt(attempt)}
                )

    case_reports: list[dict[str, Any]] = []
    for index, case_id in enumerate(CASE_ORDER, start=1):
        attempts = sorted(attempts_by_case[case_id], key=attempt_sort_key)
        successful = [
            attempt for attempt in attempts if attempt["eligible_success"]
        ]
        selected = max(successful, key=attempt_sort_key) if successful else None
        failures = [
            attempt
            for attempt in attempts
            if not attempt["eligible_success"]
            and attempt["status"] not in INCOMPLETE_STATUSES
        ]
        incomplete = [
            attempt
            for attempt in attempts
            if not attempt["eligible_success"]
            and attempt["status"] in INCOMPLETE_STATUSES
        ]
        case_reports.append(
            {
                "index": index,
                "case_id": case_id,
                "final_status": final_status(attempts),
                "attempt_count": len(attempts),
                "selected_success": (
                    public_attempt(selected) if selected is not None else None
                ),
                "failure_attempts": [
                    public_attempt(attempt) for attempt in failures
                ],
                "incomplete_attempts": [
                    public_attempt(attempt) for attempt in incomplete
                ],
                "attempts": [
                    public_attempt(attempt) for attempt in attempts
                ],
            }
        )

    status_counts = {
        status: sum(
            case["final_status"] == status for case in case_reports
        )
        for status in ("passed", "failed", "running", "incomplete", "not_run")
    }
    failure_attempt_count = sum(
        len(case["failure_attempts"]) for case in case_reports
    )
    incomplete_attempt_count = sum(
        len(case["incomplete_attempts"]) for case in case_reports
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": "ascend_npu_offline_inference_final",
        "generated_at_utc": utc_now(),
        "status": (
            "complete"
            if status_counts["passed"] == len(CASE_ORDER)
            else "incomplete"
        ),
        "source": {
            "suite_count": len(suites),
            "suites": suites,
            "warnings": warnings,
        },
        "summary": {
            "expected_cases": len(CASE_ORDER),
            **status_counts,
            "attempt_count": sum(
                case["attempt_count"] for case in case_reports
            ),
            "failure_attempt_count": failure_attempt_count,
            "incomplete_attempt_count": incomplete_attempt_count,
            "extra_case_attempt_count": len(extra_attempts),
        },
        "cases": case_reports,
        "extra_case_attempts": extra_attempts,
    }


def markdown_escape(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace(
        "\n", " "
    )


def markdown_code(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"`{str(value).replace('`', 'ˋ')}`"


def format_number(value: Any) -> str:
    number = number_value(value)
    if number is None:
        return ""
    return f"{number:.6f}".rstrip("0").rstrip(".")


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_value(report.get("summary"))
    lines = [
        "# Ascend NPU Offline Inference Final Report",
        "",
        f"- Generated (UTC): `{report['generated_at_utc']}`",
        f"- Report status: `{report['status']}`",
        f"- Source suites: `{dict_value(report['source']).get('suite_count', 0)}`",
        (
            "- Cases: "
            f"`{summary.get('passed', 0)}/{summary.get('expected_cases', 0)}` passed, "
            f"`{summary.get('failed', 0)}` failed, "
            f"`{summary.get('running', 0)}` running, "
            f"`{summary.get('incomplete', 0)}` incomplete, "
            f"`{summary.get('not_run', 0)}` not run"
        ),
        (
            "- Attempts: "
            f"`{summary.get('attempt_count', 0)}` total, "
            f"`{summary.get('failure_attempt_count', 0)}` failed"
        ),
        "",
        (
            "For each case, the selected result is the latest run whose "
            "`run.json` has `status: succeeded`, a versioned strict artifact "
            "validation, and whose suite case has `status: passed`. "
            "All metrics below come directly from that selected `run.json`."
        ),
        "",
        "## Final status and metrics",
        "",
        (
            "| # | Case | Status | Selected run | Wall (s) | Load (s) | "
            "DiT (s) | Pipeline (s) | Total (s) | DiT/step (s) | "
            "Frames/s | Images/s |"
        ),
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for case in list_value(report.get("cases")):
        selected = dict_value(case.get("selected_success"))
        metrics = dict_value(selected.get("metrics"))
        profile = dict_value(metrics.get("profile_seconds"))
        cells = [
            case.get("index"),
            markdown_code(case.get("case_id")),
            markdown_code(case.get("final_status")),
            markdown_code(selected.get("run_id")),
            format_number(metrics.get("wall_seconds")),
            format_number(profile.get("load_models")),
            format_number(profile.get("dit")),
            format_number(profile.get("pipeline")),
            format_number(profile.get("total")),
            format_number(metrics.get("dit_seconds_per_step")),
            format_number(metrics.get("generated_frames_per_second")),
            format_number(metrics.get("images_per_second")),
        ]
        lines.append("| " + " | ".join(markdown_escape(cell) for cell in cells) + " |")

    lines.extend(
        [
            "",
            "## Selected successful outputs",
            "",
            "| Case | run.log | run.json | Result |",
            "|---|---|---|---|",
        ]
    )
    selected_count = 0
    for case in list_value(report.get("cases")):
        selected = dict_value(case.get("selected_success"))
        if not selected:
            continue
        selected_count += 1
        paths = dict_value(selected.get("paths"))
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_code(case.get("case_id")),
                    markdown_code(paths.get("run_log")),
                    markdown_code(paths.get("run_record")),
                    markdown_code(paths.get("result")),
                ]
            )
            + " |"
        )
    if selected_count == 0:
        lines.append("|  |  |  |  |")

    lines.extend(
        [
            "",
            "## Failed attempts",
            "",
            "| Case | Suite | Run | Status | Finished (UTC) | Error | run.json | run.log |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    failed_count = 0
    for case in list_value(report.get("cases")):
        for attempt in list_value(case.get("failure_attempts")):
            failed_count += 1
            paths = dict_value(attempt.get("paths"))
            error_text = "; ".join(
                value
                for value in list_value(attempt.get("errors"))
                if isinstance(value, str)
            )
            cells = [
                markdown_code(case.get("case_id")),
                markdown_code(attempt.get("suite_id")),
                markdown_code(attempt.get("run_id")),
                markdown_code(attempt.get("status")),
                markdown_code(attempt.get("finished_at_utc")),
                error_text,
                markdown_code(paths.get("run_record")),
                markdown_code(paths.get("run_log")),
            ]
            lines.append(
                "| "
                + " | ".join(markdown_escape(cell) for cell in cells)
                + " |"
            )
    if failed_count == 0:
        lines.append("|  |  |  |  |  | None |  |  |")

    incomplete_count = int(summary.get("incomplete_attempt_count", 0) or 0)
    if incomplete_count:
        lines.extend(
            [
                "",
                "## Incomplete attempts",
                "",
                "| Case | Suite | Run | Status | Started (UTC) | run.json |",
                "|---|---|---|---|---|---|",
            ]
        )
        for case in list_value(report.get("cases")):
            for attempt in list_value(case.get("incomplete_attempts")):
                paths = dict_value(attempt.get("paths"))
                cells = [
                    markdown_code(case.get("case_id")),
                    markdown_code(attempt.get("suite_id")),
                    markdown_code(attempt.get("run_id")),
                    markdown_code(attempt.get("status")),
                    markdown_code(attempt.get("started_at_utc")),
                    markdown_code(paths.get("run_record")),
                ]
                lines.append(
                    "| "
                    + " | ".join(markdown_escape(cell) for cell in cells)
                    + " |"
                )

    source = dict_value(report.get("source"))
    warnings = list_value(source.get("warnings"))
    extras = list_value(report.get("extra_case_attempts"))
    if warnings or extras:
        lines.extend(["", "## Source notes", ""])
        for warning in warnings:
            lines.append(f"- {markdown_escape(warning)}")
        if extras:
            lines.append(
                f"- Ignored `{len(extras)}` attempt(s) for case IDs outside "
                "the 23-case benchmark contract."
            )

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate suite.json/run.json records into final_report.json and "
            "final_report.md without running inference."
        )
    )
    parser.add_argument(
        "suite_directories",
        nargs="*",
        help=(
            "Suite directories (or suite.json paths). If omitted, scan "
            "--suites-root for suite_* directories."
        ),
    )
    parser.add_argument(
        "--suites-root",
        type=Path,
        default=DEFAULT_SUITES_ROOT,
        help=f"Auto-discovery root (default: {DEFAULT_SUITES_ROOT})",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help=f"JSON output path (default: {DEFAULT_JSON_OUTPUT})",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
        help=f"Markdown output path (default: {DEFAULT_MARKDOWN_OUTPUT})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suite_directories = discover_suite_directories(
        args.suite_directories, args.suites_root.expanduser()
    )
    report = aggregate(suite_directories)
    atomic_write_json(args.json_output.expanduser(), report)
    atomic_write_text(
        args.markdown_output.expanduser(), render_markdown(report)
    )
    summary = report["summary"]
    print(
        f"Wrote {args.json_output} and {args.markdown_output}: "
        f"{summary['passed']}/{summary['expected_cases']} passed from "
        f"{report['source']['suite_count']} suite(s), "
        f"{summary['failure_attempt_count']} failed attempt(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
