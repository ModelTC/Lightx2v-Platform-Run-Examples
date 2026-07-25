#!/usr/bin/env python3
"""Strictly revalidate artifacts referenced by completed inference cases.

This utility is intentionally CPU/file-only.  It imports the artifact
validators from ``run_record.py`` but never initializes torch, an NPU runtime,
or a process controller.

Only this state combination is eligible for mutation:

* ``suite.json`` case status is ``passed``; and
* its referenced ``run.json`` status is ``succeeded``.

Every other case/run status is reported as skipped and left untouched.  A
suite whose own status is ``running`` is also skipped to avoid racing the live
suite controller while it updates ``suite.json``.
"""

from __future__ import annotations

import argparse
import copy
import hmac
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import run_record

REPO_PATH = Path("/data/wushuo1/Lightx2v-Platform-Run-Examples")
DEFAULT_SUITES_ROOT = (
    REPO_PATH / "logs" / "ascend_npu" / "infer" / "suites"
)
DEFAULT_SUMMARY_NAME = "artifact_revalidation.json"
SUMMARY_SCHEMA_VERSION = "1.0"
STRICT_VALIDATION_SCHEMA_VERSION = "1.0"


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


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def string_value(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def resolve_path(path_value: str | None, base_directory: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve(strict=False)


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


def result_path_for(
    record: dict[str, Any], run_record_path: Path
) -> Path | None:
    artifact = dict_value(record.get("artifact"))
    paths = dict_value(record.get("paths"))
    path_value = string_value(artifact.get("path")) or string_value(
        paths.get("result")
    )
    return resolve_path(path_value, run_record_path.parent)


def result_format_for(
    record: dict[str, Any], artifact_path: Path | None
) -> str:
    artifact = dict_value(record.get("artifact"))
    result_format = string_value(artifact.get("format"))
    if result_format:
        return result_format.lower().lstrip(".")
    if artifact_path is not None:
        return artifact_path.suffix.lower().lstrip(".")
    return ""


def append_unique_error(record: dict[str, Any], error: str) -> None:
    errors = record.get("errors")
    if not isinstance(errors, list):
        errors = []
    if error not in errors:
        errors.append(error)
    record["errors"] = errors


def failed_artifact_from_exception(
    old_artifact: dict[str, Any],
    artifact_path: Path | None,
    result_format: str,
    error: str,
) -> dict[str, Any]:
    artifact = copy.deepcopy(old_artifact)
    artifact.update(
        {
            "path": (
                str(artifact_path)
                if artifact_path is not None
                else string_value(old_artifact.get("path")) or ""
            ),
            "format": result_format,
            "valid": False,
        }
    )
    validation = dict_value(artifact.get("validation"))
    validation.update(
        {
            "checked": True,
            "method": validation.get("method"),
            "errors": [error],
        }
    )
    artifact["validation"] = validation
    return artifact


def strict_validate_artifact(
    record: dict[str, Any],
    run_record_path: Path,
    checked_at_utc: str,
) -> tuple[dict[str, Any], str | None]:
    old_artifact = dict_value(record.get("artifact"))
    original_sha256 = string_value(old_artifact.get("sha256"))
    artifact_path = result_path_for(record, run_record_path)
    result_format = result_format_for(record, artifact_path)
    benchmark = dict_value(record.get("benchmark"))
    target = dict_value(benchmark.get("target"))
    expected_audio = benchmark.get("task") in {"s2v", "ltx2_s2v"}

    if artifact_path is None:
        validation_error = "run record does not identify a result artifact"
        artifact = failed_artifact_from_exception(
            old_artifact,
            artifact_path,
            result_format,
            validation_error,
        )
    else:
        try:
            artifact = run_record.validate_artifact(
                artifact_path,
                result_format,
                target,
                expected_audio=expected_audio,
            )
        except Exception as exc:  # Keep the archival record actionable.
            validation_error = (
                "artifact validator raised "
                f"{type(exc).__name__}: {exc}"
            )
            artifact = failed_artifact_from_exception(
                old_artifact,
                artifact_path,
                result_format,
                validation_error,
            )

    validation = dict_value(artifact.get("validation"))
    validation_errors = validation.get("errors")
    if not isinstance(validation_errors, list):
        validation_errors = []
    else:
        validation_errors = [
            str(error) for error in validation_errors if str(error)
        ]

    validator_sha256 = string_value(artifact.get("sha256"))
    post_validation_sha256 = (
        run_record.sha256_file(artifact_path)
        if artifact_path is not None
        else None
    )
    stable_during_validation = (
        validator_sha256 is not None
        and post_validation_sha256 is not None
        and hmac.compare_digest(
            validator_sha256, post_validation_sha256
        )
    )
    current_sha256 = post_validation_sha256 or validator_sha256

    if not original_sha256:
        validation_errors.append(
            "original run record has no artifact SHA-256 to compare"
        )
        sha256_matches_original = False
    elif not current_sha256:
        validation_errors.append(
            "strict revalidation could not calculate the current artifact "
            "SHA-256"
        )
        sha256_matches_original = False
    else:
        sha256_matches_original = hmac.compare_digest(
            original_sha256, current_sha256
        )
        if not sha256_matches_original:
            validation_errors.append(
                "artifact SHA-256 changed since run completion: "
                f"expected {original_sha256}, got {current_sha256}"
            )

    if (
        validator_sha256 is not None
        and post_validation_sha256 is not None
        and not stable_during_validation
    ):
        validation_errors.append(
            "artifact changed while strict revalidation was in progress: "
            f"validator read {validator_sha256}, final read "
            f"{post_validation_sha256}"
        )

    if artifact.get("valid") is not True and not validation_errors:
        validation_errors.append(
            "validator returned artifact.valid=false without details"
        )
    validation["errors"] = list(dict.fromkeys(validation_errors))
    validation["strict_revalidation"] = {
        "schema_version": STRICT_VALIDATION_SCHEMA_VERSION,
        "checked": True,
        "checked_at_utc": checked_at_utc,
        "validator": "run_record.validate_artifact",
        "expected_audio": expected_audio,
        "sha256_match_required": True,
        "original_sha256": original_sha256,
        "current_sha256": current_sha256,
        "sha256_matches_original": sha256_matches_original,
        "stable_during_validation": stable_during_validation,
    }
    artifact["validation"] = validation
    artifact["sha256"] = current_sha256
    artifact["valid"] = (
        artifact.get("valid") is True
        and not validation["errors"]
        and sha256_matches_original
        and stable_during_validation
    )

    if artifact["valid"]:
        return artifact, None

    details = "; ".join(validation["errors"])
    if not details:
        details = "validator returned artifact.valid=false without details"
    return artifact, f"strict artifact revalidation failed: {details}"


def skipped_case_result(
    suite_case: dict[str, Any],
    reason: str,
    run_record_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "case_id": string_value(suite_case.get("case_id")),
        "run_id": string_value(suite_case.get("run_id")),
        "status": "skipped",
        "reason": reason,
        "run_record": (
            str(run_record_path)
            if run_record_path is not None
            else string_value(suite_case.get("run_record"))
        ),
        "artifact": None,
        "original_sha256": None,
        "current_sha256": None,
        "error": "",
        "mutated": False,
    }


def fail_suite_case(suite_case: dict[str, Any], error: str) -> None:
    suite_case["status"] = "failed"
    suite_case["run_record_status"] = "failed"
    suite_case["error"] = error


def revalidate_suite(suite_directory: Path) -> dict[str, Any]:
    suite_directory = suite_directory.expanduser().resolve(strict=False)
    suite_path = suite_directory / "suite.json"
    suite, suite_error = read_json(suite_path)
    result: dict[str, Any] = {
        "suite_id": suite_directory.name,
        "suite_directory": str(suite_directory),
        "suite_record": str(suite_path),
        "suite_status": None,
        "status": "completed",
        "suite_updated": False,
        "error": "",
        "cases": [],
    }
    if suite is None:
        result["status"] = "error"
        result["error"] = (
            f"cannot read suite record: {suite_error or 'invalid JSON'}"
        )
        return result

    result["suite_id"] = string_value(suite.get("suite_id")) or suite_directory.name
    suite_status = string_value(suite.get("status"))
    result["suite_status"] = suite_status
    if suite_status == "running":
        result["status"] = "skipped"
        result["error"] = (
            "suite is still running; skipped to avoid racing the suite "
            "controller"
        )
        return result

    cases = suite.get("cases")
    if not isinstance(cases, list):
        result["status"] = "error"
        result["error"] = "suite record field 'cases' is not a list"
        return result

    suite_changed = False
    for suite_case_raw in cases:
        if not isinstance(suite_case_raw, dict):
            result["cases"].append(
                {
                    "case_id": None,
                    "run_id": None,
                    "status": "skipped",
                    "reason": "suite case entry is not an object",
                    "run_record": None,
                    "artifact": None,
                    "original_sha256": None,
                    "current_sha256": None,
                    "error": "",
                    "mutated": False,
                }
            )
            continue
        suite_case = suite_case_raw
        case_status = string_value(suite_case.get("status"))
        if case_status != "passed":
            result["cases"].append(
                skipped_case_result(
                    suite_case,
                    f"suite case status is {case_status!r}, not 'passed'",
                )
            )
            continue

        run_record_path = resolve_path(
            string_value(suite_case.get("run_record")), suite_directory
        )
        if run_record_path is None:
            failure = (
                "strict artifact revalidation failed: suite case has no "
                "run_record reference"
            )
            fail_suite_case(suite_case, failure)
            suite_changed = True
            result["cases"].append(
                {
                    **skipped_case_result(
                        suite_case,
                        "suite case has no run_record reference",
                    ),
                    "status": "error",
                    "error": failure,
                }
            )
            continue

        record, record_error = read_json(run_record_path)
        if record is None:
            failure = (
                "strict artifact revalidation failed: referenced run record "
                f"cannot be read: {record_error or 'invalid run record'}"
            )
            fail_suite_case(suite_case, failure)
            suite_changed = True
            result["cases"].append(
                {
                    **skipped_case_result(
                        suite_case,
                        "referenced run record cannot be read",
                        run_record_path,
                    ),
                    "status": "error",
                    "error": failure,
                }
            )
            continue

        run_status = string_value(record.get("status"))
        if run_status != "succeeded":
            result["cases"].append(
                skipped_case_result(
                    suite_case,
                    f"run record status is {run_status!r}, not 'succeeded'",
                    run_record_path,
                )
            )
            continue

        original_sha256 = string_value(
            dict_value(record.get("artifact")).get("sha256")
        )
        checked_at_utc = utc_now()
        artifact, failure = strict_validate_artifact(
            record, run_record_path, checked_at_utc
        )
        record["artifact"] = artifact
        if failure is not None:
            record["status"] = "failed"
            append_unique_error(record, failure)

        try:
            run_record.atomic_write_json(run_record_path, record)
        except OSError as exc:
            result["cases"].append(
                {
                    "case_id": string_value(suite_case.get("case_id")),
                    "run_id": string_value(suite_case.get("run_id")),
                    "status": "error",
                    "reason": "",
                    "run_record": str(run_record_path),
                    "artifact": string_value(artifact.get("path")),
                    "original_sha256": original_sha256,
                    "current_sha256": string_value(artifact.get("sha256")),
                    "error": f"cannot atomically update run record: {exc}",
                    "mutated": False,
                }
            )
            continue

        case_result = {
            "case_id": string_value(suite_case.get("case_id")),
            "run_id": string_value(suite_case.get("run_id")),
            "status": "valid" if failure is None else "invalid",
            "reason": "",
            "run_record": str(run_record_path),
            "artifact": string_value(artifact.get("path")),
            "original_sha256": original_sha256,
            "current_sha256": string_value(artifact.get("sha256")),
            "error": failure or "",
            "mutated": True,
        }
        result["cases"].append(case_result)

        if failure is not None:
            fail_suite_case(suite_case, failure)
            suite_changed = True

    if suite_changed:
        try:
            run_record.atomic_write_json(suite_path, suite)
        except OSError as exc:
            result["status"] = "error"
            result["error"] = (
                "run record(s) were updated, but suite record could not be "
                f"atomically updated: {exc}"
            )
        else:
            result["suite_updated"] = True

    if any(case.get("status") == "invalid" for case in result["cases"]):
        result["status"] = (
            "error" if result["status"] == "error" else "invalid"
        )
    elif any(case.get("status") == "error" for case in result["cases"]):
        result["status"] = "error"
    return result


def build_summary(
    suite_directories: Sequence[Path],
) -> dict[str, Any]:
    started_at_utc = utc_now()
    suite_results = [
        revalidate_suite(suite_directory)
        for suite_directory in suite_directories
    ]
    case_results = [
        case
        for suite_result in suite_results
        for case in list_value(suite_result.get("cases"))
        if isinstance(case, dict)
    ]
    counts = {
        status: sum(
            case.get("status") == status for case in case_results
        )
        for status in ("valid", "invalid", "skipped", "error")
    }
    suite_counts = {
        status: sum(
            suite_result.get("status") == status
            for suite_result in suite_results
        )
        for status in ("completed", "invalid", "skipped", "error")
    }
    has_failure = (
        counts["invalid"] > 0
        or counts["error"] > 0
        or suite_counts["invalid"] > 0
        or suite_counts["error"] > 0
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "report_kind": "ascend_npu_offline_inference_artifact_revalidation",
        "status": "failed" if has_failure else "completed",
        "started_at_utc": started_at_utc,
        "finished_at_utc": utc_now(),
        "summary": {
            "suites_discovered": len(suite_directories),
            "suites": suite_counts,
            "cases_seen": len(case_results),
            "cases": counts,
        },
        "suites": suite_results,
    }


def default_summary_path(
    suite_directories: Sequence[Path],
    explicit_paths: Sequence[str],
    suites_root: Path,
) -> Path:
    if len(suite_directories) == 1 and explicit_paths:
        return suite_directories[0] / DEFAULT_SUMMARY_NAME
    return suites_root / DEFAULT_SUMMARY_NAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly revalidate artifacts for passed/succeeded inference "
            "records without using an NPU or operating on processes."
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
        "--output",
        type=Path,
        help=(
            "Machine-readable JSON summary path. By default, write beside "
            "one explicitly selected suite or under --suites-root."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    suites_root = arguments.suites_root.expanduser().resolve(strict=False)
    suite_directories = discover_suite_directories(
        arguments.suite_directories, suites_root
    )
    summary = build_summary(suite_directories)
    if not suite_directories:
        summary["status"] = "failed"
        summary["error"] = "no suite directories were discovered"

    output_path = (
        arguments.output.expanduser().resolve(strict=False)
        if arguments.output is not None
        else default_summary_path(
            suite_directories,
            arguments.suite_directories,
            suites_root,
        )
    )
    run_record.atomic_write_json(output_path, summary)
    counts = dict_value(summary.get("summary"))
    case_counts = dict_value(counts.get("cases"))
    print(
        "[ArtifactRevalidation] "
        f"status={summary['status']} "
        f"valid={case_counts.get('valid', 0)} "
        f"invalid={case_counts.get('invalid', 0)} "
        f"skipped={case_counts.get('skipped', 0)} "
        f"errors={case_counts.get('error', 0)} "
        f"summary={output_path}"
    )
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
