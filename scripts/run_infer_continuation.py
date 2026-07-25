#!/usr/bin/env python3
"""Gate one inference suite behind a completed, strictly revalidated suite.

This process is intended to be launched under ``nohup``/``setsid``.  It does
not query or operate on an NPU while waiting.  After the exact prerequisite
controller process exits, it:

1. requires the prerequisite ``suite.json`` and every case to be passed;
2. explicitly runs ``revalidate_infer_artifacts.py`` for that suite;
3. verifies the strict summary and strict markers in every ``run.json``;
4. reserves a unique next suite ID; and
5. runs ``run_infer_suite.py`` with the explicitly selected cases.

The continuation has its own durable JSON state and append-only log.  A
continuation ID and next suite ID are both single-use.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_PATH = Path("/data/wushuo1/Lightx2v-Platform-Run-Examples")
LOG_ROOT = REPO_PATH / "logs" / "ascend_npu" / "infer"
SUITES_ROOT = LOG_ROOT / "suites"
CONTINUATIONS_ROOT = LOG_ROOT / "continuations"
RESERVATIONS_ROOT = CONTINUATIONS_ROOT / "suite_id_reservations"
REVALIDATE_SCRIPT = REPO_PATH / "scripts" / "revalidate_infer_artifacts.py"
RUN_SUITE_SCRIPT = REPO_PATH / "scripts" / "run_infer_suite.py"

STATE_SCHEMA_VERSION = "1.0"
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ContinuationError(RuntimeError):
    """A checked condition prevented the continuation from launching."""


class ContinuationInterrupted(ContinuationError):
    """The continuation received SIGINT or SIGTERM."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    os.replace(temporary, path)


def read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuationError(f"cannot read {description} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContinuationError(f"{description} at {path} is not a JSON object")
    return payload


def read_process_start_time_ticks(pid: int, proc_root: Path = Path("/proc")) -> int | None:
    """Return Linux /proc stat field 22, or None after process exit."""
    try:
        stat_text = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ContinuationError(f"cannot read process identity for PID {pid}: {exc}") from exc

    close_parenthesis = stat_text.rfind(")")
    if close_parenthesis < 0:
        raise ContinuationError(f"cannot parse process identity for PID {pid}: missing comm")
    fields_after_command = stat_text[close_parenthesis + 2 :].split()
    try:
        # fields_after_command[0] is stat field 3 (state), so index 19 is
        # field 22 (starttime).
        return int(fields_after_command[19])
    except (IndexError, ValueError) as exc:
        raise ContinuationError(f"cannot parse process start time for PID {pid}") from exc


def wait_for_exact_process_exit(
    pid: int,
    expected_start_time_ticks: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
    *,
    proc_root: Path = Path("/proc"),
    sleep_fn: Any = time.sleep,
    monotonic_fn: Any = time.monotonic,
    interrupted_fn: Any = lambda: False,
) -> str:
    """Wait only for the process with the supplied PID/starttime identity."""
    observed_start_time = read_process_start_time_ticks(pid, proc_root)
    if observed_start_time is None:
        return "already_absent"
    if observed_start_time != expected_start_time_ticks:
        raise ContinuationError(
            "prerequisite PID identity mismatch before waiting: "
            f"PID {pid} has starttime {observed_start_time}, expected "
            f"{expected_start_time_ticks}; refusing to treat a reused PID "
            "as the prerequisite controller"
        )

    started = monotonic_fn()
    while True:
        if interrupted_fn():
            raise ContinuationInterrupted("continuation was interrupted while waiting for prerequisite")
        if timeout_seconds > 0 and monotonic_fn() - started >= timeout_seconds:
            raise ContinuationError(f"timed out waiting for prerequisite PID {pid} after {timeout_seconds:g} seconds")
        sleep_fn(poll_interval_seconds)
        observed_start_time = read_process_start_time_ticks(pid, proc_root)
        if observed_start_time is None:
            return "exited"
        if observed_start_time != expected_start_time_ticks:
            # We observed the exact identity first.  A different starttime
            # means the original exited and Linux reused its PID.
            return "exited_then_pid_reused"


def validate_identifier(value: str, option: str) -> str:
    if not SAFE_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(f"{option} must match {SAFE_ID_PATTERN.pattern!r}")
    return value


def validate_passed_suite(
    suite: dict[str, Any],
    expected_suite_id: str,
    expected_case_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    suite_id = suite.get("suite_id")
    if suite_id != expected_suite_id:
        raise ContinuationError(f"prerequisite suite_id is {suite_id!r}, expected {expected_suite_id!r}")
    if suite.get("status") != "passed":
        raise ContinuationError(f"prerequisite suite status is {suite.get('status')!r}, not 'passed'")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ContinuationError("prerequisite passed suite has no non-empty cases list")

    checked_cases: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for index, raw_case in enumerate(cases, start=1):
        if not isinstance(raw_case, dict):
            raise ContinuationError(f"prerequisite suite case {index} is not an object")
        case_id = raw_case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ContinuationError(f"prerequisite suite case {index} has no case_id")
        if case_id in seen_case_ids:
            raise ContinuationError(f"prerequisite suite contains duplicate case_id {case_id!r}")
        seen_case_ids.add(case_id)
        failures: list[str] = []
        if raw_case.get("status") != "passed":
            failures.append(f"status={raw_case.get('status')!r}")
        if raw_case.get("exit_code") != 0:
            failures.append(f"exit_code={raw_case.get('exit_code')!r}")
        if raw_case.get("run_record_status") != "succeeded":
            failures.append(f"run_record_status={raw_case.get('run_record_status')!r}")
        if failures:
            raise ContinuationError(f"prerequisite case {case_id!r} is not a clean pass: " + ", ".join(failures))
        checked_cases.append(raw_case)

    if expected_case_ids is not None and seen_case_ids != expected_case_ids:
        raise ContinuationError(f"next suite completed with unexpected cases: actual={sorted(seen_case_ids)}, expected={sorted(expected_case_ids)}")
    return checked_cases


def verify_strict_run_records(cases: Sequence[dict[str, Any]], suite_directory: Path) -> None:
    for suite_case in cases:
        case_id = str(suite_case["case_id"])
        path_value = suite_case.get("run_record")
        if not isinstance(path_value, str) or not path_value:
            raise ContinuationError(f"strictly revalidated case {case_id!r} has no run_record")
        run_record_path = Path(path_value).expanduser()
        if not run_record_path.is_absolute():
            run_record_path = suite_directory / run_record_path
        run_record_path = run_record_path.resolve(strict=False)
        record = read_json_object(run_record_path, f"run record for case {case_id!r}")
        artifact = record.get("artifact")
        if not isinstance(artifact, dict):
            raise ContinuationError(f"run record for case {case_id!r} has no artifact object")
        validation = artifact.get("validation")
        strict = validation.get("strict_revalidation") if isinstance(validation, dict) else None
        failures: list[str] = []
        if record.get("status") != "succeeded":
            failures.append(f"run status={record.get('status')!r}")
        if artifact.get("valid") is not True:
            failures.append("artifact.valid is not true")
        if not isinstance(strict, dict) or strict.get("checked") is not True:
            failures.append("strict_revalidation.checked is not true")
        elif strict.get("sha256_matches_original") is not True or strict.get("stable_during_validation") is not True:
            failures.append("strict SHA-256 match/stability checks did not both pass")
        if failures:
            raise ContinuationError(f"strict run-record verification failed for {case_id!r}: " + "; ".join(failures))


def verify_revalidation_summary(
    summary: dict[str, Any],
    prerequisite_suite_directory: Path,
    expected_suite_id: str,
    expected_case_ids: set[str],
) -> None:
    if summary.get("status") != "completed":
        raise ContinuationError(f"artifact revalidation status is {summary.get('status')!r}, not 'completed'")
    summary_counts = summary.get("summary")
    if not isinstance(summary_counts, dict):
        raise ContinuationError("artifact revalidation summary has no summary object")
    if summary_counts.get("suites_discovered") != 1:
        raise ContinuationError("artifact revalidation did not process exactly one suite")
    case_counts = summary_counts.get("cases")
    expected_valid_count = len(expected_case_ids)
    expected_counts = {
        "valid": expected_valid_count,
        "invalid": 0,
        "skipped": 0,
        "error": 0,
    }
    if not isinstance(case_counts, dict) or any(case_counts.get(key) != value for key, value in expected_counts.items()):
        raise ContinuationError(f"artifact revalidation case counts are not an all-valid result: actual={case_counts!r}, expected={expected_counts!r}")

    suite_results = summary.get("suites")
    if not isinstance(suite_results, list) or len(suite_results) != 1:
        raise ContinuationError("artifact revalidation has no unique suite result")
    suite_result = suite_results[0]
    if not isinstance(suite_result, dict):
        raise ContinuationError("artifact revalidation suite result is not an object")
    actual_directory = Path(str(suite_result.get("suite_directory", ""))).resolve(strict=False)
    if (
        suite_result.get("suite_id") != expected_suite_id
        or actual_directory != prerequisite_suite_directory
        or suite_result.get("suite_status") != "passed"
        or suite_result.get("status") != "completed"
    ):
        raise ContinuationError(f"artifact revalidation suite identity/status does not match the prerequisite: {suite_result!r}")
    case_results = suite_result.get("cases")
    if not isinstance(case_results, list):
        raise ContinuationError("artifact revalidation suite result has no cases list")
    actual_case_ids = {case.get("case_id") for case in case_results if isinstance(case, dict) and case.get("status") == "valid"}
    if len(case_results) != expected_valid_count or actual_case_ids != expected_case_ids or any(not isinstance(case, dict) or case.get("status") != "valid" for case in case_results):
        raise ContinuationError("artifact revalidation did not return one valid result for every prerequisite case")


class InferContinuation:
    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments
        self.prerequisite_suite_directory = arguments.prerequisite_suite.expanduser().resolve(strict=False)
        self.expected_prerequisite_suite_id = arguments.prerequisite_suite_id or self.prerequisite_suite_directory.name
        self.continuation_directory = CONTINUATIONS_ROOT / arguments.continuation_id
        self.continuation_directory.mkdir(parents=True, exist_ok=False)
        self.log_path = self.continuation_directory / "continuation.log"
        self.state_path = self.continuation_directory / "continuation.json"
        self.revalidation_summary_path = self.continuation_directory / "prerequisite_artifact_revalidation.json"
        self.reservation_path = RESERVATIONS_ROOT / f"{arguments.next_suite_id}.json"
        self.current_child: subprocess.Popen[Any] | None = None
        self.received_signal: int | None = None
        self.state: dict[str, Any] = {
            "schema_version": STATE_SCHEMA_VERSION,
            "continuation_id": arguments.continuation_id,
            "status": "initializing",
            "started_at_utc": utc_now(),
            "finished_at_utc": None,
            "error": "",
            "paths": {
                "continuation_directory": str(self.continuation_directory),
                "log": str(self.log_path),
                "state": str(self.state_path),
                "revalidation_summary": str(self.revalidation_summary_path),
            },
            "prerequisite": {
                "controller_pid": arguments.prerequisite_pid,
                "controller_start_time_ticks": (arguments.prerequisite_start_time_ticks),
                "suite_id": self.expected_prerequisite_suite_id,
                "suite_directory": str(self.prerequisite_suite_directory),
                "wait_outcome": None,
                "suite_status_before_revalidation": None,
                "suite_status_after_revalidation": None,
                "case_ids": [],
                "revalidation_command": [],
                "revalidation_exit_code": None,
                "strict_revalidation_passed": False,
            },
            "next_suite": {
                "suite_id": arguments.next_suite_id,
                "suite_directory": str(SUITES_ROOT / arguments.next_suite_id),
                "case_ids": list(arguments.only),
                "reservation": str(self.reservation_path),
                "reserved": False,
                "command": [],
                "status": "pending",
                "exit_code": None,
            },
            "signal": None,
        }
        self.persist()

    def persist(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def log(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(line + "\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())

    def set_status(self, status: str) -> None:
        self.state["status"] = status
        self.persist()

    def handle_signal(self, signal_number: int, _frame: Any) -> None:
        if self.received_signal is None:
            self.received_signal = signal_number
            self.state["signal"] = {
                "number": signal_number,
                "name": signal.Signals(signal_number).name,
                "received_at_utc": utc_now(),
            }
            self.persist()
        if self.current_child is not None and self.current_child.poll() is None:
            try:
                self.current_child.send_signal(signal_number)
            except ProcessLookupError:
                pass

    def raise_if_interrupted(self) -> None:
        if self.received_signal is not None:
            signal_name = signal.Signals(self.received_signal).name
            raise ContinuationInterrupted(f"continuation received {signal_name}")

    def ensure_next_suite_id_available(self) -> None:
        next_suite_directory = SUITES_ROOT / self.arguments.next_suite_id
        if next_suite_directory.exists():
            raise ContinuationError(f"next suite ID {self.arguments.next_suite_id!r} already exists at {next_suite_directory}; refusing duplicate launch")
        if self.reservation_path.exists():
            raise ContinuationError(f"next suite ID {self.arguments.next_suite_id!r} is already reserved at {self.reservation_path}; refusing duplicate launch")

    def reserve_next_suite_id(self) -> None:
        self.ensure_next_suite_id_available()
        self.reservation_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "suite_id": self.arguments.next_suite_id,
            "continuation_id": self.arguments.continuation_id,
            "reserved_at_utc": utc_now(),
            "continuation_state": str(self.state_path),
        }
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(self.reservation_path, flags, 0o644)
        except FileExistsError as exc:
            raise ContinuationError(f"next suite ID {self.arguments.next_suite_id!r} was reserved concurrently; refusing duplicate launch") from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file_obj:
                json.dump(payload, file_obj, ensure_ascii=False, indent=2)
                file_obj.write("\n")
                file_obj.flush()
                os.fsync(file_obj.fileno())
        except Exception:
            try:
                self.reservation_path.unlink()
            except OSError:
                pass
            raise
        if (SUITES_ROOT / self.arguments.next_suite_id).exists():
            raise ContinuationError(f"next suite ID {self.arguments.next_suite_id!r} appeared while it was being reserved; refusing duplicate launch")
        self.state["next_suite"]["reserved"] = True
        self.persist()

    def run_revalidation(self, expected_case_ids: set[str]) -> dict[str, Any]:
        command = [
            sys.executable,
            str(REVALIDATE_SCRIPT),
            str(self.prerequisite_suite_directory),
            "--output",
            str(self.revalidation_summary_path),
        ]
        self.state["prerequisite"]["revalidation_command"] = command
        self.persist()
        self.log("[Continuation] strict revalidation command: " + " ".join(command))
        completed = subprocess.run(
            command,
            cwd=REPO_PATH,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        self.state["prerequisite"]["revalidation_exit_code"] = completed.returncode
        self.persist()
        if completed.stdout:
            for line in completed.stdout.rstrip().splitlines():
                self.log(f"[Revalidation] {line}")
        if completed.returncode != 0:
            raise ContinuationError(f"strict artifact revalidation exited with {completed.returncode}")
        summary = read_json_object(
            self.revalidation_summary_path,
            "artifact revalidation summary",
        )
        verify_revalidation_summary(
            summary,
            self.prerequisite_suite_directory,
            self.expected_prerequisite_suite_id,
            expected_case_ids,
        )
        return summary

    def launch_next_suite(self) -> int:
        command = [
            sys.executable,
            str(RUN_SUITE_SCRIPT),
            "--suite-id",
            self.arguments.next_suite_id,
            "--case-timeout-seconds",
            str(self.arguments.case_timeout_seconds),
            "--idle-timeout-seconds",
            str(self.arguments.idle_timeout_seconds),
        ]
        for case_id in self.arguments.only:
            command.extend(["--only", case_id])
        self.state["next_suite"]["command"] = command
        self.state["next_suite"]["status"] = "running"
        self.persist()
        self.log("[Continuation] launching next suite: " + " ".join(command))
        with self.log_path.open("a", encoding="utf-8") as log_file:
            self.current_child = subprocess.Popen(
                command,
                cwd=REPO_PATH,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            return_code = self.current_child.wait()
        self.current_child = None
        self.state["next_suite"]["exit_code"] = return_code
        self.persist()
        self.raise_if_interrupted()
        return return_code

    def run(self) -> int:
        previous_sigint = signal.signal(signal.SIGINT, self.handle_signal)
        previous_sigterm = signal.signal(signal.SIGTERM, self.handle_signal)
        try:
            self.ensure_next_suite_id_available()
            self.set_status("waiting_for_prerequisite")
            self.log(f"[Continuation] waiting for prerequisite controller PID={self.arguments.prerequisite_pid}, starttime={self.arguments.prerequisite_start_time_ticks}")
            wait_outcome = wait_for_exact_process_exit(
                self.arguments.prerequisite_pid,
                self.arguments.prerequisite_start_time_ticks,
                self.arguments.poll_interval_seconds,
                self.arguments.wait_timeout_seconds,
                interrupted_fn=lambda: self.received_signal is not None,
            )
            self.state["prerequisite"]["wait_outcome"] = wait_outcome
            self.persist()
            self.log(f"[Continuation] prerequisite controller is no longer running ({wait_outcome})")
            self.raise_if_interrupted()

            suite_path = self.prerequisite_suite_directory / "suite.json"
            suite_before = read_json_object(suite_path, "prerequisite suite record")
            cases_before = validate_passed_suite(suite_before, self.expected_prerequisite_suite_id)
            expected_case_ids = {str(case["case_id"]) for case in cases_before}
            prerequisite_state = self.state["prerequisite"]
            prerequisite_state["suite_status_before_revalidation"] = suite_before["status"]
            prerequisite_state["case_ids"] = sorted(expected_case_ids)
            self.persist()

            self.ensure_next_suite_id_available()
            self.set_status("revalidating_prerequisite")
            self.run_revalidation(expected_case_ids)
            suite_after = read_json_object(suite_path, "revalidated prerequisite suite record")
            cases_after = validate_passed_suite(
                suite_after,
                self.expected_prerequisite_suite_id,
                expected_case_ids,
            )
            verify_strict_run_records(cases_after, self.prerequisite_suite_directory)
            prerequisite_state["suite_status_after_revalidation"] = suite_after["status"]
            prerequisite_state["strict_revalidation_passed"] = True
            self.persist()
            self.log(f"[Continuation] prerequisite suite passed strict revalidation ({len(cases_after)} cases)")
            self.raise_if_interrupted()

            self.reserve_next_suite_id()
            self.set_status("running_next_suite")
            return_code = self.launch_next_suite()
            if return_code != 0:
                raise ContinuationError(f"next suite controller exited with {return_code}")
            next_suite_path = SUITES_ROOT / self.arguments.next_suite_id / "suite.json"
            next_suite = read_json_object(next_suite_path, "next suite record")
            validate_passed_suite(
                next_suite,
                self.arguments.next_suite_id,
                set(self.arguments.only),
            )
            self.state["next_suite"]["status"] = "passed"
            self.state["status"] = "passed"
            self.state["finished_at_utc"] = utc_now()
            self.persist()
            self.log(f"[Continuation] FINISH status=passed, next_suite={self.arguments.next_suite_id}")
            return 0
        except ContinuationInterrupted as exc:
            self.state["status"] = "interrupted"
            self.state["error"] = str(exc)
            self.state["next_suite"]["status"] = "interrupted" if self.state["next_suite"]["status"] == "running" else self.state["next_suite"]["status"]
            self.state["finished_at_utc"] = utc_now()
            self.persist()
            self.log(f"[Continuation] FINISH status=interrupted: {exc}")
            return 128 + (self.received_signal or signal.SIGTERM)
        except (ContinuationError, OSError, subprocess.SubprocessError) as exc:
            self.state["status"] = "failed"
            self.state["error"] = str(exc)
            if self.state["next_suite"]["status"] == "running":
                self.state["next_suite"]["status"] = "failed"
            self.state["finished_at_utc"] = utc_now()
            self.persist()
            self.log(f"[Continuation] FINISH status=failed: {exc}")
            return 1
        except Exception as exc:
            error = f"unexpected {type(exc).__name__} in continuation: {exc}"
            self.state["status"] = "failed"
            self.state["error"] = error
            if self.state["next_suite"]["status"] == "running":
                self.state["next_suite"]["status"] = "failed"
            self.state["finished_at_utc"] = utc_now()
            self.persist()
            self.log(f"[Continuation] FINISH status=failed: {error}")
            return 1
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--continuation-id",
        required=True,
        type=lambda value: validate_identifier(value, "--continuation-id"),
        help="Unique, single-use ID for this continuation.",
    )
    parser.add_argument(
        "--prerequisite-pid",
        required=True,
        type=int,
        help="PID of the prerequisite run_infer_suite.py controller.",
    )
    parser.add_argument(
        "--prerequisite-start-time-ticks",
        required=True,
        type=int,
        help="Linux /proc/PID/stat field 22 captured for that controller.",
    )
    parser.add_argument(
        "--prerequisite-suite",
        required=True,
        type=Path,
        help="Directory containing the prerequisite suite.json.",
    )
    parser.add_argument(
        "--prerequisite-suite-id",
        type=lambda value: validate_identifier(value, "--prerequisite-suite-id"),
        help="Expected suite_id; defaults to prerequisite directory name.",
    )
    parser.add_argument(
        "--next-suite-id",
        required=True,
        type=lambda value: validate_identifier(value, "--next-suite-id"),
        help="Unique suite ID to reserve and pass to run_infer_suite.py.",
    )
    parser.add_argument(
        "--only",
        action="append",
        required=True,
        type=lambda value: validate_identifier(value, "--only"),
        help="Exact next-suite case_id; repeat for every requested case.",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=int,
        default=8 * 60 * 60,
        help="Per-case timeout passed to run_infer_suite.py.",
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=int,
        default=30 * 60,
        help="NPU idle wait timeout passed to run_infer_suite.py.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=5.0,
        help="Prerequisite PID poll interval (default: 5).",
    )
    parser.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=0.0,
        help="Prerequisite wait timeout; 0 waits indefinitely.",
    )
    arguments = parser.parse_args(argv)
    if arguments.prerequisite_pid <= 0:
        parser.error("--prerequisite-pid must be positive")
    if arguments.prerequisite_start_time_ticks <= 0:
        parser.error("--prerequisite-start-time-ticks must be positive")
    if arguments.case_timeout_seconds < 60:
        parser.error("--case-timeout-seconds must be at least 60")
    if arguments.idle_timeout_seconds < 10:
        parser.error("--idle-timeout-seconds must be at least 10")
    if arguments.poll_interval_seconds <= 0:
        parser.error("--poll-interval-seconds must be positive")
    if arguments.wait_timeout_seconds < 0:
        parser.error("--wait-timeout-seconds cannot be negative")
    if len(arguments.only) != len(set(arguments.only)):
        parser.error("--only case IDs must not be duplicated")
    if arguments.next_suite_id == (arguments.prerequisite_suite_id or arguments.prerequisite_suite.name):
        parser.error("--next-suite-id must differ from the prerequisite suite ID")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        continuation = InferContinuation(arguments)
    except FileExistsError:
        path = CONTINUATIONS_ROOT / arguments.continuation_id
        print(
            f"[Continuation] ERROR: continuation ID {arguments.continuation_id!r} already exists at {path}; refusing duplicate start",
            file=sys.stderr,
        )
        return 1
    return continuation.run()


if __name__ == "__main__":
    raise SystemExit(main())
