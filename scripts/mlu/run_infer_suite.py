#!/usr/bin/env python3
"""Run the MLU590 offline-inference matrix sequentially with resumable state."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_PATH = Path("/data/Lightx2v-Platform-Run-Examples")
SCRIPTS_PATH = REPO_PATH / "scripts"
INFER_ROOT = REPO_PATH / "scripts" / "mlu" / "infer"
SUITE_ROOT = REPO_PATH / "logs" / "mlu" / "infer" / "suites"

sys.path.insert(0, str(SCRIPTS_PATH))
import run_record as run_record_tools  # noqa: E402

# Cheap cases go first so environment and distributed setup failures surface
# quickly. Cases sharing a model stay adjacent to benefit from the page cache.
CASES: tuple[tuple[str, str, str], ...] = (
    ("single", "z_image_turbo_t2i_1664x928", "single/run_z_image_turbo_t2i_1664x928.sh"),
    ("dist_2", "z_image_turbo_t2i_1664x928_sp2", "dist_2/run_z_image_turbo_t2i_1664x928_sp2.sh"),
    ("single", "wan21_1_3b_self_forcing_t2v_480p_81f", "single/run_wan21_1_3b_self_forcing_t2v_480p_81f.sh"),
    ("single", "wan21_1_3b_t2v_480p_81f", "single/run_wan21_1_3b_t2v_480p_81f.sh"),
    ("dist_8", "wan21_1_3b_t2v_480p_81f_cfg2_sp4", "dist_8/run_wan21_1_3b_t2v_480p_81f_cfg2_sp4.sh"),
    ("single", "longcat_image_t2i_1344x768", "single/run_longcat_image_t2i_1344x768.sh"),
    ("dist_8", "longcat_image_t2i_1344x768_cfg2_sp4", "dist_8/run_longcat_image_t2i_1344x768_cfg2_sp4.sh"),
    ("single", "qwen_image_2512_t2i_1664x928", "single/run_qwen_image_2512_t2i_1664x928.sh"),
    ("dist_8", "qwen_image_2512_t2i_1664x928_cfg2_sp4", "dist_8/run_qwen_image_2512_t2i_1664x928_cfg2_sp4.sh"),
    ("single", "flux2_dev_t2i_1344x768", "single/run_flux2_dev_t2i_1344x768.sh"),
    ("dist_8", "flux2_dev_t2i_1344x768_tp8", "dist_8/run_flux2_dev_t2i_1344x768_tp8.sh"),
    ("single", "wan22_moe_a14b_t2v_480p_81f", "single/run_wan22_moe_a14b_t2v_480p_81f.sh"),
    ("dist_8", "wan22_moe_a14b_t2v_480p_81f_cfg2_sp4", "dist_8/run_wan22_moe_a14b_t2v_480p_81f_cfg2_sp4.sh"),
    ("dist_8", "wan22_moe_a14b_t2v_480p_81f_tp8", "dist_8/run_wan22_moe_a14b_t2v_480p_81f_tp8.sh"),
    ("single", "wan22_moe_a14b_t2v_720p_81f", "single/run_wan22_moe_a14b_t2v_720p_81f.sh"),
    ("dist_8", "wan22_moe_a14b_t2v_720p_81f_cfg2_sp4", "dist_8/run_wan22_moe_a14b_t2v_720p_81f_cfg2_sp4.sh"),
    ("dist_8", "wan22_moe_a14b_t2v_720p_81f_tp8", "dist_8/run_wan22_moe_a14b_t2v_720p_81f_tp8.sh"),
    ("single", "hunyuan_video_15_t2v_480p_121f", "single/run_hunyuan_video_15_t2v_480p_121f.sh"),
    ("dist_8", "hunyuan_video_15_t2v_480p_121f_cfg2_sp4", "dist_8/run_hunyuan_video_15_t2v_480p_121f_cfg2_sp4.sh"),
    ("single", "hunyuan_video_15_t2v_720p_121f", "single/run_hunyuan_video_15_t2v_720p_121f.sh"),
    ("dist_8", "hunyuan_video_15_t2v_720p_121f_cfg2_sp4", "dist_8/run_hunyuan_video_15_t2v_720p_121f_cfg2_sp4.sh"),
    ("single", "ltx2_3_22b_dev_s2v_768x512_241f", "single/run_ltx2_3_22b_dev_s2v_768x512_241f.sh"),
    ("dist_8", "ltx2_3_22b_dev_s2v_768x512_241f_sp8", "dist_8/run_ltx2_3_22b_dev_s2v_768x512_241f_sp8.sh"),
)

ACTIVE_PROCESS: subprocess.Popen[bytes] | None = None
INTERRUPTED_SIGNAL: int | None = None
SUITE_LOCK_HANDLE: Any = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    os.replace(temporary, path)


def acquire_suite_lock(suite_dir: Path) -> None:
    global SUITE_LOCK_HANDLE

    lock_path = suite_dir / "controller.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(
            lock_handle.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError:
        lock_handle.close()
        raise RuntimeError(
            f"suite already has an active controller: {lock_path}"
        ) from None
    SUITE_LOCK_HANDLE = lock_handle


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def handle_signal(signum: int, _frame: object) -> None:
    global INTERRUPTED_SIGNAL
    INTERRUPTED_SIGNAL = signum
    process = ACTIVE_PROCESS
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def select_cases(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    selected = list(CASES)
    if args.scope == "single":
        selected = [case for case in selected if case[0] == "single"]
    elif args.scope == "multi":
        selected = [case for case in selected if case[0] != "single"]
    if args.only:
        requested = set(args.only)
        known = {case_id for _, case_id, _ in CASES}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown --only case(s): {', '.join(unknown)}")
        selected = [case for case in selected if case[1] in requested]
    return selected


def build_initial_state(suite_id: str, selected: list[tuple[str, str, str]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite_id": suite_id,
        "platform": "cambricon_mlu",
        "device_model": "MLU590-M9D",
        "device_memory_gib": 80,
        "repo_path": str(REPO_PATH),
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "status": "running",
        "pid": os.getpid(),
        "cases": [
            {
                "index": index,
                "group": group,
                "case_id": case_id,
                "script": str(INFER_ROOT / relative_script),
                "status": "pending",
                "attempts": [],
            }
            for index, (group, case_id, relative_script) in enumerate(selected, start=1)
        ],
    }


def validate_case_files(state: dict[str, Any]) -> None:
    missing = [
        case["script"]
        for case in state["cases"]
        if not Path(case["script"]).is_file()
    ]
    if missing:
        raise FileNotFoundError("missing suite script(s): " + ", ".join(missing))


def successful_attempt_is_valid(case: dict[str, Any]) -> tuple[bool, str]:
    attempts = case.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return False, "successful case has no attempts"
    latest_attempt = attempts[-1]
    if not isinstance(latest_attempt, dict):
        return False, "latest attempt is not an object"
    record_value = latest_attempt.get("run_record")
    if not isinstance(record_value, str) or not record_value:
        return False, "latest attempt has no run record path"
    record_path = Path(record_value)
    if not record_path.is_file():
        return False, f"run record does not exist: {record_path}"
    try:
        record = load_json(record_path)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"cannot read run record: {exc}"
    if record.get("status") != "succeeded":
        return False, f"run record status is {record.get('status')!r}"
    artifact = record.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("valid") is not True:
        return False, "run record artifact is not valid"
    artifact_path_value = artifact.get("path")
    if not isinstance(artifact_path_value, str) or not artifact_path_value:
        return False, "run record artifact has no path"
    artifact_path = Path(artifact_path_value)
    if not artifact_path.is_file():
        return False, f"result artifact does not exist: {artifact_path}"
    benchmark = record.get("benchmark")
    if not isinstance(benchmark, dict):
        return False, "run record benchmark is not an object"
    target = benchmark.get("target")
    if not isinstance(target, dict):
        return False, "run record target is not an object"
    result_format = str(artifact.get("format", "")).casefold().lstrip(".")
    try:
        fresh_artifact = run_record_tools.validate_artifact(
            artifact_path,
            result_format,
            target,
            expected_audio=benchmark.get("task") in {"s2v", "ltx2_s2v"},
        )
    except Exception as exc:
        return False, f"fresh artifact validation raised {type(exc).__name__}: {exc}"
    if fresh_artifact.get("valid") is not True:
        validation = fresh_artifact.get("validation")
        errors = validation.get("errors") if isinstance(validation, dict) else None
        reason = "; ".join(str(error) for error in errors) if isinstance(errors, list) else ""
        return False, reason or "fresh artifact validation failed"
    recorded_sha256 = artifact.get("sha256")
    current_sha256 = fresh_artifact.get("sha256")
    if not isinstance(recorded_sha256, str) or recorded_sha256 != current_sha256:
        return False, "result artifact SHA-256 changed after run completion"
    return True, ""


def run_suite(args: argparse.Namespace) -> int:
    global ACTIVE_PROCESS

    selected = select_cases(args)
    if not selected:
        raise ValueError("no cases selected")

    suite_id = args.resume or args.suite_id or datetime.now(timezone.utc).strftime("mlu_%Y%m%dT%H%M%SZ")
    if not suite_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"invalid suite id: {suite_id}")
    suite_dir = SUITE_ROOT / suite_id
    state_path = suite_dir / "suite.json"
    suite_dir.mkdir(parents=True, exist_ok=True)
    acquire_suite_lock(suite_dir)

    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"resume state does not exist: {state_path}")
        state = load_json(state_path)
        state["status"] = "running"
        state["finished_at_utc"] = None
        state["pid"] = os.getpid()
    else:
        if state_path.exists():
            raise FileExistsError(f"suite state already exists: {state_path}")
        state = build_initial_state(suite_id, selected)
        atomic_write_json(state_path, state)

    validate_case_files(state)
    print(f"[Suite] id={suite_id}", flush=True)
    print(f"[Suite] state={state_path}", flush=True)

    for case in state["cases"]:
        if INTERRUPTED_SIGNAL is not None:
            break
        if case["status"] == "success":
            valid, reason = successful_attempt_is_valid(case)
            if valid:
                print(f"[Suite] skip successful case: {case['case_id']}", flush=True)
                continue
            case["status"] = "failed"
            atomic_write_json(state_path, state)
            print(
                f"[Suite] retry stale successful case {case['case_id']}: {reason}",
                flush=True,
            )

        attempt_number = len(case["attempts"]) + 1
        run_id = f"{suite_id}_{int(case['index']):02d}_a{attempt_number}"
        group = str(case["group"])
        case_id = str(case["case_id"])
        record_path = REPO_PATH / "logs" / "mlu" / "infer" / group / case_id / run_id / "run.json"
        result_dir = REPO_PATH / "results" / "mlu" / "infer" / group / case_id / run_id
        attempt = {
            "attempt": attempt_number,
            "run_id": run_id,
            "started_at_utc": utc_now(),
            "finished_at_utc": None,
            "elapsed_seconds": None,
            "exit_code": None,
            "run_record": str(record_path),
            "result_dir": str(result_dir),
        }
        case["attempts"].append(attempt)
        case["status"] = "running"
        atomic_write_json(state_path, state)

        env = os.environ.copy()
        env["RUN_ID"] = run_id
        env["MASTER_PORT"] = str(29600 + int(case["index"]))
        env["MLU_VISIBLE_DEVICES"] = {
            "single": "0",
            "dist_2": "0,1",
            "dist_8": "0,1,2,3,4,5,6,7",
        }[group]
        env.pop("CN_VISIBLE_DEVICES", None)
        started = time.monotonic()
        print(
            f"\n[Suite] ({case['index']}/{len(state['cases'])}) start "
            f"{group}/{case_id}, run_id={run_id}",
            flush=True,
        )
        ACTIVE_PROCESS = subprocess.Popen(
            ["bash", str(case["script"])],
            cwd=REPO_PATH,
            env=env,
            start_new_session=True,
        )
        exit_code = ACTIVE_PROCESS.wait()
        ACTIVE_PROCESS = None

        attempt["exit_code"] = exit_code
        attempt["finished_at_utc"] = utc_now()
        attempt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        record = load_json(record_path) if record_path.is_file() else None
        recorded_status = record.get("status") if record else None
        case["status"] = (
            "success"
            if exit_code == 0 and recorded_status == "succeeded"
            else "failed"
        )
        print(
            f"[Suite] finish {case_id}: status={case['status']}, "
            f"exit_code={exit_code}, elapsed={attempt['elapsed_seconds']}s",
            flush=True,
        )
        atomic_write_json(state_path, state)
        if case["status"] == "failed" and args.stop_on_error:
            break

    if INTERRUPTED_SIGNAL is not None:
        state["status"] = "interrupted"
    elif any(case["status"] == "failed" for case in state["cases"]):
        state["status"] = "completed_with_failures"
    elif all(case["status"] == "success" for case in state["cases"]):
        state["status"] = "success"
    else:
        state["status"] = "incomplete"
    state["finished_at_utc"] = utc_now()
    atomic_write_json(state_path, state)

    success_count = sum(case["status"] == "success" for case in state["cases"])
    failed_count = sum(case["status"] == "failed" for case in state["cases"])
    print(
        f"\n[Suite] status={state['status']}, success={success_count}, "
        f"failed={failed_count}, total={len(state['cases'])}",
        flush=True,
    )
    if INTERRUPTED_SIGNAL is not None:
        return 128 + INTERRUPTED_SIGNAL
    return 0 if state["status"] == "success" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("all", "single", "multi"), default="all")
    parser.add_argument("--only", action="append", help="run one named case; repeat as needed")
    suite_selection = parser.add_mutually_exclusive_group()
    suite_selection.add_argument("--suite-id", help="new suite identifier")
    suite_selection.add_argument("--resume", help="resume an existing suite identifier")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        return run_suite(parse_args())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[Suite] fatal: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
