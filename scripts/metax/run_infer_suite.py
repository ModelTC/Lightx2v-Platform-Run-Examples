#!/usr/bin/env python3
"""Run the MetaX C500 offline-inference matrix sequentially with resumable state."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from collections import namedtuple
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_PATH = Path("/data/Lightx2v-Platform-Run-Examples")
INFER_ROOT = REPO_PATH / "scripts" / "metax" / "infer"
SUITE_ROOT = REPO_PATH / "logs" / "metax" / "infer" / "suites"
METAX_GPU_LOCK_NAME = "metax_gpu.lock"
METAX_MCCL_SHM_PATH = Path("/dev/shm")
MCCL_FILE_OWNER_RE = re.compile(r"^mccl-([0-9a-fA-F]+)-")
RUN_RECORD_TOOL = REPO_PATH / "scripts" / "run_record.py"
FINAL_REPORT_TOOL = REPO_PATH / "scripts" / "aggregate_infer_reports.py"
DEFAULT_STALL_TIMEOUT_SECONDS = 1800
DEFAULT_GPU_RELEASE_COOLDOWN_SECONDS = 30.0
WATCHDOG_POLL_INTERVAL_SECONDS = 30.0
WATCHDOG_TERMINATION_GRACE_SECONDS = 30.0
WATCHDOG_TERMINATION_POLL_SECONDS = 0.1
DEFAULT_QUEUE_RETRY_TIMEOUT_SECONDS = 300.0
DEFAULT_QUEUE_RETRY_MIN_COUNT = 12
METAX_QUEUE_RETRY_MARKER = b"[mxkwCreateQueueBlock][Hint]ioctl create queue block timeout"
METAX_QUEUE_RETRY_TYPE_MARKER = b"type:21."
METAX_QUEUE_RETRY_ACTION_MARKER = b"Retrying."
VALIDATION_RECOVERY_SCHEMA_VERSION = "1.0"
VALIDATION_RECOVERY_ARCHIVE_NAME = "run.pre_validation_recovery.json"
# Validation-only recovery is deliberately limited to the exact historical
# artifacts reviewed for this suite.  A valid MP4 container cannot establish
# that an older model/TP numerical path was correct, so neither identifiers nor
# the recorded artifact digest may drift.
VALIDATION_ONLY_RECOVERY_ALLOWLIST = {
    (
        "wan22_moe_a14b_t2v_480p_81f",
        "metax_20260725T072300Z_12_a1",
    ): "ca77c1d592ed172982fe2d4c42fc976873e6996e20ad0cae01544e16c51e255f",
    (
        "wan22_moe_a14b_t2v_720p_81f",
        "metax_20260725T072300Z_15_a1",
    ): "a92524d4057247c580e2ed204b1f057832e7adccd53293b6c0d9e016139db6c2",
    (
        "hunyuan_video_15_t2v_480p_121f",
        "metax_20260725T072300Z_18_a1",
    ): "b5bccb2205ded77ccf1b87168f24b4aa67584a3d8ff0bcbd22fb256a1a86aaa7",
    (
        "hunyuan_video_15_t2v_720p_121f",
        "metax_20260725T072300Z_20_a1",
    ): "ca017ddad3b04f233cc1f476ef3ed5fc90c4f027a1166f0201f20e49aa73230d",
}
# These two attempts predate ltx_tmp's corrected Wan TP weight splitting and
# global RMSNorm.  A valid container cannot certify their old numerical result.
NON_RECOVERABLE_VALIDATION_ONLY_RUNS = {
    (
        "wan22_moe_a14b_t2v_480p_81f_tp8",
        "metax_20260725T072300Z_14_a1",
    ),
    (
        "wan22_moe_a14b_t2v_720p_81f_tp8",
        "metax_20260725T072300Z_17_a1",
    ),
}

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
RUN_RECORD_MODULE: Any | None = None


class _QueueRetryWatch:
    """Incrementally recognize the known MetaX type:21 queue deadlock."""

    def __init__(self) -> None:
        self.log_offset = 0
        self.partial_line = b""
        self.first_retry_monotonic: float | None = None
        self.last_retry_monotonic: float | None = None
        self.retry_count = 0
        self.total_retry_count = 0

    def reset_incident(self) -> None:
        self.first_retry_monotonic = None
        self.last_retry_monotonic = None
        self.retry_count = 0

    def incident_is_armed(self, minimum_count: int) -> bool:
        return self.first_retry_monotonic is not None and self.retry_count >= minimum_count

    def consume_log(
        self,
        run_log_path: Path,
        *,
        now_monotonic: float,
    ) -> bool:
        business_progress = False
        lines, self.log_offset, self.partial_line = _read_appended_log_lines(
            run_log_path,
            self.log_offset,
            self.partial_line,
        )
        for line in lines:
            if _is_metax_queue_retry_line(line):
                if self.first_retry_monotonic is None:
                    self.first_retry_monotonic = now_monotonic
                self.last_retry_monotonic = now_monotonic
                self.retry_count += 1
                self.total_retry_count += 1
            elif line.strip():
                # Any non-retry application/runtime output is treated as
                # recovery. This deliberately biases the early detector
                # against false positives; the ordinary stall watchdog still
                # handles silent hangs.
                self.reset_incident()
                business_progress = True
        return business_progress


# Named tuples avoid importing-module registration requirements when the suite
# tool is loaded directly through importlib in lightweight validation tests.
_ProcessIdentity = namedtuple(
    "_ProcessIdentity",
    ("pid", "start_time_ticks"),
)
_ProcStat = namedtuple(
    "_ProcStat",
    ("identity", "state", "parent_pid", "process_group_id"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def generate_final_report() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(FINAL_REPORT_TOOL), "--platform", "metax"],
        cwd=REPO_PATH,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print(f"[Suite] {result.stdout.strip()}", flush=True)
    if result.stderr.strip():
        print(f"[Suite] report stderr: {result.stderr.strip()}", flush=True)
    return {
        "status": "succeeded" if result.returncode == 0 else "failed",
        "exit_code": result.returncode,
        "generated_at_utc": utc_now(),
        "markdown": str(SUITE_ROOT / "final_report.md"),
        "json": str(SUITE_ROOT / "final_report.json"),
        "error": result.stderr.strip() if result.returncode else "",
    }


def generate_benchmark_summary(state: dict[str, Any], suite_dir: Path) -> dict[str, str]:
    """Write a compact summary using only the final requested repetition."""
    repetitions = int(state.get("repetitions", 1))
    rows: list[dict[str, Any]] = []
    for case in state["cases"]:
        if int(case.get("repetition", 1)) != repetitions:
            continue
        attempts = case.get("attempts") or []
        attempt = attempts[-1] if attempts else {}
        record_path_value = str(attempt.get("run_record", ""))
        record_path = Path(record_path_value) if record_path_value else None
        record = (
            load_json(record_path)
            if record_path is not None and record_path.is_file()
            else {}
        )
        metrics = record.get("metrics") if isinstance(record, dict) else {}
        metrics = metrics if isinstance(metrics, dict) else {}
        profile = metrics.get("profile_seconds")
        profile = profile if isinstance(profile, dict) else {}
        benchmark = record.get("benchmark") if isinstance(record, dict) else {}
        benchmark = benchmark if isinstance(benchmark, dict) else {}
        configuration = record.get("configuration") if isinstance(record, dict) else {}
        configuration = configuration if isinstance(configuration, dict) else {}
        config_file = configuration.get("file")
        config_file = config_file if isinstance(config_file, dict) else {}
        rows.append(
            {
                "group": case["group"],
                "case_id": case["case_id"],
                "repetition": repetitions,
                "status": case["status"],
                "parallel_strategy": benchmark.get("parallel_strategy"),
                "config_path": config_file.get("path"),
                "config_sha256": config_file.get("sha256"),
                "steady_dit_seconds_per_step": metrics.get("dit_steady_state_seconds"),
                "pipeline_cost_seconds": profile.get("pipeline"),
                "run_log": (
                    str(record_path.with_name("run.log"))
                    if record_path is not None
                    else ""
                ),
                "run_record": str(record_path) if record_path is not None else "",
            }
        )

    payload = {
        "suite_id": state["suite_id"],
        "repetition_used": repetitions,
        "metric_definitions": {
            "steady_dit_seconds_per_step": "median synchronized DiT step cost after excluding the first step; distributed steps use the slowest rank",
            "pipeline_cost_seconds": "LightX2V RUN pipeline profile cost with output-file saving disabled",
        },
        "cases": rows,
    }
    json_path = suite_dir / "second_run_results.json"
    markdown_path = suite_dir / "second_run_results.md"
    atomic_write_json(json_path, payload)
    lines = [
        f"# MetaX no-save benchmark: {state['suite_id']}",
        "",
        f"Only repetition {repetitions} is authoritative.",
        "",
        "| Group | Case | Parallel | Status | Steady DiT (s/step) | Pipeline cost (s) | Run log |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        steady = row["steady_dit_seconds_per_step"]
        pipeline = row["pipeline_cost_seconds"]
        steady_text = f"{steady:.6f}" if isinstance(steady, (int, float)) else "—"
        pipeline_text = f"{pipeline:.6f}" if isinstance(pipeline, (int, float)) else "—"
        lines.append(
            f"| {row['group']} | `{row['case_id']}` | {row['parallel_strategy'] or '—'} | "
            f"{row['status']} | {steady_text} | {pipeline_text} | `{row['run_log']}` |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def gpu_release_cooldown_remaining(
    last_release_monotonic: float,
    now_monotonic: float,
    cooldown_seconds: float,
) -> float:
    return max(
        float(last_release_monotonic) + max(float(cooldown_seconds), 0.0) - float(now_monotonic),
        0.0,
    )


def case_requires_gpu_release_cooldown(group: str) -> bool:
    return group in {"dist_2", "dist_8"}


def wait_for_gpu_release_cooldown(
    last_release_monotonic: float,
    cooldown_seconds: float,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Wait interruptibly before initializing a new MCCL communicator."""
    started = monotonic()
    remaining = gpu_release_cooldown_remaining(
        last_release_monotonic,
        started,
        cooldown_seconds,
    )
    if remaining <= 0:
        return 0.0

    print(
        f"[Suite] MetaX GPU-runtime release cooldown before distributed case: {remaining:.1f}s remaining",
        flush=True,
    )
    while remaining > 0 and INTERRUPTED_SIGNAL is None:
        sleep(min(1.0, remaining))
        remaining = gpu_release_cooldown_remaining(
            last_release_monotonic,
            monotonic(),
            cooldown_seconds,
        )
    return max(monotonic() - started, 0.0)


def snapshot_mccl_shm_files(
    shm_path: Path = METAX_MCCL_SHM_PATH,
    *,
    errors: list[str] | None = None,
) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    try:
        for candidate in shm_path.glob("mccl-*"):
            try:
                candidate_stat = candidate.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                if errors is not None:
                    errors.append(f"cannot inspect MCCL path {candidate}: {exc}")
                continue
            if stat.S_ISREG(candidate_stat.st_mode):
                snapshot[candidate.name] = (
                    candidate_stat.st_ino,
                    candidate_stat.st_size,
                )
    except OSError as exc:
        if errors is not None:
            errors.append(f"cannot scan MCCL shared memory at {shm_path}: {exc}")
    return snapshot


def quarantine_new_mccl_shm_files(
    before: dict[str, tuple[int, int]],
    destination: Path,
    owned_process_ids: set[int] | dict[int, set[int]],
    shm_path: Path = METAX_MCCL_SHM_PATH,
    proc_path: Path = Path("/proc"),
) -> dict[str, Any]:
    """Move only MCCL files attributed to this runner's exited process tree."""
    moved: list[dict[str, Any]] = []
    errors: list[str] = []
    after = snapshot_mccl_shm_files(shm_path, errors=errors)
    new_names = sorted(name for name, fingerprint in after.items() if before.get(name) != fingerprint)
    eligible_names: list[str] = []
    skipped_unowned_files: list[str] = []
    skipped_reused_pid_files: list[str] = []
    if isinstance(owned_process_ids, dict):
        owned_pids = set(owned_process_ids)
        guarded_start_times = owned_process_ids
    else:
        owned_pids = owned_process_ids
        guarded_start_times = None
    for name in new_names:
        owner_match = MCCL_FILE_OWNER_RE.match(name)
        owner_pid = int(owner_match.group(1), 16) if owner_match else None
        if owner_pid not in owned_pids:
            skipped_unowned_files.append(name)
            continue
        if guarded_start_times is not None and owner_pid is not None:
            current_stat = _proc_stat_for_pid(owner_pid, proc_path)
            if current_stat is not None and current_stat.identity.start_time_ticks not in guarded_start_times[owner_pid]:
                skipped_reused_pid_files.append(name)
                continue
        eligible_names.append(name)

    if eligible_names:
        try:
            destination.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"cannot create MCCL quarantine {destination}: {exc}")
            eligible_names = []

    for name in eligible_names:
        source = shm_path / name
        target = destination / name
        try:
            source_stat = source.lstat()
            if not stat.S_ISREG(source_stat.st_mode):
                errors.append(f"refusing non-regular MCCL path: {source}")
                continue
            current_fingerprint = (source_stat.st_ino, source_stat.st_size)
            if current_fingerprint != after[name]:
                errors.append(f"refusing changed MCCL path: {source}")
                continue
            if target.exists():
                errors.append(f"quarantine target already exists: {target}")
                continue
            shutil.move(str(source), str(target))
            moved.append(
                {
                    "source": str(source),
                    "quarantine": str(target),
                    "size_bytes": source_stat.st_size,
                }
            )
        except FileNotFoundError:
            continue
        except OSError as exc:
            errors.append(f"cannot quarantine {source}: {exc}")
    return {
        "checked": True,
        "created_file_count": len(new_names),
        "eligible_file_count": len(eligible_names),
        "quarantined_file_count": len(moved),
        "quarantined_bytes": sum(item["size_bytes"] for item in moved),
        "files": moved,
        "skipped_unowned_files": skipped_unowned_files,
        "skipped_reused_pid_files": skipped_reused_pid_files,
        "errors": errors,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as file_obj:
            file_obj.write(payload)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def acquire_suite_lock(suite_dir: Path):
    """Hold an exclusive controller lock until the returned file is closed."""
    suite_dir.mkdir(parents=True, exist_ok=True)
    lock_path = suite_dir / "controller.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        raise RuntimeError(f"suite controller lock is already held: {lock_path}") from None
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()
    return lock_file


def acquire_metax_gpu_lock(suite_root: Path, suite_id: str):
    """Exclusively reserve the fixed MetaX GPU pool for one suite controller."""
    suite_root.mkdir(parents=True, exist_ok=True)
    lock_path = suite_root / METAX_GPU_LOCK_NAME
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.seek(0)
        owner = lock_file.read().strip()
        lock_file.close()
        owner_text = f" ({owner})" if owner else ""
        raise RuntimeError(f"MetaX GPU global controller lock is already held: {lock_path}{owner_text}") from None
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"pid={os.getpid()} suite_id={suite_id}\n")
    lock_file.flush()
    return lock_file


def reject_active_legacy_controllers(
    suite_root: Path,
    current_suite_dir: Path,
) -> None:
    """Reject controllers started before the fixed global GPU lock existed."""
    active_locks: list[str] = []
    for lock_path in sorted(suite_root.glob("*/controller.lock")):
        if lock_path.parent == current_suite_dir:
            continue
        lock_file = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            lock_file.seek(0)
            owner = lock_file.read().strip()
            owner_text = f" ({owner})" if owner else ""
            active_locks.append(f"{lock_path}{owner_text}")
        finally:
            lock_file.close()
    if active_locks:
        raise RuntimeError("active MetaX suite controller(s) without the global GPU lock: " + ", ".join(active_locks))


@contextmanager
def hold_controller_locks(
    suite_root: Path,
    suite_dir: Path,
    suite_id: str,
):
    """Hold the global GPU and per-suite locks, releasing both on every exit."""
    global_lock = acquire_metax_gpu_lock(suite_root, suite_id)
    try:
        suite_lock = acquire_suite_lock(suite_dir)
        try:
            reject_active_legacy_controllers(suite_root, suite_dir)
            yield
        finally:
            suite_lock.close()
    finally:
        global_lock.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run_record_module() -> Any:
    """Load the CPU-only artifact validator without initializing MetaX."""
    global RUN_RECORD_MODULE
    if RUN_RECORD_MODULE is not None:
        return RUN_RECORD_MODULE
    spec = importlib.util.spec_from_file_location(
        "_metax_suite_run_record",
        RUN_RECORD_TOOL,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load artifact validator: {RUN_RECORD_TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    RUN_RECORD_MODULE = module
    return module


def recover_validation_only_failure(
    case: dict[str, Any],
) -> tuple[bool, str, dict[str, Any] | None]:
    """Recover an exit-74 run only when its unchanged artifact is now valid."""
    attempts = case.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return False, "", None
    attempt = attempts[-1]
    if not isinstance(attempt, dict) or attempt.get("exit_code") != 74:
        return False, "", None
    if (
        case.get("case_id"),
        attempt.get("run_id"),
    ) in NON_RECOVERABLE_VALIDATION_ONLY_RUNS:
        return (
            False,
            "ltx_tmp changes the Wan TP numerical path; the old artifact must be regenerated",
            None,
        )
    recovery_key = (case.get("case_id"), attempt.get("run_id"))
    allowlisted_artifact_sha256 = VALIDATION_ONLY_RECOVERY_ALLOWLIST.get(recovery_key)
    if allowlisted_artifact_sha256 is None:
        return (
            False,
            "historical attempt is not in the exact validation-only recovery allowlist",
            None,
        )

    record_value = attempt.get("run_record")
    if not isinstance(record_value, str) or not record_value:
        return False, "validation-only attempt has no run record", None
    record_path = Path(record_value)
    if not record_path.is_file():
        return False, f"run record is missing: {record_path}", None
    try:
        record = load_json(record_path)
        original_record_bytes = record_path.read_bytes()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"run record is invalid: {exc}", None
    try:
        if json.loads(original_record_bytes) != record:
            return False, "run record changed while it was being read", None
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"run record bytes are invalid: {exc}", None
    original_record_sha256 = hashlib.sha256(original_record_bytes).hexdigest()

    # If the run record was recovered but the suite update was interrupted,
    # finish synchronizing the suite without rewriting the archival record.
    artifact = record.get("artifact")
    validation = artifact.get("validation") if isinstance(artifact, dict) else None
    recovery = validation.get("validation_only_recovery") if isinstance(validation, dict) else None
    if record.get("status") == "succeeded" and isinstance(recovery, dict):
        if (
            recovery.get("schema_version") != VALIDATION_RECOVERY_SCHEMA_VERSION
            or recovery.get("artifact_sha256") != artifact.get("sha256")
            or recovery.get("artifact_sha256") != allowlisted_artifact_sha256
        ):
            return False, "recovery metadata does not match the artifact", None
        archive_value = recovery.get("previous_run_record_archive")
        archive_sha256 = recovery.get("previous_run_record_sha256")
        if not isinstance(archive_value, str) or not isinstance(archive_sha256, str) or len(archive_sha256) != 64:
            return False, "recovery metadata has no valid archival record", None
        archive_path = Path(archive_value)
        if not archive_path.is_file() or sha256_file(archive_path) != archive_sha256:
            return False, "archival run record is missing or changed", None
        valid, reason = validate_successful_case(case)
        if valid:
            return True, "synchronized an already recovered run record", recovery
        return False, f"recovered run record is no longer valid: {reason}", None

    if record.get("status") != "failed":
        return (
            False,
            f"exit-74 run record status is {record.get('status')!r}",
            None,
        )
    if record.get("run_id") != attempt.get("run_id"):
        return False, "run id does not match the latest attempt", None
    benchmark = record.get("benchmark")
    if not isinstance(benchmark, dict) or benchmark.get("case_id") != case.get("case_id"):
        return False, "case id does not match the run record", None
    exit_record = record.get("exit")
    if not isinstance(exit_record, dict):
        return False, "run record has no exit details", None
    if exit_record.get("child_exit_code") != 0 or exit_record.get("wrapper_exit_code") != 0 or exit_record.get("error"):
        return False, "inference or its wrapper did not complete cleanly", None
    if not isinstance(artifact, dict) or artifact.get("valid") is not False:
        return False, "run was not an artifact-validation failure", None
    if not isinstance(validation, dict):
        return False, "run has no artifact-validation details", None
    old_validation_errors = validation.get("errors")
    record_errors = record.get("errors")
    if (
        not isinstance(old_validation_errors, list)
        or not old_validation_errors
        or not all(isinstance(error, str) and error for error in old_validation_errors)
        or not isinstance(record_errors, list)
        or set(record_errors) != set(old_validation_errors)
    ):
        return False, "run contains errors beyond artifact validation", None
    if not all("ffmpeg" in error and "error while loading shared libraries" in error and "libfreetype.so.6" in error for error in old_validation_errors):
        return (
            False,
            "validation failure is not the known unusable-ffmpeg error",
            None,
        )

    artifact_value = artifact.get("path")
    if not isinstance(artifact_value, str) or not artifact_value:
        return False, "artifact path is missing", None
    artifact_path = Path(artifact_value)
    if not artifact_path.is_file():
        return False, f"artifact is missing: {artifact_path}", None
    original_size = artifact.get("size_bytes")
    original_sha256 = artifact.get("sha256")
    if not isinstance(original_size, int) or original_size <= 0:
        return False, "original artifact size is invalid", None
    if not isinstance(original_sha256, str) or len(original_sha256) != 64:
        return False, "original artifact SHA-256 is invalid", None
    if original_sha256 != allowlisted_artifact_sha256:
        return False, "artifact SHA-256 is not the reviewed allowlisted digest", None
    try:
        if artifact_path.stat().st_size != original_size:
            return False, "artifact size changed after inference", None
        if sha256_file(artifact_path) != original_sha256:
            return False, "artifact SHA-256 changed after inference", None
    except OSError as exc:
        return False, f"artifact cannot be read: {exc}", None

    result_format = artifact.get("format")
    target = benchmark.get("target")
    if not isinstance(result_format, str) or not isinstance(target, dict):
        return False, "artifact format or benchmark target is invalid", None
    validator = load_run_record_module()
    checked_at_utc = utc_now()
    new_artifact = validator.validate_artifact(
        artifact_path,
        result_format,
        target,
        expected_audio=benchmark.get("task") in {"s2v", "ltx2_s2v"},
    )
    if new_artifact.get("valid") is not True:
        errors = new_artifact.get("validation", {}).get("errors", [])
        return (
            False,
            "strict artifact revalidation failed: " + ("; ".join(map(str, errors)) or "unknown validation error"),
            None,
        )
    try:
        current_sha256 = sha256_file(artifact_path)
    except OSError as exc:
        return False, f"artifact cannot be re-read: {exc}", None
    if new_artifact.get("size_bytes") != original_size or new_artifact.get("sha256") != original_sha256 or current_sha256 != original_sha256:
        return False, "artifact changed during strict revalidation", None

    recovery = {
        "schema_version": VALIDATION_RECOVERY_SCHEMA_VERSION,
        "recovered_at_utc": checked_at_utc,
        "reason": "unchanged artifact passed the corrected strict validator",
        "previous_attempt_exit_code": 74,
        "previous_validation_method": validation.get("method"),
        "previous_validation_errors": old_validation_errors,
        "artifact_sha256": original_sha256,
        "validator_sha256": sha256_file(RUN_RECORD_TOOL),
        "previous_run_record_sha256": original_record_sha256,
        "previous_run_record_archive": str(record_path.with_name(VALIDATION_RECOVERY_ARCHIVE_NAME)),
    }
    new_validation = new_artifact.get("validation")
    if not isinstance(new_validation, dict):
        return False, "corrected validator returned invalid details", None
    new_validation["validation_only_recovery"] = recovery
    record["artifact"] = new_artifact
    record["status"] = "succeeded"
    record["errors"] = []
    exit_record["error"] = ""
    archive_path = record_path.with_name(VALIDATION_RECOVERY_ARCHIVE_NAME)
    try:
        if record_path.read_bytes() != original_record_bytes:
            return False, "run record changed during strict revalidation", None
        if archive_path.exists():
            if not archive_path.is_file() or sha256_file(archive_path) != original_record_sha256:
                return False, "existing archival run record does not match", None
        else:
            atomic_write_bytes(archive_path, original_record_bytes)
        if sha256_file(archive_path) != original_record_sha256:
            return False, "archival run record verification failed", None
    except OSError as exc:
        return False, f"cannot preserve archival run record: {exc}", None
    atomic_write_json(record_path, record)
    return True, "unchanged artifact passed strict revalidation", recovery


def validate_successful_case(
    case: dict[str, Any],
    *,
    require_dit_profile: bool = False,
) -> tuple[bool, str]:
    """Revalidate the latest recorded artifact before a resume skips a case."""
    attempts = case.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return False, "missing attempt history"
    attempt = attempts[-1]
    if not isinstance(attempt, dict):
        return False, "invalid latest attempt"
    record_value = attempt.get("run_record")
    if not isinstance(record_value, str) or not record_value:
        return False, "missing run record path"
    record_path = Path(record_value)
    if not record_path.is_file():
        return False, f"run record is missing: {record_path}"
    try:
        record = load_json(record_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"run record is invalid: {exc}"
    if record.get("status") != "succeeded":
        return False, f"run record status is {record.get('status')!r}"
    if record.get("run_id") != attempt.get("run_id"):
        return False, "run id does not match the latest attempt"
    benchmark = record.get("benchmark")
    if not isinstance(benchmark, dict) or benchmark.get("case_id") != case.get("case_id"):
        return False, "case id does not match the run record"
    artifact = record.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("valid") is not True:
        return False, "artifact was not recorded as valid"
    validation = artifact.get("validation")
    if artifact.get("enabled") is False:
        if not isinstance(validation, dict) or validation.get("checked") is not False:
            return False, "disabled artifact record is malformed"
        if not require_dit_profile:
            return True, "ok"
        metrics = record.get("metrics")
        profile = metrics.get("profile_seconds") if isinstance(metrics, dict) else None
        pipeline = profile.get("pipeline") if isinstance(profile, dict) else None
        steady_dit = metrics.get("dit_steady_state_seconds") if isinstance(metrics, dict) else None
        dit_step_profile = metrics.get("dit_step_profile") if isinstance(metrics, dict) else None
        if not isinstance(pipeline, (int, float)) or isinstance(pipeline, bool) or pipeline <= 0:
            return False, "run record has no positive pipeline cost"
        if not isinstance(steady_dit, (int, float)) or isinstance(steady_dit, bool) or steady_dit <= 0:
            return False, "run record has no positive steady-state DiT metric"
        if not isinstance(dit_step_profile, dict) or dit_step_profile.get("authoritative") is not True:
            return False, "run record DiT step profile is not authoritative"
        return True, "ok"
    recovery = validation.get("validation_only_recovery") if isinstance(validation, dict) else None
    if isinstance(recovery, dict):
        recovery_key = (case.get("case_id"), attempt.get("run_id"))
        expected_artifact_sha256 = VALIDATION_ONLY_RECOVERY_ALLOWLIST.get(recovery_key)
        if (
            expected_artifact_sha256 is None
            or recovery.get("schema_version") != VALIDATION_RECOVERY_SCHEMA_VERSION
            or recovery.get("artifact_sha256") != expected_artifact_sha256
            or artifact.get("sha256") != expected_artifact_sha256
        ):
            return False, "validation-only recovery is not allowlisted"
        archive_value = recovery.get("previous_run_record_archive")
        archive_sha256 = recovery.get("previous_run_record_sha256")
        if not isinstance(archive_value, str) or not isinstance(archive_sha256, str) or len(archive_sha256) != 64:
            return False, "validation-only recovery archive is invalid"
        archive_path = Path(archive_value)
        try:
            if not archive_path.is_file() or sha256_file(archive_path) != archive_sha256:
                return False, "validation-only recovery archive changed"
        except OSError as exc:
            return False, f"validation-only recovery archive cannot be read: {exc}"
    artifact_value = artifact.get("path")
    if not isinstance(artifact_value, str) or not artifact_value:
        return False, "artifact path is missing"
    artifact_path = Path(artifact_value)
    if not artifact_path.is_file():
        return False, f"artifact is missing: {artifact_path}"
    try:
        expected_size = artifact.get("size_bytes")
        if not isinstance(expected_size, int) or artifact_path.stat().st_size != expected_size:
            return False, "artifact size does not match the run record"
        expected_sha256 = artifact.get("sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or sha256_file(artifact_path) != expected_sha256:
            return False, "artifact SHA-256 does not match the run record"
    except OSError as exc:
        return False, f"artifact cannot be read: {exc}"
    if not require_dit_profile:
        return True, "ok"

    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return False, "run record metrics are not an object"
    dit_seconds_per_step = metrics.get("dit_seconds_per_step")
    if (
        not isinstance(dit_seconds_per_step, (int, float))
        or isinstance(dit_seconds_per_step, bool)
        or dit_seconds_per_step <= 0
    ):
        return False, "run record has no positive synchronized DiT/step metric"
    dit_step_profile = metrics.get("dit_step_profile")
    if (
        not isinstance(dit_step_profile, dict)
        or dit_step_profile.get("authoritative") is not True
    ):
        return False, "run record DiT step profile is not authoritative"
    return True, "ok"


def suite_controller_is_running(pid_value: object, suite_id: str) -> bool:
    if not isinstance(pid_value, int) or pid_value <= 1 or pid_value == os.getpid():
        return False
    try:
        command = Path(f"/proc/{pid_value}/cmdline").read_bytes()
    except OSError:
        return False
    return b"scripts/metax/run_infer_suite.py" in command and suite_id.encode() in command


def validate_suite_id(suite_id: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", suite_id) is None:
        raise ValueError(f"invalid suite id: {suite_id}")


def handle_signal(signum: int, _frame: object) -> None:
    global INTERRUPTED_SIGNAL
    INTERRUPTED_SIGNAL = signum
    process = ACTIVE_PROCESS
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _progress_snapshot(
    run_log_path: Path,
    run_record_path: Path,
    result_dir: Path,
) -> tuple[tuple[str, int, int], ...]:
    """Return the watched files' path, size and nanosecond mtime."""
    files: list[tuple[str, Path]] = [
        ("run.log", run_log_path),
        ("run.json", run_record_path),
    ]
    try:
        result_files = sorted(path for path in result_dir.rglob("*") if path.is_file())
    except OSError:
        result_files = []
    for path in result_files:
        try:
            relative_path = path.relative_to(result_dir)
        except ValueError:
            relative_path = path
        files.append((f"result/{relative_path}", path))

    snapshot: list[tuple[str, int, int]] = []
    for label, path in files:
        try:
            stat_result = path.stat()
        except OSError:
            continue
        if path.is_file():
            snapshot.append((label, stat_result.st_size, stat_result.st_mtime_ns))
    return tuple(snapshot)


def _business_progress_snapshot(
    snapshot: tuple[tuple[str, int, int], ...],
) -> tuple[tuple[str, int, int], ...]:
    """Exclude run.log, whose repeating driver errors are not progress."""
    return tuple(item for item in snapshot if item[0] != "run.log")


def _read_appended_log_lines(
    path: Path,
    offset: int,
    partial_line: bytes,
) -> tuple[list[bytes], int, bytes]:
    """Read complete newly appended lines, tolerating truncation and races."""
    try:
        current_size = path.stat().st_size
        if current_size < offset:
            offset = 0
            partial_line = b""
        with path.open("rb") as file_obj:
            file_obj.seek(offset)
            payload = file_obj.read()
            new_offset = file_obj.tell()
    except OSError:
        return [], offset, partial_line

    chunks = (partial_line + payload).split(b"\n")
    if payload or partial_line:
        partial_line = chunks.pop()
    return chunks, new_offset, partial_line


def _is_metax_queue_retry_line(line: bytes) -> bool:
    return METAX_QUEUE_RETRY_MARKER in line and METAX_QUEUE_RETRY_TYPE_MARKER in line and METAX_QUEUE_RETRY_ACTION_MARKER in line


def _parse_proc_stat(
    stat_text: str,
    *,
    expected_pid: int | None = None,
) -> _ProcStat | None:
    """Parse PPID/PGRP/starttime without splitting a spaced `comm` field."""
    command_start = stat_text.find("(")
    command_end = stat_text.rfind(")")
    if command_start <= 0 or command_end <= command_start:
        return None
    try:
        pid = int(stat_text[:command_start].strip())
    except ValueError:
        return None
    if expected_pid is not None and pid != expected_pid:
        return None

    # fields_after_command[0] is stat field 3 (state), so PPID/PGRP are
    # indices 1/2 and starttime (field 22) is index 19.
    fields_after_command = stat_text[command_end + 1 :].split()
    if len(fields_after_command) < 20:
        return None
    try:
        state = fields_after_command[0]
        parent_pid = int(fields_after_command[1])
        process_group_id = int(fields_after_command[2])
        start_time_ticks = int(fields_after_command[19])
    except ValueError:
        return None
    if pid <= 0 or len(state) != 1 or parent_pid < 0 or process_group_id <= 0 or start_time_ticks < 0:
        return None
    return _ProcStat(
        identity=_ProcessIdentity(pid=pid, start_time_ticks=start_time_ticks),
        state=state,
        parent_pid=parent_pid,
        process_group_id=process_group_id,
    )


def _proc_stat_for_pid(
    pid: int,
    proc_path: Path = Path("/proc"),
) -> _ProcStat | None:
    try:
        stat_text = (proc_path / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    return _parse_proc_stat(stat_text, expected_pid=pid)


def _proc_stat_snapshot(
    proc_path: Path = Path("/proc"),
) -> dict[int, _ProcStat]:
    snapshot: dict[int, _ProcStat] = {}
    try:
        process_entries = tuple(proc_path.iterdir())
    except OSError:
        return snapshot
    for process_entry in process_entries:
        if not process_entry.name.isdigit():
            continue
        pid = int(process_entry.name)
        process_stat = _proc_stat_for_pid(pid, proc_path)
        if process_stat is not None:
            snapshot[pid] = process_stat
    return snapshot


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_group_member_pids(
    process_group_id: int,
    proc_path: Path = Path("/proc"),
) -> set[int]:
    """Return the live PIDs currently belonging to one Linux process group."""
    return {pid for pid, process_stat in _proc_stat_snapshot(proc_path).items() if process_stat.process_group_id == process_group_id}


def _descendant_processes(
    root_identity: _ProcessIdentity | None,
    known_processes: dict[_ProcessIdentity, int],
    proc_path: Path = Path("/proc"),
) -> dict[_ProcessIdentity, int]:
    """Return the live identity-checked process tree, including setsid workers.

    Previously observed live descendants remain roots after reparenting. Their
    PID must still have the same `/proc` starttime, so a recycled numeric PID
    can never pull an unrelated process tree into the case.
    """
    snapshot = _proc_stat_snapshot(proc_path)
    active: dict[_ProcessIdentity, int] = {}
    if root_identity is not None:
        root_stat = snapshot.get(root_identity.pid)
        if root_stat is not None and root_stat.identity == root_identity and root_stat.state != "Z":
            active[root_identity] = 0
    for identity, depth in known_processes.items():
        current = snapshot.get(identity.pid)
        if current is not None and current.identity == identity and current.state != "Z":
            active[identity] = max(active.get(identity, 0), depth)

    children_by_parent: dict[int, list[_ProcStat]] = {}
    for process_stat in snapshot.values():
        if process_stat.state == "Z":
            continue
        children_by_parent.setdefault(process_stat.parent_pid, []).append(process_stat)

    pending = list(active)
    while pending:
        parent_identity = pending.pop()
        parent_depth = active[parent_identity]
        for child_stat in children_by_parent.get(parent_identity.pid, []):
            child_identity = child_stat.identity
            child_depth = parent_depth + 1
            previous_depth = active.get(child_identity)
            if previous_depth is not None and previous_depth >= child_depth:
                continue
            active[child_identity] = child_depth
            pending.append(child_identity)
    return active


def _process_identity_payload(
    processes: dict[_ProcessIdentity, int],
) -> list[dict[str, int]]:
    return [
        {
            "pid": identity.pid,
            "start_time_ticks": identity.start_time_ticks,
            "depth": depth,
        }
        for identity, depth in sorted(
            processes.items(),
            key=lambda item: (item[1], item[0].pid, item[0].start_time_ticks),
        )
    ]


def _ordered_process_identity_payload(
    identities: list[_ProcessIdentity],
    processes: dict[_ProcessIdentity, int],
) -> list[dict[str, int]]:
    return [
        {
            "pid": identity.pid,
            "start_time_ticks": identity.start_time_ticks,
            "depth": processes[identity],
        }
        for identity in identities
    ]


def _signal_matching_processes(
    processes: dict[_ProcessIdentity, int],
    signum: int,
    *,
    proc_path: Path = Path("/proc"),
    signal_process: Callable[[int, int], None] = os.kill,
    exclude_pids: set[int] | None = None,
) -> tuple[list[_ProcessIdentity], list[str]]:
    """Signal only still-matching PID incarnations, deepest descendants first."""
    excluded = exclude_pids or set()
    signaled: list[_ProcessIdentity] = []
    errors: list[str] = []
    ordered = sorted(
        processes.items(),
        key=lambda item: (-item[1], -item[0].pid, -item[0].start_time_ticks),
    )
    for identity, _depth in ordered:
        if identity.pid in excluded:
            continue
        current = _proc_stat_for_pid(identity.pid, proc_path)
        if current is None or current.identity != identity or current.state == "Z":
            continue
        try:
            signal_process(identity.pid, signum)
            signaled.append(identity)
        except ProcessLookupError:
            continue
        except OSError as exc:
            errors.append(f"cannot signal PID {identity.pid} with {signum}: {exc}")
    return signaled, errors


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float,
    root_identity: _ProcessIdentity | None = None,
    known_processes: dict[_ProcessIdentity, int] | None = None,
    proc_path: Path = Path("/proc"),
) -> dict[str, Any]:
    """Terminate the leader PGID, then identity-checked setsid descendants."""
    process_group_id = process.pid
    tracked_processes = dict(known_processes or {})
    tracked_processes.update(
        _descendant_processes(
            root_identity,
            tracked_processes,
            proc_path,
        )
    )
    sigterm_sent = False
    sigkill_sent = False
    signal_errors: list[str] = []
    try:
        os.killpg(process_group_id, signal.SIGTERM)
        sigterm_sent = True
    except ProcessLookupError:
        pass
    descendant_sigterm, errors = _signal_matching_processes(
        tracked_processes,
        signal.SIGTERM,
        proc_path=proc_path,
        exclude_pids={process_group_id},
    )
    signal_errors.extend(errors)

    deadline = time.monotonic() + max(0.0, grace_seconds)
    active_processes = _descendant_processes(
        root_identity,
        tracked_processes,
        proc_path,
    )
    while _process_group_exists(process_group_id) or active_processes:
        # Reap a leader that honored SIGTERM; otherwise its zombie keeps the
        # process group observable until the full grace period expires.
        process.poll()
        active_processes = _descendant_processes(
            root_identity,
            tracked_processes,
            proc_path,
        )
        tracked_processes.update(active_processes)
        if not _process_group_exists(process_group_id) and not active_processes:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(WATCHDOG_TERMINATION_POLL_SECONDS, remaining))

    if _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
            sigkill_sent = True
        except ProcessLookupError:
            pass
    active_processes = _descendant_processes(
        root_identity,
        tracked_processes,
        proc_path,
    )
    tracked_processes.update(active_processes)
    descendant_sigkill, errors = _signal_matching_processes(
        active_processes,
        signal.SIGKILL,
        proc_path=proc_path,
        exclude_pids={process_group_id},
    )
    signal_errors.extend(errors)
    if descendant_sigkill:
        sigkill_sent = True

    child_return_code = process.wait()
    if sigkill_sent:
        # A killed descendant can remain briefly as an orphaned zombie until
        # init reaps it.  Wait before deciding that the process group survived;
        # otherwise the suite could refuse an otherwise safe shm cleanup.
        reap_deadline = time.monotonic() + max(
            min(grace_seconds, WATCHDOG_TERMINATION_GRACE_SECONDS),
            1.0,
        )
        active_processes = _descendant_processes(
            root_identity,
            tracked_processes,
            proc_path,
        )
        while _process_group_exists(process_group_id) or active_processes:
            remaining = reap_deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(WATCHDOG_TERMINATION_POLL_SECONDS, remaining))
            active_processes = _descendant_processes(
                root_identity,
                tracked_processes,
                proc_path,
            )
    surviving_processes = _descendant_processes(
        root_identity,
        tracked_processes,
        proc_path,
    )
    return {
        "sigterm_sent": sigterm_sent,
        "sigkill_sent": sigkill_sent,
        "child_return_code": child_return_code,
        "process_group_alive_after_cleanup": _process_group_exists(process_group_id),
        "descendant_sigterm_processes": _ordered_process_identity_payload(
            descendant_sigterm,
            tracked_processes,
        ),
        "descendant_sigkill_processes": _ordered_process_identity_payload(
            descendant_sigkill,
            tracked_processes,
        ),
        "owned_processes_seen_during_cleanup": _process_identity_payload(tracked_processes),
        "surviving_owned_processes": _process_identity_payload(surviving_processes),
        "signal_errors": signal_errors,
    }


def wait_for_case_process(
    process: subprocess.Popen[bytes],
    *,
    run_log_path: Path,
    run_record_path: Path,
    result_dir: Path,
    stall_timeout_seconds: float,
    queue_retry_timeout_seconds: float = DEFAULT_QUEUE_RETRY_TIMEOUT_SECONDS,
    queue_retry_min_count: int = DEFAULT_QUEUE_RETRY_MIN_COUNT,
    poll_interval_seconds: float = WATCHDOG_POLL_INTERVAL_SECONDS,
    termination_grace_seconds: float = WATCHDOG_TERMINATION_GRACE_SECONDS,
    on_timeout: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Wait for a case, terminating its identity-checked process tree on hangs."""
    if stall_timeout_seconds < 0:
        raise ValueError("stall timeout must be non-negative")
    if queue_retry_timeout_seconds < 0:
        raise ValueError("queue-retry timeout must be non-negative")
    if queue_retry_min_count <= 0:
        raise ValueError("queue-retry minimum count must be positive")
    if poll_interval_seconds <= 0 or poll_interval_seconds > 30:
        raise ValueError("watchdog poll interval must be in (0, 30] seconds")
    if termination_grace_seconds < 0 or termination_grace_seconds > 30:
        raise ValueError("termination grace must be in [0, 30] seconds")

    stall_enabled = stall_timeout_seconds > 0
    queue_retry_enabled = queue_retry_timeout_seconds > 0
    previous_snapshot = _progress_snapshot(
        run_log_path,
        run_record_path,
        result_dir,
    )
    previous_business_snapshot = _business_progress_snapshot(previous_snapshot)
    last_progress_monotonic = time.monotonic()
    last_progress_at_utc = utc_now()
    last_business_progress_at_utc = last_progress_at_utc
    process_group_pids_seen = {process.pid}
    root_stat = _proc_stat_for_pid(process.pid)
    root_identity = root_stat.identity if root_stat is not None else None
    process_tree_seen: dict[_ProcessIdentity, int] = {}
    if root_identity is not None:
        process_tree_seen[root_identity] = 0
    queue_retry_watch = _QueueRetryWatch()
    if queue_retry_watch.consume_log(
        run_log_path,
        now_monotonic=last_progress_monotonic,
    ):
        last_business_progress_at_utc = utc_now()

    def sample_processes() -> dict[_ProcessIdentity, int]:
        process_group_pids_seen.update(_process_group_member_pids(process.pid))
        active_processes = _descendant_processes(
            root_identity,
            process_tree_seen,
        )
        for identity, depth in active_processes.items():
            process_tree_seen[identity] = max(
                process_tree_seen.get(identity, depth),
                depth,
            )
        return active_processes

    def status_evidence(
        *,
        active_processes: dict[_ProcessIdentity, int],
    ) -> dict[str, Any]:
        return {
            "last_progress_at_utc": last_progress_at_utc,
            "last_business_progress_at_utc": last_business_progress_at_utc,
            "observed_file_count": len(previous_snapshot),
            "process_group_pids_seen": sorted(process_group_pids_seen),
            "process_tree_pids_seen": sorted({identity.pid for identity in process_tree_seen}),
            "process_tree_processes_seen": _process_identity_payload(process_tree_seen),
            "active_owned_processes_after_exit": _process_identity_payload(active_processes),
            "queue_retry": {
                "enabled": queue_retry_enabled,
                "timeout_seconds": queue_retry_timeout_seconds,
                "minimum_count": queue_retry_min_count,
                "incident_retry_count": queue_retry_watch.retry_count,
                "total_retry_count": queue_retry_watch.total_retry_count,
            },
        }

    while True:
        active_processes = sample_processes()
        return_code = process.poll()
        if return_code is not None:
            active_processes = sample_processes()
            return return_code, {
                "timed_out": False,
                **status_evidence(active_processes=active_processes),
            }

        wait_seconds = poll_interval_seconds
        now_monotonic = time.monotonic()
        if stall_enabled:
            remaining = stall_timeout_seconds - (time.monotonic() - last_progress_monotonic)
            wait_seconds = min(wait_seconds, max(0.001, remaining))
        if queue_retry_enabled and queue_retry_watch.incident_is_armed(queue_retry_min_count):
            queue_remaining = queue_retry_timeout_seconds - (now_monotonic - queue_retry_watch.first_retry_monotonic)
            wait_seconds = min(wait_seconds, max(0.001, queue_remaining))
        # Sample the process tree frequently enough to catch short-lived
        # workers for MCCL ownership evidence.
        wait_seconds = min(wait_seconds, 1.0)
        try:
            return_code = process.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            pass
        else:
            active_processes = sample_processes()
            return return_code, {
                "timed_out": False,
                **status_evidence(active_processes=active_processes),
            }

        sample_processes()
        now_monotonic = time.monotonic()
        business_log_progress = queue_retry_watch.consume_log(
            run_log_path,
            now_monotonic=now_monotonic,
        )
        current_snapshot = _progress_snapshot(
            run_log_path,
            run_record_path,
            result_dir,
        )
        current_business_snapshot = _business_progress_snapshot(current_snapshot)
        if business_log_progress:
            last_business_progress_at_utc = utc_now()
        if current_business_snapshot != previous_business_snapshot:
            queue_retry_watch.reset_incident()
            last_business_progress_at_utc = utc_now()
        previous_business_snapshot = current_business_snapshot
        if current_snapshot != previous_snapshot:
            previous_snapshot = current_snapshot
            last_progress_monotonic = now_monotonic
            last_progress_at_utc = utc_now()

        stalled_seconds = now_monotonic - last_progress_monotonic
        queue_retry_hang_seconds = 0.0
        if queue_retry_watch.first_retry_monotonic is not None:
            queue_retry_hang_seconds = now_monotonic - queue_retry_watch.first_retry_monotonic
        timeout_reason: str | None = None
        if queue_retry_enabled and queue_retry_watch.incident_is_armed(queue_retry_min_count) and queue_retry_hang_seconds >= queue_retry_timeout_seconds:
            timeout_reason = "metax_queue_retry_hang"
        elif stall_enabled and stalled_seconds >= stall_timeout_seconds:
            timeout_reason = "file_inactivity"
        if timeout_reason is None:
            continue
        if INTERRUPTED_SIGNAL is not None:
            # The controller's existing interrupt contract is to forward
            # SIGTERM and report 128 + the user's signal after the child exits.
            # Do not relabel that path as a watchdog timeout.
            continue

        # Check once more immediately before signaling so a child that exited
        # on the timeout boundary is never mislabeled as stalled.
        return_code = process.poll()
        if return_code is not None:
            active_processes = sample_processes()
            return return_code, {
                "timed_out": False,
                **status_evidence(active_processes=active_processes),
            }

        timeout_status: dict[str, Any] = {
            "timed_out": True,
            "timeout_reason": timeout_reason,
            "timed_out_at_utc": utc_now(),
            "stalled_seconds": round(stalled_seconds, 3),
            "queue_retry_hang_seconds": round(queue_retry_hang_seconds, 3),
            "timeout_exit_code": 124,
            **status_evidence(active_processes=sample_processes()),
        }
        try:
            if on_timeout is not None:
                on_timeout(dict(timeout_status))
        finally:
            cleanup_status = _terminate_process_group(
                process,
                grace_seconds=termination_grace_seconds,
                root_identity=root_identity,
                known_processes=process_tree_seen,
            )
            for item in cleanup_status["owned_processes_seen_during_cleanup"]:
                identity = _ProcessIdentity(
                    int(item["pid"]),
                    int(item["start_time_ticks"]),
                )
                process_tree_seen[identity] = int(item["depth"])
            timeout_status.update(cleanup_status)
            timeout_status["process_tree_pids_seen"] = sorted({identity.pid for identity in process_tree_seen})
            timeout_status["process_tree_processes_seen"] = _process_identity_payload(process_tree_seen)
        timeout_status["active_owned_processes_after_exit"] = timeout_status["surviving_owned_processes"]
        return 124, timeout_status


def select_cases(args: argparse.Namespace) -> list[tuple[str, str, str, int]]:
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
    return [
        (*case, repetition)
        for case in selected
        for repetition in range(1, args.repetitions + 1)
    ]


def build_initial_state(suite_id: str, selected: list[tuple[str, str, str, int]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite_id": suite_id,
        "platform": "metax_cuda",
        "device_model": "C500",
        "device_memory_gib": 64,
        "repo_path": str(REPO_PATH),
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "status": "running",
        "repetitions": max(repetition for _, _, _, repetition in selected),
        "pid": os.getpid(),
        "cases": [
            {
                "index": index,
                "group": group,
                "case_id": case_id,
                "repetition": repetition,
                "script": str(INFER_ROOT / relative_script),
                "status": "pending",
                "attempts": [],
            }
            for index, (group, case_id, relative_script, repetition) in enumerate(selected, start=1)
        ],
    }


def validate_case_files(state: dict[str, Any]) -> None:
    missing = [case["script"] for case in state["cases"] if not Path(case["script"]).is_file()]
    if missing:
        raise FileNotFoundError("missing suite script(s): " + ", ".join(missing))


def print_case_plan(state: dict[str, Any], *, resume: bool) -> None:
    cases = state["cases"]
    action_count = 0
    for case in cases:
        status = str(case.get("status", "pending"))
        action = "skip" if resume and status == "success" else "run"
        action_count += action == "run"
        print(
            f"{int(case['index']):02d} {case['group']:<7} {action:<4} status={status:<8} {case['case_id']}",
            flush=True,
        )
        print(f"   {case['script']}", flush=True)
    print(
        f"[Suite] selected={len(cases)}, would_run={action_count}, would_skip={len(cases) - action_count}",
        flush=True,
    )


def run_read_only(args: argparse.Namespace) -> int:
    """List or dry-run a selection without creating or updating suite state."""
    if args.resume:
        suite_id = args.resume
        validate_suite_id(suite_id)
        state_path = SUITE_ROOT / suite_id / "suite.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"resume state does not exist: {state_path}")
        state = load_json(state_path)
        validate_case_files(state)
        print(f"[Suite] read-only resume preview: {suite_id}", flush=True)
        print(f"[Suite] state={state_path}", flush=True)
        print_case_plan(state, resume=True)
    else:
        selected = select_cases(args)
        if not selected:
            raise ValueError("no cases selected")
        suite_id = args.suite_id or datetime.now(timezone.utc).strftime("metax_%Y%m%dT%H%M%SZ")
        validate_suite_id(suite_id)
        state = build_initial_state(suite_id, selected)
        validate_case_files(state)
        mode = "list" if args.list else "dry-run"
        print(f"[Suite] {mode}: scope={args.scope}", flush=True)
        print(f"[Suite] planned_id={suite_id}", flush=True)
        print(f"[Suite] planned_state={SUITE_ROOT / suite_id / 'suite.json'}", flush=True)
        print_case_plan(state, resume=False)
    print("[Suite] read-only mode: no process started and no state changed", flush=True)
    return 0


def run_suite(args: argparse.Namespace) -> int:
    if args.resume and (args.scope != "all" or args.only):
        raise ValueError("--scope and --only are not supported with --resume")

    if args.list or args.dry_run:
        return run_read_only(args)

    selected = select_cases(args)
    if not selected:
        raise ValueError("no cases selected")

    suite_id = args.resume or args.suite_id or datetime.now(timezone.utc).strftime("metax_%Y%m%dT%H%M%SZ")
    validate_suite_id(suite_id)
    suite_dir = SUITE_ROOT / suite_id
    with hold_controller_locks(SUITE_ROOT, suite_dir, suite_id):
        return run_locked_suite(args, selected, suite_id, suite_dir)


def run_locked_suite(
    args: argparse.Namespace,
    selected: list[tuple[str, str, str, int]],
    suite_id: str,
    suite_dir: Path,
) -> int:
    """Execute a suite while the caller holds both controller locks."""
    global ACTIVE_PROCESS

    state_path = suite_dir / "suite.json"
    last_gpu_release_monotonic = time.monotonic()

    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"resume state does not exist: {state_path}")
        state = load_json(state_path)
        if suite_controller_is_running(state.get("pid"), suite_id):
            raise RuntimeError(f"suite controller is still running with PID {state['pid']}")
        state["status"] = "running"
        state["finished_at_utc"] = None
        state["pid"] = os.getpid()
    else:
        suite_dir.mkdir(parents=True, exist_ok=True)
        if state_path.exists():
            raise FileExistsError(f"suite state already exists: {state_path}")
        state = build_initial_state(suite_id, selected)
        validate_case_files(state)
        atomic_write_json(state_path, state)

    if args.resume:
        validate_case_files(state)
    state["gpu_release_cooldown_seconds"] = args.gpu_cooldown_seconds
    state["watchdog_defaults"] = {
        "stall_timeout_seconds": args.stall_timeout_seconds,
        "queue_retry_timeout_seconds": args.queue_retry_timeout_seconds,
        "queue_retry_min_count": args.queue_retry_min_count,
    }
    atomic_write_json(state_path, state)
    print(f"[Suite] id={suite_id}", flush=True)
    print(f"[Suite] state={state_path}", flush=True)

    for case in state["cases"]:
        if INTERRUPTED_SIGNAL is not None:
            break
        if args.resume and case.get("status") != "success":
            recovered, reason, recovery = recover_validation_only_failure(case)
            if recovered:
                case["status"] = "success"
                case["attempts"][-1]["validation_only_recovery"] = recovery
                atomic_write_json(state_path, state)
                profile_valid, profile_reason = validate_successful_case(
                    case,
                    require_dit_profile=True,
                )
                if profile_valid:
                    print(
                        f"[Suite] recovered validation-only case: {case['case_id']}: {reason}",
                        flush=True,
                    )
                    continue
                print(
                    f"[Suite] recovered artifact still needs a profiled rerun: "
                    f"{case['case_id']}: {profile_reason}",
                    flush=True,
                )
                case["status"] = "failed"
                atomic_write_json(state_path, state)
            if reason:
                print(
                    f"[Suite] validation-only case needs rerun: {case['case_id']}: {reason}",
                    flush=True,
                )
        if case["status"] == "success":
            valid, reason = validate_successful_case(
                case,
                require_dit_profile=True,
            )
            if valid:
                print(f"[Suite] skip successful case: {case['case_id']}", flush=True)
                continue
            print(
                f"[Suite] successful case needs rerun: {case['case_id']}: {reason}",
                flush=True,
            )
            case["status"] = "failed"
            atomic_write_json(state_path, state)

        group = str(case["group"])
        case_id = str(case["case_id"])
        cooldown_waited_seconds = 0.0
        mccl_shm_before: dict[str, tuple[int, int]] = {}
        if case_requires_gpu_release_cooldown(group):
            cooldown_waited_seconds = wait_for_gpu_release_cooldown(
                last_gpu_release_monotonic,
                args.gpu_cooldown_seconds,
            )
            if INTERRUPTED_SIGNAL is not None:
                break
            mccl_shm_before = snapshot_mccl_shm_files()

        attempt_number = len(case["attempts"]) + 1
        run_id = (
            f"{suite_id}_{int(case['index']):02d}_"
            f"r{int(case.get('repetition', 1))}_a{attempt_number}"
        )
        record_path = REPO_PATH / "logs" / "metax" / "infer" / group / case_id / run_id / "run.json"
        result_dir = REPO_PATH / "results" / "metax" / "infer" / group / case_id / run_id
        attempt = {
            "attempt": attempt_number,
            "run_id": run_id,
            "started_at_utc": utc_now(),
            "finished_at_utc": None,
            "elapsed_seconds": None,
            "exit_code": None,
            "run_record": str(record_path),
            "result_dir": str(result_dir),
            "gpu_release_cooldown": {
                "configured_seconds": args.gpu_cooldown_seconds,
                "waited_seconds": round(cooldown_waited_seconds, 3),
                "reason": ("avoid C500/MCCL first-collective SIGBUS after a previous GPU process exits"),
            },
            "watchdog": {
                "enabled": (args.stall_timeout_seconds > 0 or args.queue_retry_timeout_seconds > 0),
                "stall_timeout_seconds": args.stall_timeout_seconds,
                "queue_retry_timeout_seconds": args.queue_retry_timeout_seconds,
                "queue_retry_min_count": args.queue_retry_min_count,
                "poll_interval_seconds": WATCHDOG_POLL_INTERVAL_SECONDS,
                "termination_grace_seconds": WATCHDOG_TERMINATION_GRACE_SECONDS,
                "timed_out": False,
            },
        }
        case["attempts"].append(attempt)
        case["status"] = "running"
        atomic_write_json(state_path, state)

        env = os.environ.copy()
        env["RUN_ID"] = run_id
        env["MASTER_PORT"] = str(29600 + int(case["index"]))
        env["CUDA_VISIBLE_DEVICES"] = {
            "single": "0",
            "dist_2": "0,1",
            "dist_8": "0,1,2,3,4,5,6,7",
        }[group]
        env.pop("MLU_VISIBLE_DEVICES", None)
        env.pop("CN_VISIBLE_DEVICES", None)
        env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
        started = time.monotonic()
        print(
            f"\n[Suite] ({case['index']}/{len(state['cases'])}) start "
            f"{group}/{case_id}, repetition={case.get('repetition', 1)}, run_id={run_id}",
            flush=True,
        )
        process = subprocess.Popen(
            ["bash", str(case["script"])],
            cwd=REPO_PATH,
            env=env,
            start_new_session=True,
        )
        ACTIVE_PROCESS = process

        def record_watchdog_timeout(
            timeout_status: dict[str, Any],
        ) -> None:
            attempt["watchdog"].update(timeout_status)
            attempt["exit_code"] = 124
            atomic_write_json(state_path, state)
            if timeout_status["timeout_reason"] == "metax_queue_retry_hang":
                reason = (
                    f"MetaX type:21 queue creation retried {timeout_status['queue_retry']['incident_retry_count']} times without business progress for {timeout_status['queue_retry_hang_seconds']}s"
                )
            else:
                reason = f"no run.log/run.json/result progress for {timeout_status['stalled_seconds']}s"
            print(
                f"[Suite] watchdog timeout for {case_id}: {reason}; terminating owned process tree rooted at {process.pid}",
                flush=True,
            )

        try:
            exit_code, watchdog_status = wait_for_case_process(
                process,
                run_log_path=record_path.with_name("run.log"),
                run_record_path=record_path,
                result_dir=result_dir,
                stall_timeout_seconds=args.stall_timeout_seconds,
                queue_retry_timeout_seconds=args.queue_retry_timeout_seconds,
                queue_retry_min_count=args.queue_retry_min_count,
                on_timeout=record_watchdog_timeout,
            )
            attempt["watchdog"].update(watchdog_status)
        finally:
            ACTIVE_PROCESS = None
            last_gpu_release_monotonic = time.monotonic()

        if case_requires_gpu_release_cooldown(group):
            known_processes: dict[_ProcessIdentity, int] = {}
            for item in attempt["watchdog"].get(
                "process_tree_processes_seen",
                [],
            ):
                try:
                    identity = _ProcessIdentity(
                        pid=int(item["pid"]),
                        start_time_ticks=int(item["start_time_ticks"]),
                    )
                    known_processes[identity] = int(item["depth"])
                except (KeyError, TypeError, ValueError):
                    continue
            root_identity = next(
                (identity for identity in known_processes if identity.pid == process.pid),
                None,
            )
            active_owned_processes = _descendant_processes(
                root_identity,
                known_processes,
            )
            if _process_group_exists(process.pid) or active_owned_processes:
                cleanup_status = _terminate_process_group(
                    process,
                    grace_seconds=WATCHDOG_TERMINATION_GRACE_SECONDS,
                    root_identity=root_identity,
                    known_processes=known_processes,
                )
                attempt["post_exit_process_tree_cleanup"] = cleanup_status
                for item in cleanup_status["owned_processes_seen_during_cleanup"]:
                    identity = _ProcessIdentity(
                        pid=int(item["pid"]),
                        start_time_ticks=int(item["start_time_ticks"]),
                    )
                    known_processes[identity] = int(item["depth"])
            process_group_alive = _process_group_exists(process.pid)
            surviving_owned_processes = _descendant_processes(
                root_identity,
                known_processes,
            )
            if process_group_alive or surviving_owned_processes:
                attempt["mccl_shm_cleanup"] = {
                    "checked": False,
                    "created_file_count": 0,
                    "eligible_file_count": 0,
                    "quarantined_file_count": 0,
                    "quarantined_bytes": 0,
                    "files": [],
                    "skipped_unowned_files": [],
                    "skipped_reused_pid_files": [],
                    "errors": [
                        f"refusing MCCL quarantine while owned case processes remain alive: group_alive={process_group_alive}, processes={_process_identity_payload(surviving_owned_processes)}"
                    ],
                }
            else:
                owned_process_ids: dict[int, set[int]] = {}
                for identity in known_processes:
                    owned_process_ids.setdefault(identity.pid, set()).add(identity.start_time_ticks)
                attempt["mccl_shm_cleanup"] = quarantine_new_mccl_shm_files(
                    mccl_shm_before,
                    record_path.parent / "mccl_shm_quarantine",
                    owned_process_ids,
                )

        attempt["exit_code"] = exit_code
        attempt["finished_at_utc"] = utc_now()
        attempt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        record = None
        if record_path.is_file():
            try:
                record = load_json(record_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(
                    f"[Suite] warning: invalid run record for {case_id}: {exc}",
                    flush=True,
                )
        recorded_status = record.get("status") if record else None
        attempt_valid = False
        validation_reason = ""
        if exit_code == 0 and recorded_status == "succeeded":
            attempt_valid, validation_reason = validate_successful_case(
                case,
                require_dit_profile=True,
            )
        case["status"] = "success" if attempt_valid else "failed"
        if not attempt_valid:
            attempt["validation_error"] = (
                validation_reason
                or f"exit={exit_code}, run_record_status={recorded_status!r}"
            )
        print(
            f"[Suite] finish {case_id}: status={case['status']}, exit_code={exit_code}, elapsed={attempt['elapsed_seconds']}s",
            flush=True,
        )
        if not attempt_valid:
            print(
                f"[Suite] validation failed {case_id}: "
                f"{attempt['validation_error']}",
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
    state["benchmark_summary"] = generate_benchmark_summary(state, suite_dir)
    atomic_write_json(state_path, state)
    state["final_report"] = (
        {"status": "skipped", "reason": "no-save benchmark"}
        if args.skip_final_report
        else generate_final_report()
    )
    atomic_write_json(state_path, state)

    success_count = sum(case["status"] == "success" for case in state["cases"])
    failed_count = sum(case["status"] == "failed" for case in state["cases"])
    print(
        f"\n[Suite] status={state['status']}, success={success_count}, failed={failed_count}, total={len(state['cases'])}",
        flush=True,
    )
    if INTERRUPTED_SIGNAL is not None:
        return 128 + INTERRUPTED_SIGNAL
    return (
        0
        if state["status"] == "success"
        and state["final_report"]["status"] in {"succeeded", "skipped"}
        else 1
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("all", "single", "multi"), default="all")
    parser.add_argument("--only", action="append", help="run one named case; repeat as needed")
    suite_source = parser.add_mutually_exclusive_group()
    suite_source.add_argument("--suite-id", help="new suite identifier")
    suite_source.add_argument("--resume", help="resume an existing suite identifier")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--repetitions",
        type=int,
        choices=range(1, 11),
        default=1,
        metavar="N",
        help="run each selected entrypoint N consecutive times",
    )
    parser.add_argument(
        "--skip-final-report",
        action="store_true",
        help="skip the artifact-oriented historical aggregate report",
    )
    parser.add_argument(
        "--stall-timeout-seconds",
        type=int,
        default=DEFAULT_STALL_TIMEOUT_SECONDS,
        help=(f"terminate a case after this many seconds without run.log, run.json or result-file progress; 0 disables this inactivity detector (default: {DEFAULT_STALL_TIMEOUT_SECONDS})"),
    )
    parser.add_argument(
        "--queue-retry-timeout-seconds",
        type=float,
        default=DEFAULT_QUEUE_RETRY_TIMEOUT_SECONDS,
        help=(f"terminate after a sustained MetaX type:21 queue-retry incident without business progress; 0 disables this detector (default: {DEFAULT_QUEUE_RETRY_TIMEOUT_SECONDS:g})"),
    )
    parser.add_argument(
        "--queue-retry-min-count",
        type=int,
        default=DEFAULT_QUEUE_RETRY_MIN_COUNT,
        help=(f"minimum type:21 queue-retry lines before the early hang detector may fire (default: {DEFAULT_QUEUE_RETRY_MIN_COUNT})"),
    )
    parser.add_argument(
        "--gpu-cooldown-seconds",
        type=float,
        default=DEFAULT_GPU_RELEASE_COOLDOWN_SECONDS,
        help=(f"minimum C500 runtime-release interval before each distributed case; 0 disables it (default: {DEFAULT_GPU_RELEASE_COOLDOWN_SECONDS:g})"),
    )
    read_only = parser.add_mutually_exclusive_group()
    read_only.add_argument(
        "--list",
        action="store_true",
        help="list and validate selected entrypoints without changing state",
    )
    read_only.add_argument(
        "--dry-run",
        action="store_true",
        help="preview execution or resume without starting cases or changing state",
    )
    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        args = parse_args()
        if args.stall_timeout_seconds < 0:
            raise ValueError("--stall-timeout-seconds must be non-negative")
        if args.queue_retry_timeout_seconds < 0:
            raise ValueError("--queue-retry-timeout-seconds must be non-negative")
        if args.queue_retry_min_count <= 0:
            raise ValueError("--queue-retry-min-count must be positive")
        if args.gpu_cooldown_seconds < 0:
            raise ValueError("--gpu-cooldown-seconds must be non-negative")
        return run_suite(args)
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[Suite] fatal: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
