#!/usr/bin/env python3
"""MetaX C500 adapter for the shared LightX2V service-suite controller.

This module deliberately does not modify the shared Ascend/MLU controller.
It registers MetaX paths at runtime, adds mx-smi parsing and applies the
MetaX-specific lifecycle safeguards already used by offline inference.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_PATH = Path("/data/Lightx2v-Platform-Run-Examples")
SCRIPTS_PATH = REPO_PATH / "scripts"
SERVER_PATH = SCRIPTS_PATH / "metax" / "server"
LIGHTX2V_PATH = Path("/data/LightX2V-metax")
PLATFORM_NAME = "metax"
DISPLAY_NAME = "MetaX C500"
PROFILE_LEVEL = 0
GPU_COOLDOWN_SECONDS = 30.0
METAX_IDLE_MEMORY_LIMIT_MIB = 2048
PINNED_SHARED_FILES = {
    REPO_PATH / "bench_t2i_service.py": (
        "0cb59b1f51ee6f24fa16bcfbb518ba75593654e6ef2bf756a8b44be6ee9faf7d"
    ),
    REPO_PATH / "bench_t2v_service.py": (
        "e87dcbec116cf85ef92e1d6d6c6f1e795b6793f98bf1743c53919709665c4fd0"
    ),
    SCRIPTS_PATH / "service_benchmark_common.py": (
        "a0cdc4a3f5cf9d025247065601155d689103fa41bb1bf2eb54f202f2f492ed39"
    ),
}
FATAL_SERVER_LOG_PATTERNS = (
    b"torch.OutOfMemoryError:",
    b"CUDA out of memory.",
)

if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))
if str(SERVER_PATH) not in sys.path:
    sys.path.insert(0, str(SERVER_PATH))

import _service_suite_core as common  # noqa: E402
from preflight_infer import metax_shared_memory_errors  # noqa: E402


def _load_offline_helpers() -> Any:
    path = SCRIPTS_PATH / "metax" / "run_infer_suite.py"
    spec = importlib.util.spec_from_file_location(
        "_metax_offline_suite_helpers", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MetaX lifecycle helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


offline_helpers = _load_offline_helpers()

METAX_PATHS = common.PlatformPaths(
    name=PLATFORM_NAME,
    display_name=DISPLAY_NAME,
    repo_path=REPO_PATH,
    lightx2v_path=LIGHTX2V_PATH,
    log_root=REPO_PATH / "logs" / PLATFORM_NAME / "server",
    result_root=REPO_PATH / "results" / PLATFORM_NAME / "server",
    server_root=SERVER_PATH / "dist",
    infer_root=SCRIPTS_PATH / "metax" / "infer",
    config_root=REPO_PATH / "configs" / "metax",
    data_root=REPO_PATH / "data",
    monitor_command=("mx-smi",),
    memory_label="VRAM",
)
common.PLATFORMS[PLATFORM_NAME] = METAX_PATHS


def parse_mx_smi_snapshot(output: str) -> tuple[set[int], dict[int, int]]:
    """Parse PIDs and per-card VRAM MiB from mx-smi text output."""

    process_ids: set[int] = set()
    memory_used_mb: dict[int, int] = {}
    current_device: int | None = None
    saw_process_table = False
    saw_empty_process_table = False
    detailed_device_pattern = re.compile(r"^GPU#(\d+)\s+")
    summary_device_pattern = re.compile(
        r"^\|\s*(\d+)\s+MetaX\s+C500\s*\|"
    )
    detailed_memory_pattern = re.compile(
        r"^\s+vram used\s*:\s*([\d,]+)\s+KB\s*$"
    )
    summary_memory_pattern = re.compile(
        r"([\d,]+)\s*/\s*[\d,]+\s+MiB"
    )
    process_pattern = re.compile(
        r"^\|\s*\d+\s+(\d+)\s+.+?\s+[\d,]+"
        r"(?:\s+MiB)?\s*\|\s*$"
    )

    for line in output.splitlines():
        if "| Process:" in line:
            saw_process_table = True
        if "no process found" in line.lower():
            saw_empty_process_table = True
        device_match = (
            detailed_device_pattern.match(line)
            or summary_device_pattern.match(line)
        )
        if device_match:
            current_device = int(device_match.group(1))
            continue
        detailed_memory_match = detailed_memory_pattern.match(line)
        summary_memory_match = summary_memory_pattern.search(line)
        if detailed_memory_match and current_device is not None:
            used_kib = int(
                detailed_memory_match.group(1).replace(",", "")
            )
            memory_used_mb[current_device] = (used_kib + 1023) // 1024
            current_device = None
            continue
        if summary_memory_match and current_device is not None:
            memory_used_mb[current_device] = int(
                summary_memory_match.group(1).replace(",", "")
            )
            current_device = None
            continue
        process_match = process_pattern.match(line)
        if process_match:
            process_ids.add(int(process_match.group(1)))

    expected_devices = set(range(8))
    if set(memory_used_mb) != expected_devices:
        raise ValueError(
            "mx-smi device set is incomplete: "
            f"observed={sorted(memory_used_mb)}, expected={sorted(expected_devices)}"
        )
    if not saw_process_table:
        raise ValueError("unable to find process table in mx-smi")
    if not process_ids and not saw_empty_process_table:
        raise ValueError("unable to parse processes from non-empty mx-smi table")
    return process_ids, memory_used_mb


_shared_parse_device_snapshot = common.parse_device_snapshot


def parse_device_snapshot(
    output: str, platform_name: str
) -> tuple[set[int], dict[int, int]]:
    if platform_name == PLATFORM_NAME:
        return parse_mx_smi_snapshot(output)
    return _shared_parse_device_snapshot(output, platform_name)


common.parse_device_snapshot = parse_device_snapshot


def _process_start_times(process_ids: Sequence[int]) -> dict[int, set[int]]:
    guarded: dict[int, set[int]] = {}
    for process_id in process_ids:
        process_stat = offline_helpers._proc_stat_for_pid(process_id)
        if process_stat is not None:
            guarded.setdefault(process_id, set()).add(
                process_stat.identity.start_time_ticks
            )
    return guarded


def validate_pinned_shared_files() -> None:
    changed_shared_files = []
    for path, expected_sha256 in PINNED_SHARED_FILES.items():
        observed = common.sha256_file(path) if path.is_file() else "missing"
        if observed != expected_sha256:
            changed_shared_files.append(
                f"{path}: {observed} != {expected_sha256}"
            )
    if changed_shared_files:
        raise RuntimeError(
            "pinned shared client changed; re-review before MetaX run: "
            + "; ".join(changed_shared_files)
        )


class MetaXServiceSuite(common.ServiceSuite):
    """Shared controller plus MetaX-exclusive locks and MCCL hygiene."""

    def __init__(self, arguments: Any) -> None:
        if arguments.platform != PLATFORM_NAME:
            raise ValueError(
                f"MetaX wrapper only accepts --platform {PLATFORM_NAME}"
            )
        super().__init__(arguments)
        self.state["parameters"]["profiling_debug_level"] = PROFILE_LEVEL
        self.state["parameters"]["gpu_release_cooldown_seconds"] = (
            GPU_COOLDOWN_SECONDS
        )
        self.state["device"] = {
            "model": "MetaX C500",
            "count": 8,
            "memory_gib_per_device": 64,
        }
        self.metax_global_lock: Any = None
        # Match the offline controller: the first distributed case also gets
        # one full runtime-release interval before MCCL initialization.
        self.last_gpu_release_monotonic = time.monotonic()
        self.mccl_before: dict[str, tuple[int, int]] = {}
        self.mccl_case_log_dir: Path | None = None
        self.active_server_log_path: Path | None = None
        self.server_root_identity: Any = None
        self.server_known_processes: dict[Any, int] = {}

    def validate(self, cases: Sequence[Any]) -> None:
        super().validate(cases)
        validate_pinned_shared_files()
        expected = (
            (1, 10, 1)
            if self.suite_kind == "formal"
            else (1, 1, 1)
        )
        actual = (
            self.arguments.warmup_count,
            self.arguments.sample_count,
            self.arguments.concurrency,
        )
        if actual != expected:
            raise ValueError(
                f"{self.suite_kind} MetaX service contract is "
                f"warmup/sample/concurrency={expected}, got {actual}"
            )
        if os.environ.get("PROFILING_DEBUG_LEVEL") != "0":
            raise ValueError("MetaX service benchmark requires profiling level 0")

    def ensure_device_idle(self, snapshot_path: Path) -> dict[int, int]:
        process_ids, memory_used = self.snapshot_device(snapshot_path)
        if process_ids:
            raise RuntimeError(
                "MetaX is already in use; refusing to terminate unowned "
                f"processes: {sorted(process_ids)}"
            )
        excessive = {
            device: used
            for device, used in memory_used.items()
            if used > METAX_IDLE_MEMORY_LIMIT_MIB
        }
        if excessive:
            raise RuntimeError(
                "MetaX has no reported process but VRAM exceeds the idle "
                f"limit of {METAX_IDLE_MEMORY_LIMIT_MIB} MiB: {excessive}"
            )
        return memory_used

    def _acquire_global_lock(self) -> None:
        if self.metax_global_lock is not None:
            return
        infer_suite_root = (
            REPO_PATH / "logs" / "metax" / "infer" / "suites"
        )
        self.metax_global_lock = offline_helpers.acquire_metax_gpu_lock(
            infer_suite_root, f"server:{self.suite_id}"
        )
        try:
            offline_helpers.reject_active_legacy_controllers(
                infer_suite_root, Path("/nonexistent/metax-server-suite")
            )
        except BaseException:
            self.metax_global_lock.close()
            self.metax_global_lock = None
            raise

    def acquire_lock(self) -> None:
        self._acquire_global_lock()
        super().acquire_lock()

    def run(self) -> int:
        try:
            if not self.arguments.dry_run:
                # Acquire before the pinned core creates result directories.
                self._acquire_global_lock()
            return super().run()
        finally:
            if self.lock_handle is not None:
                self.lock_handle.close()
                self.lock_handle = None
            if self.metax_global_lock is not None:
                self.metax_global_lock.close()
                self.metax_global_lock = None

    def start_server(
        self,
        case: Any,
        case_log_dir: Path,
        case_result_dir: Path,
        attempt_number: int = 1,
    ) -> tuple[Any, float]:
        remaining = (
            self.last_gpu_release_monotonic
            + GPU_COOLDOWN_SECONDS
            - time.monotonic()
        )
        if remaining > 0:
            self.log(
                f"MetaX GPU/MCCL cooldown before {case.case_id}: "
                f"{remaining:.1f}s"
            )
            time.sleep(remaining)
        shared_memory_failures = metax_shared_memory_errors()
        if shared_memory_failures:
            raise RuntimeError("; ".join(shared_memory_failures))
        self.mccl_before = offline_helpers.snapshot_mccl_shm_files()
        self.mccl_case_log_dir = case_log_dir
        self.active_server_log_path = case_log_dir / "server.log"
        try:
            return super().start_server(
                case, case_log_dir, case_result_dir, attempt_number
            )
        finally:
            self._sample_server_tree()

    def prepare_server_environment(
        self,
        case: Any,
        environment: dict[str, str],
    ) -> dict[str, str]:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(device_id) for device_id in range(case.world_size)
        )
        environment["INFER_PROFILE_LEVEL"] = "0"
        environment["PROFILING_DEBUG_LEVEL"] = "0"
        environment.pop("ASCEND_RT_VISIBLE_DEVICES", None)
        environment.pop("MLU_VISIBLE_DEVICES", None)
        environment.pop("CN_VISIBLE_DEVICES", None)
        return environment

    def _queue_retry_timed_out(
        self,
        watch: Any,
        now_monotonic: float,
    ) -> bool:
        if self.active_server_log_path is None:
            return False
        watch.consume_log(
            self.active_server_log_path,
            now_monotonic=now_monotonic,
        )
        return (
            watch.incident_is_armed(12)
            and watch.first_retry_monotonic is not None
            and now_monotonic - watch.first_retry_monotonic >= 300.0
        )

    def _sample_server_tree(self) -> dict[Any, int]:
        process = self.active_process
        if process is not None and self.server_root_identity is None:
            process_stat = offline_helpers._proc_stat_for_pid(process.pid)
            if process_stat is not None:
                self.server_root_identity = process_stat.identity
        active = offline_helpers._descendant_processes(
            self.server_root_identity,
            self.server_known_processes,
        )
        self.server_known_processes.update(active)
        return active

    def wait_health(self, process: Any) -> float:
        started = time.perf_counter()
        deadline = started + self.arguments.startup_timeout_seconds
        url = f"http://127.0.0.1:{self.arguments.port}/health"
        last_error = ""
        watch = offline_helpers._QueueRetryWatch()
        while time.perf_counter() < deadline:
            self._sample_server_tree()
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"service exited before ready with code {exit_code}"
                )
            now = time.monotonic()
            if self._queue_retry_timed_out(watch, now):
                raise TimeoutError(
                    "MetaX type:21 queue retry persisted for 300 seconds "
                    "during service startup"
                )
            try:
                with common.NO_PROXY_OPENER.open(url, timeout=2.0) as response:
                    if response.getcode() == 200:
                        return time.perf_counter() - started
            except (OSError, common.URLError) as exception:
                last_error = str(exception)
            time.sleep(1)
        raise TimeoutError(
            "service health check timed out after "
            f"{self.arguments.startup_timeout_seconds}s: {last_error}"
        )

    def run_client(
        self,
        command: Sequence[str],
        log_path: Path,
    ) -> tuple[int, float]:
        # Re-check for every warm-up and measured client launch. The examples
        # checkout is shared with the MLU task and may change during a long
        # 12-case run.
        validate_pinned_shared_files()
        started = time.perf_counter()
        deadline = started + self.arguments.case_timeout_seconds
        watch = offline_helpers._QueueRetryWatch()
        server_log_offset = (
            self.active_server_log_path.stat().st_size
            if self.active_server_log_path is not None
            and self.active_server_log_path.is_file()
            else 0
        )
        with log_path.open("wb") as file_obj:
            process = subprocess.Popen(
                list(command),
                cwd=self.repo_path,
                stdout=file_obj,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.active_client_process = process
            exit_code = 124
            watchdog_reason = ""
            try:
                while process.poll() is None:
                    self._sample_server_tree()
                    now = time.monotonic()
                    if (
                        self.active_server_log_path is not None
                        and self.active_server_log_path.is_file()
                    ):
                        with self.active_server_log_path.open("rb") as log_obj:
                            log_obj.seek(server_log_offset)
                            appended = log_obj.read()
                            server_log_offset = log_obj.tell()
                        fatal_pattern = next(
                            (
                                pattern
                                for pattern in FATAL_SERVER_LOG_PATTERNS
                                if pattern in appended
                            ),
                            None,
                        )
                        if fatal_pattern is not None:
                            watchdog_reason = (
                                "fatal server log pattern detected: "
                                + fatal_pattern.decode(
                                    "utf-8", errors="replace"
                                )
                            )
                            self.stop_client()
                            break
                    if self._queue_retry_timed_out(watch, now):
                        watchdog_reason = (
                            "MetaX type:21 queue retry persisted for "
                            "300 seconds without business progress"
                        )
                        self.stop_client()
                        break
                    if time.perf_counter() >= deadline:
                        watchdog_reason = "client case timeout"
                        self.stop_client()
                        break
                    time.sleep(2)
                if process.poll() is not None:
                    exit_code = int(process.returncode)
            except KeyboardInterrupt:
                self.stop_client()
                raise
            finally:
                if self.active_client_process is process:
                    self.active_client_process = None
            if watchdog_reason:
                file_obj.write(
                    f"\n[MetaX watchdog] {watchdog_reason}\n".encode()
                )
        return exit_code, time.perf_counter() - started

    def benchmark_command(
        self,
        case: Any,
        *,
        phase: str,
        limit: int,
        run_dir: Path,
    ) -> list[str]:
        command = super().benchmark_command(
            case, phase=phase, limit=limit, run_dir=run_dir
        )
        if self.active_server_run_id and "--run-id" in command:
            index = command.index("--run-id") + 1
            command[index] = self.active_server_run_id.replace(":", "_")
        return command

    def stop_server(self) -> dict[str, Any]:
        run_id = self.active_server_run_id
        self._sample_server_tree()
        owned = _process_start_times(
            self.owned_server_pids(run_id) if run_id else []
        )
        for identity in self.server_known_processes:
            owned.setdefault(identity.pid, set()).add(
                identity.start_time_ticks
            )
        result = super().stop_server()
        identity_errors: list[str] = []
        identity_term: list[Any] = []
        identity_kill: list[Any] = []
        active = offline_helpers._descendant_processes(
            self.server_root_identity,
            self.server_known_processes,
        )
        self.server_known_processes.update(active)
        if active:
            identity_term, errors = offline_helpers._signal_matching_processes(
                active, signal.SIGTERM
            )
            identity_errors.extend(errors)
            deadline = time.monotonic() + 30.0
            while active and time.monotonic() < deadline:
                time.sleep(0.5)
                active = offline_helpers._descendant_processes(
                    self.server_root_identity,
                    self.server_known_processes,
                )
            if active:
                identity_kill, errors = (
                    offline_helpers._signal_matching_processes(
                        active, signal.SIGKILL
                    )
                )
                identity_errors.extend(errors)
                time.sleep(0.5)
                active = offline_helpers._descendant_processes(
                    self.server_root_identity,
                    self.server_known_processes,
                )
        result["identity_cleanup"] = {
            "term": offline_helpers._ordered_process_identity_payload(
                identity_term, self.server_known_processes
            ),
            "kill": offline_helpers._ordered_process_identity_payload(
                identity_kill, self.server_known_processes
            ),
            "residual": offline_helpers._process_identity_payload(active),
            "errors": identity_errors,
        }
        # The shared cleanup may have observed PIDs before the guarded
        # descendant cleanup completed. Recompute rather than retaining stale
        # residuals that would incorrectly block safe MCCL quarantine.
        exact_residuals = (
            set(self.owned_server_pids(run_id)) if run_id else set()
        )
        result["residual_pids"] = sorted(
            exact_residuals | {identity.pid for identity in active}
        )
        if not run_id:
            self.mccl_before = {}
            self.mccl_case_log_dir = None
            self.active_server_log_path = None
            self.server_root_identity = None
            self.server_known_processes = {}
            return result
        cleanup: dict[str, Any]
        if result.get("residual_pids"):
            cleanup = {
                "moved": [],
                "errors": [
                    "refusing MCCL quarantine while owned server "
                    f"processes remain: {result['residual_pids']}"
                ],
            }
        elif self.mccl_case_log_dir is not None:
            cleanup = offline_helpers.quarantine_new_mccl_shm_files(
                self.mccl_before,
                self.mccl_case_log_dir / "mccl_shm_quarantine",
                owned,
            )
        else:
            cleanup = {"moved": [], "errors": ["case log directory missing"]}
        result["mccl_shm_cleanup"] = cleanup
        if self.mccl_case_log_dir is not None:
            common.atomic_write_json(
                self.mccl_case_log_dir / "mccl_shm_cleanup.json",
                cleanup,
            )
        self.mccl_before = {}
        self.mccl_case_log_dir = None
        self.active_server_log_path = None
        self.server_root_identity = None
        self.server_known_processes = {}
        # Cooldown starts only after owned processes are gone and SHM hygiene
        # has completed.
        self.last_gpu_release_monotonic = time.monotonic()
        return result

    def create_manifest(self, cases: Sequence[Any]) -> None:
        super().create_manifest(cases)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.suite_kind == "formal":
            provenance_failures = []
            repositories = manifest.get("repositories")
            if not isinstance(repositories, dict):
                provenance_failures.append("repositories record is missing")
            else:
                for name in ("examples", "lightx2v"):
                    state = repositories.get(name)
                    if not isinstance(state, dict):
                        provenance_failures.append(f"{name}: state is missing")
                    elif (
                        state.get("commands_succeeded") is not True
                        or not state.get("revision")
                    ):
                        provenance_failures.append(
                            f"{name}: Git provenance is incomplete"
                        )
            if provenance_failures:
                raise RuntimeError(
                    "formal MetaX suite requires complete Git provenance: "
                    + "; ".join(provenance_failures)
                )
        manifest["profiling"] = {
            "level": 0,
            "reason": "formal server E2E latency benchmark",
        }
        manifest.setdefault("environment", {})["metax"] = {
            "device": "MetaX C500",
            "device_memory_gib": 64,
            "maca_path": os.environ.get("MACA_PATH", "/opt/maca-3.7.1"),
            "monitor_command": list(METAX_PATHS.monitor_command),
        }
        manifest.setdefault("tools", {})[str(Path(__file__).resolve())] = (
            common.sha256_file(Path(__file__))
        )
        runtime = SERVER_PATH / "metax_server_runtime.sh"
        manifest["tools"][str(runtime.resolve())] = common.sha256_file(runtime)
        common.atomic_write_json(self.manifest_path, manifest)


def _metax_arguments() -> Any:
    if not any(
        value == "--platform" or value.startswith("--platform=")
        for value in sys.argv[1:]
    ):
        sys.argv.extend(["--platform", PLATFORM_NAME])
    arguments = common.parse_arguments()
    if arguments.platform != PLATFORM_NAME:
        raise ValueError(
            f"this entrypoint only supports --platform {PLATFORM_NAME}"
        )
    return arguments


def main() -> int:
    os.environ["INFER_PROFILE_LEVEL"] = "0"
    os.environ["PROFILING_DEBUG_LEVEL"] = "0"
    return MetaXServiceSuite(_metax_arguments()).run()


if __name__ == "__main__":
    raise SystemExit(main())
