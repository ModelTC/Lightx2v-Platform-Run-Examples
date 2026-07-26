#!/usr/bin/env python3
"""Run all current Ascend distributed LightX2V service benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from run_infer_suite import parse_npu_snapshot
from service_benchmark_common import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    utc_now,
)

REPO_PATH = Path("/data/wushuo1/Lightx2v-Platform-Run-Examples")
LIGHTX2V_PATH = Path("/data/wushuo1/LightX2V")
LOG_ROOT = REPO_PATH / "logs" / "ascend_npu" / "server"
RESULT_ROOT = REPO_PATH / "results" / "ascend_npu" / "server"
SERVER_ROOT = REPO_PATH / "scripts" / "ascend" / "server" / "dist"
INFER_ROOT = REPO_PATH / "scripts" / "ascend" / "infer"
CONFIG_ROOT = REPO_PATH / "configs" / "ascend_npu"
DATA_ROOT = REPO_PATH / "data"
NO_PROXY_OPENER = build_opener(ProxyHandler({}))
SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ServiceCase:
    case_id: str
    task_kind: str
    world_size: int
    parallel_strategy: str
    service_script: str
    source_script: str
    model_config: str
    data: str


CASES: tuple[ServiceCase, ...] = (
    ServiceCase(
        "z_image_turbo_t2i_1664x928_sp2",
        "t2i",
        2,
        "sp2",
        "start_server_z_image_turbo_t2i_1664x928_sp2.sh",
        "dist_2/run_z_image_turbo_t2i_1664x928_sp2.sh",
        "dist_2/z_image_turbo_t2i_1664x928_sp2.json",
        "t2i_100.jsonl",
    ),
    ServiceCase(
        "flux2_dev_t2i_1344x768_tp8",
        "t2i",
        8,
        "tp8",
        "start_server_flux2_dev_t2i_1344x768_tp8.sh",
        "dist_8/run_flux2_dev_t2i_1344x768_tp8.sh",
        "dist_8/flux2_dev_t2i_1344x768_tp8.json",
        "t2i_100.jsonl",
    ),
    ServiceCase(
        "longcat_image_t2i_1344x768_cfg2_sp4",
        "t2i",
        8,
        "cfg2_sp4",
        "start_server_longcat_image_t2i_1344x768_cfg2_sp4.sh",
        "dist_8/run_longcat_image_t2i_1344x768_cfg2_sp4.sh",
        "dist_8/longcat_image_t2i_1344x768_cfg2_sp4.json",
        "t2i_100.jsonl",
    ),
    ServiceCase(
        "qwen_image_2512_t2i_1664x928_cfg2_sp4",
        "t2i",
        8,
        "cfg2_sp4",
        "start_server_qwen_image_2512_t2i_1664x928_cfg2_sp4.sh",
        "dist_8/run_qwen_image_2512_t2i_1664x928_cfg2_sp4.sh",
        "dist_8/qwen_image_2512_t2i_1664x928_cfg2_sp4.json",
        "t2i_100.jsonl",
    ),
    ServiceCase(
        "wan21_1_3b_t2v_480p_81f_cfg2_sp4",
        "t2v",
        8,
        "cfg2_sp4",
        "start_server_wan21_1_3b_t2v_480p_81f_cfg2_sp4.sh",
        "dist_8/run_wan21_1_3b_t2v_480p_81f_cfg2_sp4.sh",
        "dist_8/wan21_1_3b_t2v_480p_81f_cfg2_sp4.json",
        "t2v_10.jsonl",
    ),
    ServiceCase(
        "wan22_moe_a14b_t2v_480p_81f_cfg2_sp4",
        "t2v",
        8,
        "cfg2_sp4",
        "start_server_wan22_moe_a14b_t2v_480p_81f_cfg2_sp4.sh",
        "dist_8/run_wan22_moe_a14b_t2v_480p_81f_cfg2_sp4.sh",
        "dist_8/wan22_moe_a14b_t2v_480p_81f_cfg2_sp4.json",
        "t2v_10.jsonl",
    ),
    ServiceCase(
        "wan22_moe_a14b_t2v_480p_81f_tp8",
        "t2v",
        8,
        "tp8",
        "start_server_wan22_moe_a14b_t2v_480p_81f_tp8.sh",
        "dist_8/run_wan22_moe_a14b_t2v_480p_81f_tp8.sh",
        "dist_8/wan22_moe_a14b_t2v_480p_81f_tp8.json",
        "t2v_10.jsonl",
    ),
    ServiceCase(
        "wan22_moe_a14b_t2v_720p_81f_cfg2_sp4",
        "t2v",
        8,
        "cfg2_sp4",
        "start_server_wan22_moe_a14b_t2v_720p_81f_cfg2_sp4.sh",
        "dist_8/run_wan22_moe_a14b_t2v_720p_81f_cfg2_sp4.sh",
        "dist_8/wan22_moe_a14b_t2v_720p_81f_cfg2_sp4.json",
        "t2v_10.jsonl",
    ),
    ServiceCase(
        "wan22_moe_a14b_t2v_720p_81f_tp8",
        "t2v",
        8,
        "tp8",
        "start_server_wan22_moe_a14b_t2v_720p_81f_tp8.sh",
        "dist_8/run_wan22_moe_a14b_t2v_720p_81f_tp8.sh",
        "dist_8/wan22_moe_a14b_t2v_720p_81f_tp8.json",
        "t2v_10.jsonl",
    ),
    ServiceCase(
        "hunyuan_video_15_t2v_480p_121f_cfg2_sp4",
        "t2v",
        8,
        "cfg2_sp4",
        "start_server_hunyuan_video_15_t2v_480p_121f_cfg2_sp4.sh",
        "dist_8/run_hunyuan_video_15_t2v_480p_121f_cfg2_sp4.sh",
        "dist_8/hunyuan_video_15_t2v_480p_121f_cfg2_sp4.json",
        "t2v_10.jsonl",
    ),
    ServiceCase(
        "hunyuan_video_15_t2v_720p_121f_cfg2_sp4",
        "t2v",
        8,
        "cfg2_sp4",
        "start_server_hunyuan_video_15_t2v_720p_121f_cfg2_sp4.sh",
        "dist_8/run_hunyuan_video_15_t2v_720p_121f_cfg2_sp4.sh",
        "dist_8/hunyuan_video_15_t2v_720p_121f_cfg2_sp4.json",
        "t2v_10.jsonl",
    ),
    ServiceCase(
        "ltx2_3_22b_dev_s2v_768x512_241f_sp8",
        "s2v",
        8,
        "sp8",
        "start_server_ltx2_3_22b_dev_s2v_768x512_241f_sp8.sh",
        "dist_8/run_ltx2_3_22b_dev_s2v_768x512_241f_sp8.sh",
        "dist_8/ltx2_3_22b_dev_s2v_768x512_241f_sp8.json",
        "s2v_10.jsonl",
    ),
)


def suite_id_now() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"service_suite_{timestamp}_p{os.getpid()}"


def run_capture(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float = 30.0,
) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return completed.returncode, completed.stdout
    except (OSError, subprocess.TimeoutExpired) as exception:
        return 127, str(exception)


def git_state(repo: Path) -> dict[str, Any]:
    revision_code, revision = run_capture(
        ["git", "rev-parse", "HEAD"], cwd=repo
    )
    branch_code, branch = run_capture(
        ["git", "branch", "--show-current"], cwd=repo
    )
    status_code, status = run_capture(
        ["git", "status", "--short"], cwd=repo
    )
    diff_code, diff = run_capture(
        ["git", "diff", "--binary", "--no-ext-diff"], cwd=repo
    )
    cached_code, cached_diff = run_capture(
        ["git", "diff", "--binary", "--cached", "--no-ext-diff"],
        cwd=repo,
    )
    combined_diff = diff + cached_diff
    return {
        "path": str(repo),
        "revision": revision.strip() if revision_code == 0 else None,
        "branch": branch.strip() if branch_code == 0 else None,
        "dirty": bool(status.strip()) if status_code == 0 else None,
        "status": status,
        "diff_sha256": hashlib.sha256(
            combined_diff.encode("utf-8")
        ).hexdigest(),
        "commands_succeeded": all(
            code == 0
            for code in (
                revision_code,
                branch_code,
                status_code,
                diff_code,
                cached_code,
            )
        ),
        "working_diff": diff,
        "cached_diff": cached_diff,
    }


def case_paths(case: ServiceCase) -> dict[str, Path]:
    benchmark = (
        REPO_PATH / "bench_t2i_service.py"
        if case.task_kind == "t2i"
        else REPO_PATH / "bench_t2v_service.py"
    )
    return {
        "benchmark": benchmark,
        "service_script": SERVER_ROOT / case.service_script,
        "source_script": INFER_ROOT / case.source_script,
        "model_config": CONFIG_ROOT / case.model_config,
        "data": DATA_ROOT / case.data,
    }


def validate_parallel(case: ServiceCase, config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parallel = config.get("parallel")
    if not isinstance(parallel, dict):
        raise ValueError(f"{config_path}: missing parallel object")
    tensor_size = int(parallel.get("tensor_p_size", 1))
    cfg_size = int(parallel.get("cfg_p_size", 1))
    sequence_size = int(parallel.get("seq_p_size", 1))
    effective_size = (
        tensor_size if tensor_size > 1 else cfg_size * sequence_size
    )
    if effective_size != case.world_size:
        raise ValueError(
            f"{case.case_id}: config parallel size {effective_size} "
            f"does not match world_size {case.world_size}"
        )


class NpuSampler:
    def __init__(self, path: Path, interval_seconds: float) -> None:
        self.path = path
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.peak_hbm_mb: dict[int, int] = {}
        self.errors: list[str] = []

    def start(self) -> None:
        if self.interval_seconds <= 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(
            target=self._run,
            name="service-suite-npu-sampler",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        with self.path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(
                ["timestamp", "device_id", "hbm_used_mb", "error"]
            )
            while not self.stop_event.is_set():
                code, output = run_capture(
                    ["npu-smi", "info"], timeout=20.0
                )
                timestamp = utc_now()
                if code == 0:
                    try:
                        _, hbm_used = parse_npu_snapshot(output)
                    except ValueError as exception:
                        error = str(exception)
                        self.errors.append(error)
                        writer.writerow([timestamp, "", "", error])
                    else:
                        for device_id, used_mb in sorted(
                            hbm_used.items()
                        ):
                            writer.writerow(
                                [timestamp, device_id, used_mb, ""]
                            )
                            self.peak_hbm_mb[device_id] = max(
                                used_mb,
                                self.peak_hbm_mb.get(device_id, 0),
                            )
                else:
                    error = output.strip() or "npu-smi info failed"
                    self.errors.append(error)
                    writer.writerow([timestamp, "", "", error])
                file_obj.flush()
                self.stop_event.wait(self.interval_seconds)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=30)


class ServiceSuite:
    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments
        self.suite_id = arguments.suite_id or suite_id_now()
        self.log_root = LOG_ROOT / self.suite_id
        self.result_root = RESULT_ROOT / self.suite_id
        self.suite_log_path = self.log_root / "suite.log"
        self.state_path = self.result_root / "summary.json"
        self.report_path = self.result_root / "summary.md"
        self.manifest_path = self.result_root / "manifest.json"
        self.active_process: subprocess.Popen[bytes] | None = None
        self.active_server_run_id = ""
        self.active_client_process: subprocess.Popen[bytes] | None = None
        self.active_server_log = None
        self.interrupted = False
        self.state: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "suite_id": self.suite_id,
            "status": "pending",
            "started_at": utc_now(),
            "completed_at": None,
            "parameters": {
                "sample_count": arguments.sample_count,
                "warmup_count": arguments.warmup_count,
                "concurrency": arguments.concurrency,
                "poll_interval_seconds": (
                    arguments.poll_interval_seconds
                ),
                "startup_timeout_seconds": (
                    arguments.startup_timeout_seconds
                ),
                "task_timeout_seconds": arguments.task_timeout_seconds,
                "case_timeout_seconds": arguments.case_timeout_seconds,
                "npu_sample_interval_seconds": (
                    arguments.npu_sample_interval_seconds
                ),
                "port": arguments.port,
                "metric_port": arguments.metric_port,
                "master_port": arguments.master_port,
            },
            "log_root": str(self.log_root),
            "result_root": str(self.result_root),
            "cases": [],
        }

    def log(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        if self.suite_log_path.parent.is_dir():
            with self.suite_log_path.open(
                "a", encoding="utf-8"
            ) as file_obj:
                file_obj.write(line + "\n")

    def selected_cases(self) -> list[ServiceCase]:
        if not self.arguments.only:
            return list(CASES)
        requested = {
            item.strip()
            for value in self.arguments.only
            for item in value.split(",")
            if item.strip()
        }
        known = {case.case_id for case in CASES}
        unknown = requested - known
        if unknown:
            raise ValueError(f"unknown case ids: {sorted(unknown)}")
        return [case for case in CASES if case.case_id in requested]

    def validate(self, cases: Sequence[ServiceCase]) -> None:
        if self.arguments.sample_count <= 0:
            raise ValueError("--sample-count must be > 0")
        if self.arguments.warmup_count <= 0:
            raise ValueError("--warmup-count must be > 0")
        if self.arguments.concurrency <= 0:
            raise ValueError("--concurrency must be > 0")
        ports = {
            self.arguments.port,
            self.arguments.metric_port,
            self.arguments.master_port,
        }
        if len(ports) != 3:
            raise ValueError(
                "PORT, METRIC_PORT and MASTER_PORT must be distinct"
            )
        if not LIGHTX2V_PATH.is_dir():
            raise FileNotFoundError(
                f"LightX2V repository does not exist: {LIGHTX2V_PATH}"
            )
        for case in cases:
            paths = case_paths(case)
            for name, path in paths.items():
                if not path.is_file():
                    raise FileNotFoundError(
                        f"{case.case_id}: {name} does not exist: {path}"
                    )
            validate_parallel(case, paths["model_config"])
            data_lines = sum(
                1
                for line in paths["data"].read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )
            required_lines = max(
                self.arguments.sample_count,
                self.arguments.warmup_count,
            )
            if data_lines < required_lines:
                raise ValueError(
                    f"{paths['data']} has {data_lines} rows; "
                    f"{required_lines} required"
                )

    def case_manifest(self, case: ServiceCase) -> dict[str, Any]:
        paths = case_paths(case)
        data_dependencies: list[dict[str, str]] = []
        if case.task_kind == "s2v":
            for line in paths["data"].read_text(
                encoding="utf-8"
            ).splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                audio_path = Path(str(row["audio_path"]))
                if not audio_path.is_absolute():
                    audio_path = paths["data"].parent / audio_path
                audio_path = audio_path.resolve()
                data_dependencies.append(
                    {
                        "path": str(audio_path),
                        "sha256": sha256_file(audio_path),
                    }
                )
        return {
            **asdict(case),
            "files": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for name, path in paths.items()
            },
            "data_dependencies": data_dependencies,
        }

    def create_manifest(self, cases: Sequence[ServiceCase]) -> None:
        examples_git = git_state(REPO_PATH)
        lightx2v_git = git_state(LIGHTX2V_PATH)
        provenance_dir = self.result_root / "provenance"
        provenance_dir.mkdir(parents=True, exist_ok=True)
        for name, state in (
            ("examples", examples_git),
            ("lightx2v", lightx2v_git),
        ):
            atomic_write_text(
                provenance_dir / f"{name}.status.txt",
                state.pop("status"),
            )
            atomic_write_text(
                provenance_dir / f"{name}.working.diff.patch",
                state.pop("working_diff"),
            )
            atomic_write_text(
                provenance_dir / f"{name}.cached.diff.patch",
                state.pop("cached_diff"),
            )

        pip_code, pip_freeze = run_capture(
            [sys.executable, "-m", "pip", "freeze"], timeout=120.0
        )
        atomic_write_text(
            provenance_dir / "pip_freeze.txt", pip_freeze
        )
        npu_smi_code, npu_smi_output = run_capture(
            ["npu-smi", "info"], timeout=30.0
        )
        atomic_write_text(
            provenance_dir / "npu_smi_initial.txt", npu_smi_output
        )
        ascend_versions: dict[str, str] = {}
        for candidate in (
            Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
            Path("/usr/local/Ascend/driver/version.info"),
            Path("/usr/local/Ascend/driver/version.info.json"),
        ):
            if candidate.is_file():
                try:
                    ascend_versions[str(candidate)] = (
                        candidate.read_text(
                            encoding="utf-8", errors="replace"
                        )
                    )
                except OSError as exception:
                    ascend_versions[str(candidate)] = str(exception)

        environment = {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "pip_freeze": {
                "path": str(provenance_dir / "pip_freeze.txt"),
                "command_succeeded": pip_code == 0,
                "sha256": hashlib.sha256(
                    pip_freeze.encode("utf-8")
                ).hexdigest(),
            },
            "npu_smi_initial": {
                "path": str(provenance_dir / "npu_smi_initial.txt"),
                "command_succeeded": npu_smi_code == 0,
                "sha256": hashlib.sha256(
                    npu_smi_output.encode("utf-8")
                ).hexdigest(),
            },
            "ascend_versions": ascend_versions,
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "ASCEND_HOME_PATH",
                    "ASCEND_OPP_PATH",
                    "LD_LIBRARY_PATH",
                    "PATH",
                    "PYTHONPATH",
                )
            },
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "suite_id": self.suite_id,
            "created_at": utc_now(),
            "parameters": self.state["parameters"],
            "repositories": {
                "examples": examples_git,
                "lightx2v": lightx2v_git,
            },
            "environment": environment,
            "tools": {
                str(path.resolve()): sha256_file(path)
                for path in (
                    Path(__file__),
                    REPO_PATH / "scripts" / "service_benchmark_common.py",
                    REPO_PATH / "bench_t2i_service.py",
                    REPO_PATH / "bench_t2v_service.py",
                )
            },
            "cases": [self.case_manifest(case) for case in cases],
        }
        atomic_write_json(self.manifest_path, manifest)

    def persist(self) -> None:
        atomic_write_json(self.state_path, self.state)
        atomic_write_text(self.report_path, self.render_markdown())

    def render_markdown(self) -> str:
        lines = [
            f"# Ascend service benchmark: {self.suite_id}",
            "",
            f"- Status: `{self.state['status']}`",
            f"- Started: `{self.state['started_at']}`",
            f"- Completed: `{self.state['completed_at']}`",
            (
                "- Workload: "
                f"`{self.arguments.warmup_count}` warm-up + "
                f"`{self.arguments.sample_count}` measured samples, "
                f"concurrency `{self.arguments.concurrency}`"
            ),
            f"- Suite log: `{self.suite_log_path}`",
            "",
            "| Case | Task | Cards | Parallel | Status | Success | "
            "P50 (s) | P90 (s) | Throughput/min | Peak HBM (MB) |",
            "|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
        ]
        for record in self.state["cases"]:
            result = record.get("benchmark_result") or {}
            latency = result.get("end_to_end_latency") or {}
            throughput = result.get("throughput") or {}
            throughput_value = next(
                (
                    value
                    for name, value in throughput.items()
                    if name.endswith("_per_minute")
                ),
                None,
            )
            success = (
                f"{result.get('requests_success', 0)}/"
                f"{result.get('requests_total', 0)}"
            )
            peak = record.get("peak_hbm_mb") or {}
            peak_value = max(peak.values()) if peak else None
            lines.append(
                "| "
                + " | ".join(
                    (
                        record["case_id"],
                        record["task_kind"],
                        str(record["world_size"]),
                        record["parallel_strategy"],
                        record["status"],
                        success,
                        self.format_number(latency.get("p50_s")),
                        self.format_number(latency.get("p90_s")),
                        self.format_number(throughput_value, digits=6),
                        self.format_number(peak_value, digits=0),
                    )
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "Each case directory contains the combined server log, "
                "client logs, raw Prometheus snapshots, NPU snapshots, "
                "per-request JSONL records, result JSON and generated media.",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def format_number(value: Any, digits: int = 4) -> str:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "N/A"
        return f"{value:.{digits}f}"

    def snapshot_npu(self, path: Path) -> tuple[set[int], dict[int, int]]:
        code, output = run_capture(["npu-smi", "info"], timeout=30.0)
        atomic_write_text(path, output)
        if code != 0:
            raise RuntimeError(f"npu-smi info failed: {output}")
        try:
            process_ids, hbm_used = parse_npu_snapshot(output)
        except ValueError as exception:
            raise RuntimeError(str(exception)) from exception
        return process_ids, hbm_used

    def ensure_npu_idle(self, snapshot_path: Path) -> None:
        process_ids, _ = self.snapshot_npu(snapshot_path)
        if process_ids:
            raise RuntimeError(
                "NPU is already in use; refusing to terminate unowned "
                f"processes: {sorted(process_ids)}"
            )

    def wait_npu_idle(
        self, snapshot_path: Path, timeout_seconds: float = 180.0
    ) -> bool:
        deadline = time.monotonic() + timeout_seconds
        last_output = ""
        while time.monotonic() < deadline:
            code, output = run_capture(["npu-smi", "info"], timeout=30.0)
            last_output = output
            if code == 0:
                try:
                    process_ids, _ = parse_npu_snapshot(output)
                except ValueError:
                    pass
                else:
                    if not process_ids:
                        atomic_write_text(snapshot_path, output)
                        return True
            time.sleep(2)
        atomic_write_text(snapshot_path, last_output)
        return False

    def ensure_port_free(self, port: int) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"port {port} is already in use")

    def wait_health(self, process: subprocess.Popen[bytes]) -> float:
        start = time.perf_counter()
        deadline = start + self.arguments.startup_timeout_seconds
        url = f"http://127.0.0.1:{self.arguments.port}/health"
        last_error = ""
        while time.perf_counter() < deadline:
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"service exited before ready with code {exit_code}"
                )
            try:
                with NO_PROXY_OPENER.open(url, timeout=2.0) as response:
                    if response.getcode() == 200:
                        return time.perf_counter() - start
            except (OSError, URLError) as exception:
                last_error = str(exception)
            time.sleep(1)
        raise TimeoutError(
            "service health check timed out after "
            f"{self.arguments.startup_timeout_seconds}s: {last_error}"
        )

    def scrape_metrics(self, path: Path) -> str | None:
        url = (
            f"http://127.0.0.1:{self.arguments.metric_port}/metrics"
        )
        try:
            with NO_PROXY_OPENER.open(url, timeout=10.0) as response:
                text = response.read().decode("utf-8", errors="replace")
            atomic_write_text(path, text)
            return None
        except (OSError, URLError) as exception:
            error = str(exception)
            atomic_write_text(path, f"# scrape_error: {error}\n")
            return error

    def start_server(
        self, case: ServiceCase, case_log_dir: Path, case_result_dir: Path
    ) -> tuple[subprocess.Popen[bytes], float]:
        for port in (
            self.arguments.port,
            self.arguments.metric_port,
            self.arguments.master_port,
        ):
            self.ensure_port_free(port)
        server_log_path = case_log_dir / "server.log"
        self.active_server_log = server_log_path.open("wb")
        server_run_id = f"{self.suite_id}:{case.case_id}"
        environment = os.environ.copy()
        environment.update(
            {
                "RUN_ID": server_run_id,
                "PORT": str(self.arguments.port),
                "METRIC_PORT": str(self.arguments.metric_port),
                "MASTER_PORT": str(self.arguments.master_port),
                "MAX_QUEUE_SIZE": str(
                    max(10, self.arguments.sample_count)
                ),
                "LIGHTX2V_CACHE_DIR": str(
                    case_result_dir / "server_cache"
                ),
            }
        )
        process = subprocess.Popen(
            ["bash", str(SERVER_ROOT / case.service_script)],
            cwd=REPO_PATH,
            env=environment,
            stdout=self.active_server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.active_process = process
        self.active_server_run_id = server_run_id
        startup_seconds = self.wait_health(process)
        return process, startup_seconds

    @staticmethod
    def owned_server_pids(run_id: str) -> list[int]:
        owned: list[int] = []
        marker = f"RUN_ID={run_id}".encode()
        for proc_path in Path("/proc").iterdir():
            if not proc_path.name.isdigit():
                continue
            try:
                environment = (proc_path / "environ").read_bytes().split(b"\0")
            except OSError:
                continue
            if marker in environment:
                owned.append(int(proc_path.name))
        return sorted(owned)

    @classmethod
    def signal_owned_server(cls, run_id: str, signum: int) -> list[int]:
        signaled: list[int] = []
        for pid in reversed(cls.owned_server_pids(run_id)):
            try:
                os.kill(pid, signum)
                signaled.append(pid)
            except ProcessLookupError:
                pass
        return signaled

    @classmethod
    def wait_owned_server_exit(
        cls, run_id: str, timeout_seconds: float
    ) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if not cls.owned_server_pids(run_id):
                return True
            time.sleep(0.5)
        return not cls.owned_server_pids(run_id)

    def stop_server(self) -> dict[str, Any]:
        process = self.active_process
        run_id = self.active_server_run_id
        result = {
            "attempted": process is not None or bool(run_id),
            "pid": process.pid if process is not None else None,
            "run_id": run_id or None,
            "term_pids": [],
            "kill_pids": [],
            "residual_pids": [],
            "graceful": False,
            "forced": False,
            "exit_code": None,
        }
        if run_id:
            result["term_pids"] = self.signal_owned_server(
                run_id, signal.SIGTERM
            )
            result["graceful"] = self.wait_owned_server_exit(run_id, 60)
            if not result["graceful"]:
                result["kill_pids"] = self.signal_owned_server(
                    run_id, signal.SIGKILL
                )
                result["forced"] = bool(result["kill_pids"])
                self.wait_owned_server_exit(run_id, 30)
            result["residual_pids"] = self.owned_server_pids(run_id)
        elif process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=60)
                result["graceful"] = True
            except ProcessLookupError:
                result["graceful"] = True
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    result["forced"] = True
                    process.wait(timeout=30)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if process is not None:
            result["exit_code"] = process.poll()
        self.active_process = None
        self.active_server_run_id = ""
        if self.active_server_log is not None:
            self.active_server_log.close()
            self.active_server_log = None
        return result

    def benchmark_command(
        self,
        case: ServiceCase,
        *,
        phase: str,
        limit: int,
        run_dir: Path,
    ) -> list[str]:
        paths = case_paths(case)
        command = [
            sys.executable,
            str(paths["benchmark"]),
            "--url",
            f"http://127.0.0.1:{self.arguments.port}",
            "--data",
            str(paths["data"]),
            "--limit",
            str(limit),
            "--repeat",
            "1",
            "--concurrency",
            str(self.arguments.concurrency),
            "--platform",
            "ascend_npu",
            "--case-id",
            case.case_id,
            "--run-id",
            self.suite_id,
            "--phase",
            phase,
            "--run-dir",
            str(run_dir),
            "--source-script",
            str(paths["source_script"]),
            "--service-script",
            str(paths["service_script"]),
            "--model-config",
            str(paths["model_config"]),
        ]
        if case.task_kind == "t2i":
            command.extend(
                [
                    "--timeout-seconds",
                    str(int(self.arguments.task_timeout_seconds)),
                    "--poll-interval-seconds",
                    str(self.arguments.poll_interval_seconds),
                    "--request-timeout-seconds",
                    str(self.arguments.task_timeout_seconds + 60),
                ]
            )
        else:
            command.extend(
                [
                    "--poll-interval-seconds",
                    str(self.arguments.poll_interval_seconds),
                    "--request-timeout-seconds",
                    "30",
                    "--task-timeout-seconds",
                    str(self.arguments.task_timeout_seconds),
                ]
            )
        return command

    def run_client(
        self,
        command: Sequence[str],
        log_path: Path,
    ) -> tuple[int, float]:
        started = time.perf_counter()
        with log_path.open("wb") as file_obj:
            process = subprocess.Popen(
                list(command),
                cwd=REPO_PATH,
                stdout=file_obj,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.active_client_process = process
            try:
                exit_code = process.wait(
                    timeout=self.arguments.case_timeout_seconds
                )
            except KeyboardInterrupt:
                self.stop_client()
                raise
            except subprocess.TimeoutExpired:
                self.stop_client()
                exit_code = 124
            finally:
                if self.active_client_process is process:
                    self.active_client_process = None
        return exit_code, time.perf_counter() - started

    def stop_client(self) -> None:
        process = self.active_client_process
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
        self.active_client_process = None

    def run_case(self, index: int, case: ServiceCase) -> None:
        case_log_dir = self.log_root / case.case_id
        case_result_dir = self.result_root / case.case_id
        case_log_dir.mkdir(parents=True, exist_ok=False)
        case_result_dir.mkdir(parents=True, exist_ok=False)
        record: dict[str, Any] = {
            "index": index,
            "case_id": case.case_id,
            "task_kind": case.task_kind,
            "world_size": case.world_size,
            "parallel_strategy": case.parallel_strategy,
            "status": "running",
            "started_at": utc_now(),
            "completed_at": None,
            "log_dir": str(case_log_dir),
            "result_dir": str(case_result_dir),
            "startup_seconds": None,
            "warmup_exit_code": None,
            "benchmark_exit_code": None,
            "benchmark_result": None,
            "peak_hbm_mb": {},
            "diagnostic_warnings": [],
            "server_shutdown": None,
            "npu_released": False,
            "error": None,
        }
        self.state["cases"].append(record)
        self.persist()
        sampler: NpuSampler | None = None

        try:
            self.log(f"[{index}] {case.case_id}: checking NPU idle")
            self.ensure_npu_idle(case_log_dir / "npu_before.txt")
            _, startup_seconds = self.start_server(
                case, case_log_dir, case_result_dir
            )
            record["startup_seconds"] = startup_seconds
            self.log(
                f"[{index}] {case.case_id}: service ready in "
                f"{startup_seconds:.2f}s"
            )
            self.snapshot_npu(case_log_dir / "npu_ready.txt")
            cold_metrics_error = self.scrape_metrics(
                case_log_dir / "metrics_cold.prom"
            )
            if cold_metrics_error:
                record["diagnostic_warnings"].append(
                    f"cold metrics scrape: {cold_metrics_error}"
                )

            warmup_dir = case_result_dir / "warmup"
            warmup_command = self.benchmark_command(
                case,
                phase="warmup",
                limit=self.arguments.warmup_count,
                run_dir=warmup_dir,
            )
            warmup_code, warmup_seconds = self.run_client(
                warmup_command, case_log_dir / "warmup.log"
            )
            record["warmup_exit_code"] = warmup_code
            record["warmup_seconds"] = warmup_seconds
            if warmup_code != 0:
                raise RuntimeError(
                    f"warm-up failed with exit code {warmup_code}"
                )
            self.log(
                f"[{index}] {case.case_id}: warm-up passed in "
                f"{warmup_seconds:.2f}s"
            )

            before_metrics_error = self.scrape_metrics(
                case_log_dir / "metrics_before.prom"
            )
            if before_metrics_error:
                record["diagnostic_warnings"].append(
                    f"before metrics scrape: {before_metrics_error}"
                )
            sampler = NpuSampler(
                case_log_dir / "npu_samples.csv",
                self.arguments.npu_sample_interval_seconds,
            )
            sampler.start()
            measure_dir = case_result_dir / "measure"
            measure_command = self.benchmark_command(
                case,
                phase="measure",
                limit=self.arguments.sample_count,
                run_dir=measure_dir,
            )
            benchmark_code, benchmark_seconds = self.run_client(
                measure_command, case_log_dir / "client.log"
            )
            sampler.stop()
            record["benchmark_exit_code"] = benchmark_code
            record["benchmark_seconds"] = benchmark_seconds
            record["peak_hbm_mb"] = {
                str(device): value
                for device, value in sorted(sampler.peak_hbm_mb.items())
            }
            if sampler.errors:
                record["diagnostic_warnings"].append(
                    f"NPU sampler errors: {len(sampler.errors)}"
                )
            result_path = measure_dir / "result.json"
            if result_path.is_file():
                record["benchmark_result"] = json.loads(
                    result_path.read_text(encoding="utf-8")
                )
            if benchmark_code != 0:
                raise RuntimeError(
                    f"benchmark failed with exit code {benchmark_code}"
                )
            if (
                not isinstance(record["benchmark_result"], dict)
                or record["benchmark_result"].get("status") != "passed"
            ):
                raise RuntimeError(
                    "benchmark result is missing or not passed"
                )
            after_metrics_error = self.scrape_metrics(
                case_log_dir / "metrics_after.prom"
            )
            if after_metrics_error:
                record["diagnostic_warnings"].append(
                    f"after metrics scrape: {after_metrics_error}"
                )
            record["status"] = "passed"
            self.log(
                f"[{index}] {case.case_id}: "
                f"{self.arguments.sample_count}-sample benchmark passed"
            )
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            record["error"] = "interrupted"
            raise
        except Exception as exception:
            record["status"] = "failed"
            record["error"] = f"{type(exception).__name__}: {exception}"
            self.log(
                f"[{index}] {case.case_id}: failed: {record['error']}"
            )
        finally:
            self.stop_client()
            if sampler is not None:
                sampler.stop()
            metrics_after_path = case_log_dir / "metrics_after.prom"
            if (
                not metrics_after_path.exists()
                and self.active_process is not None
                and self.active_process.poll() is None
            ):
                after_metrics_error = self.scrape_metrics(
                    metrics_after_path
                )
                if after_metrics_error:
                    record["diagnostic_warnings"].append(
                        f"final metrics scrape: {after_metrics_error}"
                    )
            record["server_shutdown"] = self.stop_server()
            npu_after_path = case_log_dir / "npu_after.txt"
            if record["server_shutdown"]["attempted"]:
                record["npu_released"] = self.wait_npu_idle(
                    npu_after_path
                )
            else:
                try:
                    process_ids, _ = self.snapshot_npu(npu_after_path)
                    record["npu_released"] = not process_ids
                except RuntimeError as exception:
                    record["diagnostic_warnings"].append(
                        f"final NPU snapshot: {exception}"
                    )
                    record["npu_released"] = False
            if not record["npu_released"]:
                record["status"] = "failed"
                release_error = "NPU processes did not release after service"
                record["error"] = (
                    f"{record['error']}; {release_error}"
                    if record["error"]
                    else release_error
                )
            record["completed_at"] = utc_now()
            self.persist()

    def handle_signal(self, signum: int, _frame: Any) -> None:
        self.interrupted = True
        self.log(f"received {signal.Signals(signum).name}")
        raise KeyboardInterrupt

    def run(self) -> int:
        cases = self.selected_cases()
        self.validate(cases)
        if self.arguments.dry_run:
            plan = {
                "suite_id": self.suite_id,
                "parameters": self.state["parameters"],
                "cases": [
                    {
                        **self.case_manifest(case),
                        "warmup_command": self.benchmark_command(
                            case,
                            phase="warmup",
                            limit=self.arguments.warmup_count,
                            run_dir=Path("<result_dir>") / "warmup",
                        ),
                        "measure_command": self.benchmark_command(
                            case,
                            phase="measure",
                            limit=self.arguments.sample_count,
                            run_dir=Path("<result_dir>") / "measure",
                        ),
                    }
                    for case in cases
                ],
            }
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0

        self.log_root.mkdir(parents=True, exist_ok=False)
        self.result_root.mkdir(parents=True, exist_ok=False)
        self.create_manifest(cases)
        self.state["status"] = "running"
        self.persist()
        previous_handlers = {
            signum: signal.signal(signum, self.handle_signal)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        exit_code = 0
        try:
            for index, case in enumerate(cases, 1):
                self.run_case(index, case)
                if self.state["cases"][-1]["status"] != "passed":
                    exit_code = 1
                    if self.arguments.fail_fast:
                        break
        except KeyboardInterrupt:
            self.interrupted = True
            exit_code = 130
        finally:
            self.stop_client()
            self.stop_server()
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)
            statuses = {
                record["status"] for record in self.state["cases"]
            }
            if self.interrupted:
                self.state["status"] = "interrupted"
            elif len(self.state["cases"]) != len(cases):
                self.state["status"] = "incomplete"
            elif statuses == {"passed"}:
                self.state["status"] = "passed"
            else:
                self.state["status"] = "failed"
            self.state["completed_at"] = utc_now()
            self.persist()
            self.log(
                f"suite completed with status {self.state['status']}"
            )
        return exit_code


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start each current Ascend distributed service, warm it up, "
            "measure 10 samples and archive all logs/results."
        )
    )
    parser.add_argument("--suite-id", default="")
    parser.add_argument(
        "--only",
        action="append",
        help="Run only comma-separated case ids; may be repeated",
    )
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--warmup-count", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--poll-interval-seconds", type=float, default=0.5
    )
    parser.add_argument(
        "--startup-timeout-seconds", type=float, default=1800.0
    )
    parser.add_argument(
        "--task-timeout-seconds", type=float, default=7200.0
    )
    parser.add_argument(
        "--case-timeout-seconds", type=float, default=86400.0
    )
    parser.add_argument(
        "--npu-sample-interval-seconds", type=float, default=1.0
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--metric-port", type=int, default=8001)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed case; default continues safely",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and print the plan without starting services",
    )
    return parser.parse_args()


def main() -> int:
    return ServiceSuite(parse_arguments()).run()


if __name__ == "__main__":
    raise SystemExit(main())
