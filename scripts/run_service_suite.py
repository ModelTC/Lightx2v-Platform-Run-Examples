#!/usr/bin/env python3
"""Run the Ascend or MLU distributed LightX2V service benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import re
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
class PlatformPaths:
    name: str
    display_name: str
    repo_path: Path
    lightx2v_path: Path
    log_root: Path
    result_root: Path
    server_root: Path
    infer_root: Path
    config_root: Path
    data_root: Path
    monitor_command: tuple[str, ...]
    memory_label: str


MLU_REPO_PATH = Path("/data/Lightx2v-Platform-Run-Examples")
PLATFORMS: dict[str, PlatformPaths] = {
    "ascend_npu": PlatformPaths(
        name="ascend_npu",
        display_name="Ascend",
        repo_path=REPO_PATH,
        lightx2v_path=LIGHTX2V_PATH,
        log_root=LOG_ROOT,
        result_root=RESULT_ROOT,
        server_root=SERVER_ROOT,
        infer_root=INFER_ROOT,
        config_root=CONFIG_ROOT,
        data_root=DATA_ROOT,
        monitor_command=("npu-smi", "info"),
        memory_label="HBM",
    ),
    "mlu": PlatformPaths(
        name="mlu",
        display_name="Cambricon MLU590",
        repo_path=MLU_REPO_PATH,
        lightx2v_path=Path("/data/LightX2V-mlu"),
        log_root=MLU_REPO_PATH / "logs" / "mlu" / "server",
        result_root=MLU_REPO_PATH / "results" / "mlu" / "server",
        server_root=MLU_REPO_PATH
        / "scripts"
        / "mlu"
        / "server"
        / "dist",
        infer_root=MLU_REPO_PATH / "scripts" / "mlu" / "infer",
        config_root=MLU_REPO_PATH / "configs" / "mlu",
        data_root=MLU_REPO_PATH / "data",
        monitor_command=("cnmon", "all"),
        memory_label="MLU memory",
    ),
}


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


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    process_group_id: int
    session_id: int
    start_time_ticks: int
    run_id: str


def process_identity(pid: int) -> ProcessIdentity | None:
    proc_path = Path("/proc") / str(pid)
    try:
        stat_text = (proc_path / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    right_parenthesis = stat_text.rfind(")")
    if right_parenthesis < 0:
        return None
    fields = stat_text[right_parenthesis + 2 :].split()
    if len(fields) <= 19:
        return None
    try:
        process_group_id = int(fields[2])
        session_id = int(fields[3])
        start_time_ticks = int(fields[19])
    except ValueError:
        return None
    run_id = ""
    try:
        environment = (proc_path / "environ").read_bytes().split(b"\0")
    except OSError:
        environment = []
    for item in environment:
        if item.startswith(b"RUN_ID="):
            run_id = item.removeprefix(b"RUN_ID=").decode(
                "utf-8",
                errors="replace",
            )
            break
    return ProcessIdentity(
        pid=pid,
        process_group_id=process_group_id,
        session_id=session_id,
        start_time_ticks=start_time_ticks,
        run_id=run_id,
    )


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
    git_prefix = [
        "git",
        f"--git-dir={repo / '.git'}",
        f"--work-tree={repo}",
    ]
    revision_code, revision = run_capture(
        [*git_prefix, "rev-parse", "HEAD"], cwd=repo
    )
    branch_code, branch = run_capture(
        [*git_prefix, "branch", "--show-current"], cwd=repo
    )
    status_code, status = run_capture(
        [*git_prefix, "status", "--short"], cwd=repo
    )
    diff_code, diff = run_capture(
        [*git_prefix, "diff", "--binary", "--no-ext-diff"], cwd=repo
    )
    cached_code, cached_diff = run_capture(
        [
            *git_prefix,
            "diff",
            "--binary",
            "--cached",
            "--no-ext-diff",
        ],
        cwd=repo,
    )
    combined_diff = diff + cached_diff
    return {
        "path": str(repo),
        "revision": revision.strip() if revision_code == 0 else None,
        "branch": branch.strip() if branch_code == 0 else None,
        "dirty": bool(status.strip()) if status_code == 0 else None,
        "status_sha256": hashlib.sha256(
            status.encode("utf-8")
        ).hexdigest(),
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


def case_paths(
    case: ServiceCase,
    paths: PlatformPaths | None = None,
) -> dict[str, Path]:
    paths = paths or PLATFORMS["ascend_npu"]
    benchmark = (
        paths.repo_path / "bench_t2i_service.py"
        if case.task_kind == "t2i"
        else paths.repo_path / "bench_t2v_service.py"
    )
    return {
        "benchmark": benchmark,
        "service_script": paths.server_root / case.service_script,
        "source_script": paths.infer_root / case.source_script,
        "model_config": paths.config_root / case.model_config,
        "data": paths.data_root / case.data,
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


def parse_cnmon_snapshot(output: str) -> tuple[set[int], dict[int, int]]:
    """Return process IDs and per-card used memory from ``cnmon all``."""

    process_ids: set[int] = set()
    memory_used_mb: dict[int, int] = {}
    current_card: int | None = None
    in_process_table = False
    saw_process_table = False
    saw_no_process = False
    unparsed_process_rows: list[str] = []
    card_pattern = re.compile(r"^\|\s*(\d+)\s+/\s+\S+")
    memory_pattern = re.compile(
        r"\|\s*[\w/-]+\s+.*?\|\s*([\d,]+)\s+MiB\s*/\s*"
        r"[\d,]+\s+MiB\s*\|",
        re.IGNORECASE,
    )
    process_pattern = re.compile(
        r"^\|\s*(\d+)\s+\S+\s+(\d+)\s+.+?\s+"
        r"[\d,]+\s+MiB\s*\|\s*$",
        re.IGNORECASE,
    )

    for line in output.splitlines():
        if "Processes:" in line:
            in_process_table = True
            saw_process_table = True
            current_card = None
            continue
        if not in_process_table:
            card_match = card_pattern.match(line)
            if card_match:
                current_card = int(card_match.group(1))
                continue
            memory_match = memory_pattern.search(line)
            if memory_match and current_card is not None:
                memory_used_mb[current_card] = int(
                    memory_match.group(1).replace(",", "")
                )
                current_card = None
        else:
            if "No running processes found" in line:
                saw_no_process = True
                continue
            if (
                not line.strip()
                or line.lstrip().startswith("+")
                or (
                    "Card" in line
                    and "PID" in line
                    and "Command Line" in line
                )
                or set(line.strip()) <= {"|", "="}
            ):
                continue
            process_match = process_pattern.match(line)
            if process_match:
                card_id = int(process_match.group(1))
                if card_id not in range(8):
                    unparsed_process_rows.append(line)
                    continue
                process_ids.add(int(process_match.group(2)))
            elif line.lstrip().startswith("|"):
                unparsed_process_rows.append(line)

    expected_cards = set(range(8))
    if set(memory_used_mb) != expected_cards:
        raise ValueError(
            "cnmon device/memory table was incomplete: "
            f"expected={sorted(expected_cards)}, "
            f"parsed={sorted(memory_used_mb)}"
        )
    if not saw_process_table:
        raise ValueError("cnmon process table header was not found")
    if unparsed_process_rows:
        raise ValueError(
            "cnmon process table contained unparsed rows: "
            + " | ".join(unparsed_process_rows[:3])
        )
    if saw_no_process and process_ids:
        raise ValueError(
            "cnmon process table reported both no processes and process rows"
        )
    if not saw_no_process and not process_ids:
        raise ValueError(
            "cnmon process table had neither an explicit empty marker nor "
            "parseable process rows"
        )
    return process_ids, memory_used_mb


def parse_device_snapshot(
    output: str,
    platform_name: str,
) -> tuple[set[int], dict[int, int]]:
    if platform_name == "mlu":
        return parse_cnmon_snapshot(output)
    return parse_npu_snapshot(output)


class DeviceSampler:
    def __init__(
        self,
        path: Path,
        interval_seconds: float,
        platform_paths: PlatformPaths | None = None,
    ) -> None:
        self.path = path
        self.interval_seconds = interval_seconds
        self.platform_paths = (
            platform_paths or PLATFORMS["ascend_npu"]
        )
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.peak_device_memory_mb: dict[int, int] = {}
        self.errors: list[str] = []

    def start(self) -> None:
        if self.interval_seconds <= 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(
            target=self._run,
            name="service-suite-device-sampler",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        with self.path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.writer(file_obj)
            memory_column = (
                "hbm_used_mb"
                if self.platform_paths.name == "ascend_npu"
                else "device_memory_used_mb"
            )
            writer.writerow(
                [
                    "timestamp",
                    "device_id",
                    memory_column,
                    "error",
                ]
            )
            while not self.stop_event.is_set():
                code, output = run_capture(
                    self.platform_paths.monitor_command,
                    timeout=20.0,
                )
                timestamp = utc_now()
                if code == 0:
                    try:
                        _, memory_used = parse_device_snapshot(
                            output,
                            self.platform_paths.name,
                        )
                    except ValueError as exception:
                        error = str(exception)
                        self.errors.append(error)
                        writer.writerow([timestamp, "", "", error])
                    else:
                        for device_id, used_mb in sorted(
                            memory_used.items()
                        ):
                            writer.writerow(
                                [timestamp, device_id, used_mb, ""]
                            )
                            self.peak_device_memory_mb[device_id] = max(
                                used_mb,
                                self.peak_device_memory_mb.get(
                                    device_id, 0
                                ),
                            )
                else:
                    error = (
                        output.strip()
                        or "device monitor command failed"
                    )
                    self.errors.append(error)
                    writer.writerow([timestamp, "", "", error])
                file_obj.flush()
                self.stop_event.wait(self.interval_seconds)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=30)

    @property
    def peak_hbm_mb(self) -> dict[int, int]:
        """Compatibility name for existing Ascend callers and reports."""

        return self.peak_device_memory_mb


NpuSampler = DeviceSampler


class ServiceSuite:
    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments
        self.platform_name = getattr(
            arguments, "platform", "ascend_npu"
        )
        self.platform_paths = PLATFORMS[self.platform_name]
        self.repo_path = self.platform_paths.repo_path
        self.lightx2v_path = self.platform_paths.lightx2v_path
        self.server_root = self.platform_paths.server_root
        requested_suite_kind = os.environ.get(
            "LIGHTX2V_SERVICE_SUITE_KIND"
        )
        self.requested_suite_kind = (
            requested_suite_kind.strip()
            if requested_suite_kind is not None
            else None
        )
        self.suite_kind = self.requested_suite_kind or "formal"
        if self.suite_kind not in {"formal", "diagnostic"}:
            raise ValueError(
                "LIGHTX2V_SERVICE_SUITE_KIND must be formal or "
                "diagnostic"
            )
        requested_profile_level = os.environ.get(
            "INFER_PROFILE_LEVEL",
            os.environ.get("PROFILING_DEBUG_LEVEL"),
        )
        if self.platform_name == "mlu":
            self.infer_profile_level = requested_profile_level or "0"
            if self.infer_profile_level not in {"0", "1", "2"}:
                raise ValueError(
                    "INFER_PROFILE_LEVEL must be 0, 1 or 2"
                )
        else:
            self.infer_profile_level = requested_profile_level
        resume_suite_id = getattr(arguments, "resume", "") or ""
        requested_suite_id = getattr(arguments, "suite_id", "") or ""
        if (
            resume_suite_id
            and requested_suite_id
            and resume_suite_id != requested_suite_id
        ):
            raise ValueError(
                "--suite-id and --resume identify different suites"
            )
        self.resume = bool(resume_suite_id)
        self.device_sample_interval_seconds = float(
            getattr(
                arguments,
                "device_sample_interval_seconds",
                getattr(arguments, "npu_sample_interval_seconds", 1.0),
            )
        )
        self.memory_release_tolerance_mb = int(
            getattr(arguments, "memory_release_tolerance_mb", 256)
        )
        self.suite_id = (
            resume_suite_id or requested_suite_id or suite_id_now()
        )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.suite_id):
            raise ValueError(
                "suite id may contain only letters, digits, underscores "
                "and hyphens"
            )
        self.log_root = self.platform_paths.log_root / self.suite_id
        self.result_root = self.platform_paths.result_root / self.suite_id
        self.suite_log_path = self.log_root / "suite.log"
        self.state_path = self.result_root / "summary.json"
        self.report_path = self.result_root / "summary.md"
        self.manifest_path = self.result_root / "manifest.json"
        self.lock_path = self.result_root / "controller.lock"
        self.controller_pid_path = self.log_root / "controller.pid"
        self.lock_handle: Any = None
        self.active_process: subprocess.Popen[bytes] | None = None
        self.active_server_identity: ProcessIdentity | None = None
        self.active_server_run_id = ""
        self.active_client_process: subprocess.Popen[bytes] | None = None
        self.active_server_log = None
        self.interrupted = False
        self.state: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "suite_id": self.suite_id,
            "platform": self.platform_name,
            "display_name": self.platform_paths.display_name,
            "suite_kind": self.suite_kind,
            "status": "pending",
            "started_at": utc_now(),
            "completed_at": None,
            "controller_pid": os.getpid(),
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
                "device_sample_interval_seconds": (
                    self.device_sample_interval_seconds
                ),
                # Preserve the original field for Ascend consumers.
                "npu_sample_interval_seconds": (
                    self.device_sample_interval_seconds
                    if self.platform_name == "ascend_npu"
                    else None
                ),
                "memory_release_tolerance_mb": (
                    self.memory_release_tolerance_mb
                ),
                "infer_profile_level": self.infer_profile_level,
                "port": arguments.port,
                "metric_port": arguments.metric_port,
                "master_port": arguments.master_port,
            },
            "log_root": str(self.log_root),
            "result_root": str(self.result_root),
            "selected_case_ids": [],
            "cases": [],
        }

    def acquire_lock(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(
                lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            lock_handle.close()
            raise RuntimeError(
                "suite already has an active controller: "
                f"{self.lock_path}"
            ) from None
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(f"{os.getpid()}\n")
        lock_handle.flush()
        self.lock_handle = lock_handle

    def publish_controller_pid(self) -> None:
        atomic_write_text(
            self.controller_pid_path,
            f"{os.getpid()}\n",
        )

    def load_resume_state(self) -> None:
        if not self.state_path.is_file():
            raise FileNotFoundError(
                f"resume state does not exist: {self.state_path}"
            )
        with self.state_path.open("r", encoding="utf-8") as file_obj:
            state = json.load(file_obj)
        if not isinstance(state, dict):
            raise ValueError(
                f"resume state is not a JSON object: {self.state_path}"
            )
        if state.get("suite_id") != self.suite_id:
            raise ValueError(
                "resume suite id does not match state: "
                f"{state.get('suite_id')!r}"
            )
        stored_platform = state.get("platform", "ascend_npu")
        if stored_platform != self.platform_name:
            raise ValueError(
                f"resume platform is {stored_platform!r}, requested "
                f"{self.platform_name!r}"
            )
        stored_suite_kind = state.get("suite_kind", "formal")
        if stored_suite_kind not in {"formal", "diagnostic"}:
            raise ValueError(
                f"resume suite kind is invalid: {stored_suite_kind!r}"
            )
        if (
            self.requested_suite_kind is not None
            and stored_suite_kind != self.requested_suite_kind
        ):
            raise ValueError(
                f"resume suite kind is {stored_suite_kind!r}, requested "
                f"{self.requested_suite_kind!r}; set "
                "LIGHTX2V_SERVICE_SUITE_KIND to the original value"
            )
        self.suite_kind = stored_suite_kind
        if not isinstance(state.get("cases"), list):
            raise ValueError("resume state cases is not a list")
        stored_parameters = state.get("parameters")
        if isinstance(stored_parameters, dict):
            for name in (
                "sample_count",
                "warmup_count",
                "concurrency",
                "poll_interval_seconds",
                "startup_timeout_seconds",
                "task_timeout_seconds",
                "case_timeout_seconds",
                "port",
                "metric_port",
                "master_port",
            ):
                if name in stored_parameters:
                    setattr(
                        self.arguments,
                        name,
                        stored_parameters[name],
                    )
            sample_interval = stored_parameters.get(
                "device_sample_interval_seconds",
                stored_parameters.get(
                    "npu_sample_interval_seconds",
                    self.device_sample_interval_seconds,
                ),
            )
            self.device_sample_interval_seconds = float(sample_interval)
            self.memory_release_tolerance_mb = int(
                stored_parameters.get(
                    "memory_release_tolerance_mb",
                    self.memory_release_tolerance_mb,
                )
            )
            stored_profile_level = stored_parameters.get(
                "infer_profile_level",
                "0" if self.platform_name == "mlu" else None,
            )
            if stored_profile_level is not None:
                stored_profile_level = str(stored_profile_level)
            if (
                self.platform_name == "mlu"
                and stored_profile_level not in {"0", "1", "2"}
            ):
                raise ValueError(
                    "resume state has an invalid infer_profile_level: "
                    f"{stored_profile_level!r}"
                )
            self.infer_profile_level = stored_profile_level
        if self.platform_name == "mlu":
            os.environ["INFER_PROFILE_LEVEL"] = (
                self.infer_profile_level or "0"
            )
            os.environ["PROFILING_DEBUG_LEVEL"] = (
                self.infer_profile_level or "0"
            )
        self.state = state
        self.state["status"] = "running"
        self.state["completed_at"] = None
        self.state["controller_pid"] = os.getpid()

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
        if self.device_sample_interval_seconds < 0:
            raise ValueError(
                "--device-sample-interval-seconds must be >= 0"
            )
        if self.memory_release_tolerance_mb < 0:
            raise ValueError(
                "--memory-release-tolerance-mb must be >= 0"
            )
        if (
            self.platform_name == "mlu"
            and self.suite_kind == "formal"
            and self.infer_profile_level != "0"
        ):
            raise ValueError(
                "formal MLU server suites require INFER_PROFILE_LEVEL=0"
            )
        ports = {
            self.arguments.port,
            self.arguments.metric_port,
            self.arguments.master_port,
        }
        if len(ports) != 3:
            raise ValueError(
                "PORT, METRIC_PORT and MASTER_PORT must be distinct"
            )
        if not self.lightx2v_path.is_dir():
            raise FileNotFoundError(
                "LightX2V repository does not exist: "
                f"{self.lightx2v_path}"
            )
        for case in cases:
            paths = case_paths(case, self.platform_paths)
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
        paths = case_paths(case, self.platform_paths)
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
        examples_git = git_state(self.repo_path)
        lightx2v_git = git_state(self.lightx2v_path)
        if self.suite_kind == "formal":
            failed_repositories = [
                name
                for name, state in (
                    ("examples", examples_git),
                    ("lightx2v", lightx2v_git),
                )
                if state.get("commands_succeeded") is not True
            ]
            if failed_repositories:
                raise RuntimeError(
                    "formal suite requires complete Git provenance; "
                    "failed repositories: "
                    f"{failed_repositories}. Correct repository ownership "
                    "or Git access; no benchmark was started."
                )
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
        monitor_code, monitor_output = run_capture(
            self.platform_paths.monitor_command,
            timeout=30.0,
        )
        monitor_name = self.platform_paths.monitor_command[0]
        atomic_write_text(
            provenance_dir / f"{monitor_name}_initial.txt",
            monitor_output,
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
        cambricon_versions: dict[str, str] = {}
        if self.platform_name == "mlu":
            version_match = re.search(
                r"CNMON\s+v(\S+).*?Driver\s+v(\S+)",
                monitor_output,
                re.DOTALL,
            )
            if version_match:
                cambricon_versions.update(
                    {
                        "cnmon": version_match.group(1),
                        "driver": version_match.group(2),
                    }
                )
            for candidate in (
                Path("/usr/local/neuware/version.txt"),
                Path("/usr/local/neuware/version.cfg"),
                Path("/usr/local/neuware/lib64/libcndrv.so"),
            ):
                if not candidate.is_file():
                    continue
                if candidate.suffix == ".so":
                    cambricon_versions[str(candidate)] = (
                        f"sha256:{sha256_file(candidate)}"
                    )
                else:
                    try:
                        cambricon_versions[str(candidate)] = (
                            candidate.read_text(
                                encoding="utf-8",
                                errors="replace",
                            )
                        )
                    except OSError as exception:
                        cambricon_versions[str(candidate)] = str(
                            exception
                        )
            pip_show_code, pip_show_output = run_capture(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "show",
                    "torch",
                    "torch_mlu",
                ],
                timeout=60.0,
            )
            atomic_write_text(
                provenance_dir / "torch_mlu_packages.txt",
                pip_show_output,
            )
            cambricon_versions["torch_mlu_packages"] = (
                str(provenance_dir / "torch_mlu_packages.txt")
            )
            cambricon_versions["torch_mlu_packages_command_succeeded"] = (
                str(pip_show_code == 0).lower()
            )

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
            "device_monitor_initial": {
                "name": monitor_name,
                "command": list(self.platform_paths.monitor_command),
                "path": str(
                    provenance_dir / f"{monitor_name}_initial.txt"
                ),
                "command_succeeded": monitor_code == 0,
                "sha256": hashlib.sha256(
                    monitor_output.encode("utf-8")
                ).hexdigest(),
            },
            "ascend_versions": ascend_versions,
            "cambricon_versions": cambricon_versions,
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "ASCEND_HOME_PATH",
                    "ASCEND_OPP_PATH",
                    "NEUWARE_HOME",
                    "MLU_VISIBLE_DEVICES",
                    "CN_VISIBLE_DEVICES",
                    "LD_LIBRARY_PATH",
                    "PATH",
                    "PYTHONPATH",
                )
            },
        }
        if self.platform_name == "ascend_npu":
            environment["npu_smi_initial"] = dict(
                environment["device_monitor_initial"]
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "suite_id": self.suite_id,
            "platform": self.platform_name,
            "suite_kind": self.suite_kind,
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
                    self.repo_path
                    / "scripts"
                    / "service_benchmark_common.py",
                    self.repo_path / "bench_t2i_service.py",
                    self.repo_path / "bench_t2v_service.py",
                )
            },
            "cases": [self.case_manifest(case) for case in cases],
        }
        atomic_write_json(self.manifest_path, manifest)

    def validate_resume_manifest(
        self,
        cases: Sequence[ServiceCase],
    ) -> None:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"resume manifest does not exist: {self.manifest_path}"
            )
        try:
            manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exception:
            raise ValueError(
                f"cannot read resume manifest: {exception}"
            ) from exception
        if not isinstance(manifest, dict):
            raise ValueError("resume manifest is not a JSON object")
        if manifest.get("suite_id") != self.suite_id:
            raise ValueError("resume manifest suite_id does not match")
        if manifest.get("platform", "ascend_npu") != self.platform_name:
            raise ValueError("resume manifest platform does not match")
        if manifest.get("suite_kind", "formal") != self.suite_kind:
            raise ValueError("resume manifest suite_kind does not match")
        if manifest.get("parameters") != self.state.get("parameters"):
            raise RuntimeError(
                "resume manifest parameters differ from saved state; "
                "start a new suite"
            )

        stored_repositories = manifest.get("repositories")
        if not isinstance(stored_repositories, dict):
            raise ValueError("resume manifest has no repositories object")
        for name, repository in (
            ("examples", self.repo_path),
            ("lightx2v", self.lightx2v_path),
        ):
            stored = stored_repositories.get(name)
            if not isinstance(stored, dict):
                raise ValueError(
                    f"resume manifest has no {name} repository state"
                )
            current = git_state(repository)
            if (
                stored.get("commands_succeeded") is not True
                or current.get("commands_succeeded") is not True
            ):
                raise RuntimeError(
                    "cannot verify Git provenance while resuming "
                    f"{name}; start a new suite after fixing Git access"
                )
            for field in (
                "revision",
                "branch",
                "dirty",
                "status_sha256",
                "diff_sha256",
            ):
                if stored.get(field) != current.get(field):
                    raise RuntimeError(
                        "resume provenance mismatch for "
                        f"{name}.{field}; start a new suite"
                    )

        stored_cases_value = manifest.get("cases")
        if not isinstance(stored_cases_value, list):
            raise ValueError("resume manifest cases is not a list")
        stored_cases = {
            item.get("case_id"): item
            for item in stored_cases_value
            if isinstance(item, dict)
            and isinstance(item.get("case_id"), str)
        }
        expected_case_ids = {case.case_id for case in cases}
        if set(stored_cases) != expected_case_ids:
            raise RuntimeError(
                "resume manifest case plan differs from saved state; "
                "start a new suite"
            )
        for case in cases:
            current_case = self.case_manifest(case)
            stored_case = stored_cases[case.case_id]
            stored_files = stored_case.get("files")
            if not isinstance(stored_files, dict):
                raise ValueError(
                    f"{case.case_id}: resume manifest files is invalid"
                )
            for name, current_file in current_case["files"].items():
                stored_file = stored_files.get(name)
                if (
                    not isinstance(stored_file, dict)
                    or stored_file.get("path") != current_file["path"]
                    or stored_file.get("sha256") != current_file["sha256"]
                ):
                    raise RuntimeError(
                        f"{case.case_id}: {name} changed since suite "
                        "creation; start a new suite"
                    )
            if stored_case.get("data_dependencies") != current_case.get(
                "data_dependencies"
            ):
                raise RuntimeError(
                    f"{case.case_id}: data dependencies changed since "
                    "suite creation; start a new suite"
                )

        stored_tools = manifest.get("tools")
        if not isinstance(stored_tools, dict):
            raise ValueError("resume manifest tools is not an object")
        expected_tools = (
            Path(__file__),
            self.repo_path / "scripts" / "service_benchmark_common.py",
            self.repo_path / "bench_t2i_service.py",
            self.repo_path / "bench_t2v_service.py",
        )
        for path in expected_tools:
            resolved = str(path.resolve())
            stored_hash = stored_tools.get(resolved)
            if (
                not isinstance(stored_hash, str)
                or not path.is_file()
                or sha256_file(path) != stored_hash
            ):
                raise RuntimeError(
                    f"resume tool changed or is missing: {resolved}; "
                    "start a new suite"
                )

    def persist(self) -> None:
        atomic_write_json(self.state_path, self.state)
        atomic_write_text(self.report_path, self.render_markdown())

    def render_markdown(self) -> str:
        lines = [
            (
                f"# {self.platform_paths.display_name} service benchmark: "
                f"{self.suite_id}"
            ),
            "",
            f"- Platform: `{self.platform_name}`",
            f"- Suite kind: `{self.suite_kind}`",
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
            "P50 (s) | P90 (s) | Throughput/min | "
            "Peak device memory (MB) |",
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
            peak = (
                record.get("peak_device_memory_mb")
                or record.get("peak_hbm_mb")
                or {}
            )
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
                "client logs, raw Prometheus snapshots, device snapshots, "
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

    def snapshot_device(
        self, path: Path
    ) -> tuple[set[int], dict[int, int]]:
        code, output = run_capture(
            self.platform_paths.monitor_command,
            timeout=30.0,
        )
        atomic_write_text(path, output)
        if code != 0:
            raise RuntimeError(
                f"{' '.join(self.platform_paths.monitor_command)} "
                f"failed: {output}"
            )
        try:
            process_ids, memory_used = parse_device_snapshot(
                output,
                self.platform_name,
            )
        except ValueError as exception:
            raise RuntimeError(str(exception)) from exception
        return process_ids, memory_used

    def snapshot_npu(
        self, path: Path
    ) -> tuple[set[int], dict[int, int]]:
        """Compatibility wrapper for existing Ascend integrations."""

        return self.snapshot_device(path)

    def ensure_device_idle(
        self, snapshot_path: Path
    ) -> dict[int, int]:
        process_ids, memory_used = self.snapshot_device(snapshot_path)
        if process_ids:
            raise RuntimeError(
                "accelerator is already in use; refusing to terminate "
                "unowned "
                f"processes: {sorted(process_ids)}"
            )
        if self.platform_name == "mlu":
            residual_memory = {
                device_id: used_mb
                for device_id, used_mb in memory_used.items()
                if used_mb > self.memory_release_tolerance_mb
            }
            if residual_memory:
                raise RuntimeError(
                    "MLU has no reported process but device memory exceeds "
                    "the idle tolerance; refusing to accept it as the "
                    f"baseline: {residual_memory}"
                )
        return memory_used

    def memory_is_released(
        self,
        memory_used: dict[int, int],
        baseline_memory_mb: dict[int, int],
    ) -> bool:
        if set(memory_used) != set(baseline_memory_mb):
            return False
        for device_id, used_mb in memory_used.items():
            if (
                used_mb
                > baseline_memory_mb[device_id]
                + self.memory_release_tolerance_mb
            ):
                return False
            if (
                self.platform_name == "mlu"
                and used_mb > self.memory_release_tolerance_mb
            ):
                return False
        return True

    def ensure_npu_idle(self, snapshot_path: Path) -> None:
        self.ensure_device_idle(snapshot_path)

    def wait_device_idle(
        self,
        snapshot_path: Path,
        baseline_memory_mb: dict[int, int] | None = None,
        timeout_seconds: float = 180.0,
    ) -> bool:
        deadline = time.monotonic() + timeout_seconds
        last_output = ""
        while time.monotonic() < deadline:
            code, output = run_capture(
                self.platform_paths.monitor_command,
                timeout=30.0,
            )
            last_output = output
            if code == 0:
                try:
                    process_ids, memory_used = parse_device_snapshot(
                        output,
                        self.platform_name,
                    )
                except ValueError:
                    pass
                else:
                    memory_released = (
                        True
                        if baseline_memory_mb is None
                        else self.memory_is_released(
                            memory_used,
                            baseline_memory_mb,
                        )
                    )
                    if not process_ids and memory_released:
                        atomic_write_text(snapshot_path, output)
                        return True
            time.sleep(2)
        atomic_write_text(snapshot_path, last_output)
        return False

    def wait_npu_idle(
        self, snapshot_path: Path, timeout_seconds: float = 180.0
    ) -> bool:
        return self.wait_device_idle(
            snapshot_path,
            timeout_seconds=timeout_seconds,
        )

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
        self,
        case: ServiceCase,
        case_log_dir: Path,
        case_result_dir: Path,
        attempt_number: int = 1,
    ) -> tuple[subprocess.Popen[bytes], float]:
        for port in (
            self.arguments.port,
            self.arguments.metric_port,
            self.arguments.master_port,
        ):
            self.ensure_port_free(port)
        server_log_path = case_log_dir / "server.log"
        self.active_server_log = server_log_path.open("wb")
        server_run_id = (
            f"{self.suite_id}:{case.case_id}:a{attempt_number}"
        )
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
        if self.platform_name == "mlu":
            environment["MLU_VISIBLE_DEVICES"] = ",".join(
                str(device_id)
                for device_id in range(case.world_size)
            )
            environment.pop("CN_VISIBLE_DEVICES", None)
        process = subprocess.Popen(
            ["bash", str(self.server_root / case.service_script)],
            cwd=self.repo_path,
            env=environment,
            stdout=self.active_server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.active_process = process
        self.active_server_run_id = server_run_id
        identity = process_identity(process.pid)
        if identity is not None and identity.run_id == server_run_id:
            self.active_server_identity = identity
        else:
            self.active_server_identity = None
        startup_seconds = self.wait_health(process)
        return process, startup_seconds

    @staticmethod
    def owned_server_pids(run_id: str) -> list[int]:
        owned: list[int] = []
        marker = f"RUN_ID={run_id}".encode()
        try:
            proc_paths = list(Path("/proc").iterdir())
        except OSError:
            return []
        for proc_path in proc_paths:
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
            except PermissionError:
                continue
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

    @staticmethod
    def process_identity_is_unchanged(
        expected: ProcessIdentity,
    ) -> bool:
        return process_identity(expected.pid) == expected

    @staticmethod
    def process_group_members(
        process_group_id: int,
    ) -> list[ProcessIdentity]:
        members: list[ProcessIdentity] = []
        try:
            proc_paths = list(Path("/proc").iterdir())
        except OSError:
            return []
        for proc_path in proc_paths:
            if not proc_path.name.isdigit():
                continue
            identity = process_identity(int(proc_path.name))
            if (
                identity is not None
                and identity.process_group_id == process_group_id
            ):
                members.append(identity)
        return sorted(members, key=lambda item: item.pid)

    @classmethod
    def signal_owned_process_group(
        cls,
        expected: ProcessIdentity | None,
        run_id: str,
        signum: int,
    ) -> bool:
        if (
            expected is None
            or not run_id
            or not cls.process_identity_is_unchanged(expected)
        ):
            return False
        members = cls.process_group_members(expected.process_group_id)
        if (
            not members
            or expected.pid not in {member.pid for member in members}
            or any(member.run_id != run_id for member in members)
        ):
            return False
        identities_before_signal = {
            member.pid: member for member in members
        }
        if any(
            process_identity(pid) != identity
            for pid, identity in identities_before_signal.items()
        ):
            return False
        try:
            os.killpg(expected.process_group_id, signum)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return True

    def stop_server(self) -> dict[str, Any]:
        process = self.active_process
        run_id = self.active_server_run_id
        identity = self.active_server_identity
        result = {
            "attempted": process is not None or bool(run_id),
            "pid": process.pid if process is not None else None,
            "run_id": run_id or None,
            "term_pids": [],
            "kill_pids": [],
            "residual_pids": [],
            "fallback_term": False,
            "fallback_kill": False,
            "fallback_refused": False,
            "graceful": False,
            "forced": False,
            "exit_code": None,
            "cleanup_complete": False,
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
        if process is not None and process.poll() is None:
            if self.signal_owned_process_group(
                identity,
                run_id,
                signal.SIGTERM,
            ):
                result["fallback_term"] = True
                try:
                    process.wait(timeout=60)
                    result["graceful"] = True
                except subprocess.TimeoutExpired:
                    if self.signal_owned_process_group(
                        identity,
                        run_id,
                        signal.SIGKILL,
                    ):
                        result["fallback_kill"] = True
                        result["forced"] = True
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            pass
                    else:
                        result["fallback_refused"] = True
            else:
                result["fallback_refused"] = True
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if process is not None:
            result["exit_code"] = process.poll()
        if run_id:
            result["residual_pids"] = self.owned_server_pids(run_id)
        cleanup_complete = (
            (process is None or process.poll() is not None)
            and not result["residual_pids"]
        )
        result["cleanup_complete"] = cleanup_complete
        if cleanup_complete:
            self.active_process = None
            self.active_server_identity = None
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
        paths = case_paths(case, self.platform_paths)
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
            self.platform_name,
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
                cwd=self.repo_path,
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

    def benchmark_phase_is_valid(
        self,
        case: ServiceCase,
        result_dir: Path,
        phase: str,
        expected_count: int,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        phase_dir = result_dir / phase
        result_path = phase_dir / "result.json"
        if not result_path.is_file():
            return (
                False,
                f"{phase} benchmark result does not exist: {result_path}",
                None,
            )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exception:
            return (
                False,
                f"cannot read {phase} benchmark result: {exception}",
                None,
            )
        if not isinstance(result, dict) or result.get("status") != "passed":
            return False, f"{phase} benchmark result is not passed", None
        if result.get("requests_failed") != 0:
            return (
                False,
                f"{phase} benchmark result contains failed requests",
                None,
            )
        if (
            result.get("requests_total") != expected_count
            or result.get("requests_success") != expected_count
        ):
            return (
                False,
                f"{phase} benchmark request counts do not match "
                f"{expected_count}",
                None,
            )
        run_config = result.get("run_config")
        expected_task_kind = case.task_kind
        if (
            not isinstance(run_config, dict)
            or run_config.get("platform") != self.platform_name
            or run_config.get("case_id") != case.case_id
            or run_config.get("run_id") != self.suite_id
            or run_config.get("phase") != phase
            or run_config.get("task_kind") != expected_task_kind
            or run_config.get("request_count") != expected_count
            or run_config.get("concurrency")
            != self.arguments.concurrency
            or run_config.get("limit") != expected_count
            or run_config.get("repeat") != 1
        ):
            return (
                False,
                f"{phase} benchmark run_config does not match the suite",
                None,
            )
        requests = result.get("request_results")
        if not isinstance(requests, list) or len(requests) != expected_count:
            return (
                False,
                f"{phase} benchmark request results are incomplete",
                None,
            )
        observed_indices: set[int] = set()
        outputs_dir = (phase_dir / "outputs").resolve()
        for request in requests:
            if (
                not isinstance(request, dict)
                or request.get("ok") is not True
                or request.get("phase") != phase
                or not isinstance(request.get("index"), int)
                or not isinstance(request.get("task_id"), str)
                or not request.get("task_id")
            ):
                return (
                    False,
                    f"{phase} benchmark contains an invalid request",
                    None,
                )
            observed_indices.add(request["index"])
            artifact_value = request.get("artifact_path")
            artifact_hash = request.get("artifact_sha256")
            if not isinstance(artifact_value, str) or not artifact_value:
                return (
                    False,
                    f"{phase} successful request has no artifact path",
                    None,
                )
            artifact_path = Path(artifact_value).resolve()
            if not artifact_path.is_file():
                return (
                    False,
                    f"{phase} artifact does not exist: {artifact_path}",
                    None,
                )
            try:
                artifact_path.relative_to(outputs_dir)
            except ValueError:
                return (
                    False,
                    f"{phase} artifact is outside its outputs directory: "
                    f"{artifact_path}",
                    None,
                )
            if artifact_path.stat().st_size != request.get(
                "artifact_bytes"
            ):
                return (
                    False,
                    f"{phase} artifact size changed: {artifact_path}",
                    None,
                )
            if (
                not isinstance(artifact_hash, str)
                or sha256_file(artifact_path) != artifact_hash
            ):
                return (
                    False,
                    f"{phase} artifact SHA-256 changed: {artifact_path}",
                    None,
                )
        if observed_indices != set(range(expected_count)):
            return (
                False,
                f"{phase} benchmark request indices are incomplete",
                None,
            )
        return True, "", result

    def validate_dit_evidence(
        self,
        case: ServiceCase,
        server_log_path: Path,
        benchmark_result: dict[str, Any],
    ) -> dict[str, Any]:
        if self.infer_profile_level == "0":
            return {
                "status": "disabled",
                "profile_level": "0",
                "server_log": str(server_log_path),
                "expected_steps": None,
                "task_ids": [],
                "rank_step_counts": {},
                "errors": [],
                "reason": (
                    "server benchmark uses profiling Level 0 to avoid "
                    "profiling overhead"
                ),
            }
        validation: dict[str, Any] = {
            "status": "failed",
            "server_log": str(server_log_path),
            "expected_steps": None,
            "task_ids": [],
            "rank_step_counts": {},
            "errors": [],
        }
        errors: list[str] = validation["errors"]
        if not server_log_path.is_file():
            errors.append("server.log does not exist")
            return validation
        config_path = case_paths(case, self.platform_paths)["model_config"]
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            infer_steps = int(config["infer_steps"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            errors.append("cannot read a positive infer_steps from config")
            return validation
        if infer_steps <= 0:
            errors.append("config infer_steps must be positive")
            return validation
        validation["expected_steps"] = infer_steps

        requests = benchmark_result.get("request_results")
        if not isinstance(requests, list):
            errors.append("measure request_results is not a list")
            return validation
        task_ids = [
            request.get("task_id")
            for request in requests
            if isinstance(request, dict)
            and isinstance(request.get("task_id"), str)
        ]
        if (
            len(task_ids) != self.arguments.sample_count
            or len(set(task_ids)) != len(task_ids)
        ):
            errors.append("measure task_ids are missing or duplicated")
            return validation
        validation["task_ids"] = task_ids
        expected_task_ids = set(task_ids)
        starts = {task_id: 0 for task_id in task_ids}
        ends = {task_id: 0 for task_id in task_ids}
        rank_counts = {
            task_id: {rank: 0 for rank in range(case.world_size)}
            for task_id in task_ids
        }
        unexpected_measure_tasks: set[str] = set()
        current_task_id: str | None = None
        start_pattern = re.compile(r"Processing task (\S+)\s*$")
        end_pattern = re.compile(
            r"Task (\S+) completed successfully\s*$"
        )
        infer_pattern = re.compile(
            r"\[Profile\]\s+Rank\s+(\d+)\s+-\s+"
            r"Level1_Log\b.*\binfer_main cost\b"
        )
        try:
            log_lines = server_log_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except OSError as exception:
            errors.append(f"cannot read server.log: {exception}")
            return validation
        for line in log_lines:
            start_match = start_pattern.search(line)
            if start_match:
                task_id = start_match.group(1)
                if "_measure_" in task_id and task_id not in expected_task_ids:
                    unexpected_measure_tasks.add(task_id)
                if task_id in expected_task_ids:
                    starts[task_id] += 1
                current_task_id = task_id
                continue
            infer_match = infer_pattern.search(line)
            if (
                infer_match
                and current_task_id in expected_task_ids
            ):
                rank = int(infer_match.group(1))
                if rank not in rank_counts[current_task_id]:
                    errors.append(
                        f"{current_task_id}: unexpected rank {rank}"
                    )
                else:
                    rank_counts[current_task_id][rank] += 1
                continue
            end_match = end_pattern.search(line)
            if end_match:
                task_id = end_match.group(1)
                if task_id in expected_task_ids:
                    ends[task_id] += 1
                if current_task_id == task_id:
                    current_task_id = None
        if unexpected_measure_tasks:
            errors.append(
                "unexpected measure tasks: "
                f"{sorted(unexpected_measure_tasks)}"
            )
        for task_id in task_ids:
            if starts[task_id] != 1:
                errors.append(
                    f"{task_id}: Processing task count={starts[task_id]}"
                )
            if ends[task_id] != 1:
                errors.append(
                    f"{task_id}: successful completion count={ends[task_id]}"
                )
            for rank in range(case.world_size):
                count = rank_counts[task_id][rank]
                if count != infer_steps:
                    errors.append(
                        f"{task_id}: rank {rank} infer_main "
                        f"count={count}, expected={infer_steps}"
                    )
        validation["rank_step_counts"] = {
            task_id: {
                str(rank): count
                for rank, count in sorted(counts.items())
            }
            for task_id, counts in rank_counts.items()
        }
        if not errors:
            validation["status"] = "passed"
        return validation

    def passed_record_is_valid(
        self,
        case: ServiceCase,
        record: dict[str, Any],
    ) -> tuple[bool, str]:
        if record.get("status") != "passed":
            return False, f"case status is {record.get('status')!r}"
        if record.get("device_released") is not True:
            return False, "accelerator release was not proven"
        server_shutdown = record.get("server_shutdown")
        if (
            not isinstance(server_shutdown, dict)
            or server_shutdown.get("cleanup_complete") is not True
            or bool(server_shutdown.get("residual_pids"))
            or server_shutdown.get("fallback_refused") is True
        ):
            return False, "server cleanup was not proven complete"
        peak_memory = record.get("peak_device_memory_mb")
        if (
            self.platform_name == "mlu"
            and self.suite_kind == "formal"
            and (
                not isinstance(peak_memory, dict)
                or {
                    int(device)
                    for device in peak_memory
                    if str(device).isdigit()
                }
                != set(range(8))
            )
        ):
            return False, "device sampler has no complete 0..7 sample"
        result_dir_value = record.get("result_dir")
        log_dir_value = record.get("log_dir")
        if not isinstance(result_dir_value, str) or not result_dir_value:
            return False, "case has no result directory"
        if not isinstance(log_dir_value, str) or not log_dir_value:
            return False, "case has no log directory"
        result_dir = Path(result_dir_value)
        warmup_valid, reason, warmup_result = (
            self.benchmark_phase_is_valid(
                case,
                result_dir,
                "warmup",
                self.arguments.warmup_count,
            )
        )
        if not warmup_valid:
            return False, reason
        measure_valid, reason, result = self.benchmark_phase_is_valid(
            case,
            result_dir,
            "measure",
            self.arguments.sample_count,
        )
        if not measure_valid or result is None:
            return False, reason
        record["warmup_result"] = warmup_result
        record["benchmark_result"] = result
        dit_validation = self.validate_dit_evidence(
            case,
            Path(log_dir_value) / "server.log",
            result,
        )
        record["dit_validation"] = dit_validation
        if (
            self.suite_kind == "formal"
            and self.infer_profile_level != "0"
            and dit_validation["status"] != "passed"
        ):
            return False, "single-step DiT evidence is incomplete"
        if (
            self.suite_kind == "diagnostic"
            and self.infer_profile_level != "0"
            and dit_validation["status"] != "passed"
        ):
            warning = (
                "single-step DiT evidence is incomplete: "
                + "; ".join(dit_validation["errors"][:3])
            )
            warnings = record.setdefault("diagnostic_warnings", [])
            if isinstance(warnings, list) and warning not in warnings:
                warnings.append(warning)
        return True, ""

    def existing_record(
        self, case_id: str
    ) -> dict[str, Any] | None:
        for record in self.state["cases"]:
            if record.get("case_id") == case_id:
                return record
        return None

    def run_case(
        self,
        index: int,
        case: ServiceCase,
        record: dict[str, Any] | None = None,
    ) -> None:
        attempt_number = 1
        case_log_root = self.log_root / case.case_id
        case_result_root = self.result_root / case.case_id
        if record is not None:
            attempts = record.setdefault("attempts", [])
            if not isinstance(attempts, list):
                raise ValueError(
                    f"{case.case_id}: attempts is not a list"
                )
            if (
                not attempts
                and isinstance(record.get("log_dir"), str)
                and Path(record["log_dir"]).exists()
            ):
                attempts.append(
                    {
                        key: value
                        for key, value in record.items()
                        if key != "attempts"
                    }
                )
                attempts[-1].setdefault(
                    "server_run_id",
                    f"{self.suite_id}:{case.case_id}",
                )
            attempt_number = len(attempts) + 1
        else:
            attempts = []
        if attempt_number == 1:
            case_log_dir = case_log_root
            case_result_dir = case_result_root
        else:
            case_log_dir = case_log_root / f"attempt_{attempt_number:02d}"
            case_result_dir = (
                case_result_root / f"attempt_{attempt_number:02d}"
            )
        case_log_dir.mkdir(parents=True, exist_ok=False)
        case_result_dir.mkdir(parents=True, exist_ok=False)
        current_values: dict[str, Any] = {
            "index": index,
            "case_id": case.case_id,
            "task_kind": case.task_kind,
            "world_size": case.world_size,
            "parallel_strategy": case.parallel_strategy,
            "attempt": attempt_number,
            "status": "running",
            "started_at": utc_now(),
            "completed_at": None,
            "log_dir": str(case_log_dir),
            "result_dir": str(case_result_dir),
            "startup_seconds": None,
            "warmup_exit_code": None,
            "warmup_result": None,
            "benchmark_exit_code": None,
            "benchmark_result": None,
            "dit_validation": None,
            "peak_device_memory_mb": {},
            "peak_hbm_mb": {},
            "diagnostic_warnings": [],
            "server_shutdown": None,
            "device_released": False,
            "npu_released": False,
            "error": None,
        }
        if record is None:
            record = current_values
            record["attempts"] = attempts
            self.state["cases"].append(record)
        else:
            record.update(current_values)
        attempt_record: dict[str, Any] = {
            "attempt": attempt_number,
            "server_run_id": (
                f"{self.suite_id}:{case.case_id}:a{attempt_number}"
            ),
            "status": "running",
            "started_at": record["started_at"],
            "completed_at": None,
            "log_dir": str(case_log_dir),
            "result_dir": str(case_result_dir),
            "error": None,
        }
        attempts.append(attempt_record)
        self.persist()
        sampler: DeviceSampler | None = None
        baseline_memory_mb: dict[int, int] = {}
        device_prefix = (
            "mlu" if self.platform_name == "mlu" else "npu"
        )

        try:
            for prior_attempt in attempts[:-1]:
                if not isinstance(prior_attempt, dict):
                    continue
                prior_run_id = prior_attempt.get("server_run_id")
                if not isinstance(prior_run_id, str) or not prior_run_id:
                    continue
                stale_pids = self.owned_server_pids(prior_run_id)
                if not stale_pids:
                    continue
                self.log(
                    f"[{index}] {case.case_id}: cleaning stale owned "
                    f"RUN_ID={prior_run_id}, pids={stale_pids}"
                )
                self.signal_owned_server(prior_run_id, signal.SIGTERM)
                if not self.wait_owned_server_exit(prior_run_id, 60):
                    self.signal_owned_server(
                        prior_run_id, signal.SIGKILL
                    )
                    self.wait_owned_server_exit(prior_run_id, 30)
                residual = self.owned_server_pids(prior_run_id)
                if residual:
                    raise RuntimeError(
                        "stale owned service processes did not exit: "
                        f"{residual}"
                    )
            self.log(
                f"[{index}] {case.case_id}: checking accelerator idle"
            )
            baseline_memory_mb = self.ensure_device_idle(
                case_log_dir / f"{device_prefix}_before.txt"
            )
            record["device_memory_before_mb"] = {
                str(device): value
                for device, value in sorted(
                    baseline_memory_mb.items()
                )
            }
            _, startup_seconds = self.start_server(
                case,
                case_log_dir,
                case_result_dir,
                attempt_number,
            )
            record["startup_seconds"] = startup_seconds
            self.log(
                f"[{index}] {case.case_id}: service ready in "
                f"{startup_seconds:.2f}s"
            )
            self.snapshot_device(
                case_log_dir / f"{device_prefix}_ready.txt"
            )
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
            (
                warmup_valid,
                warmup_reason,
                warmup_result,
            ) = self.benchmark_phase_is_valid(
                case,
                warmup_dir.parent,
                "warmup",
                self.arguments.warmup_count,
            )
            record["warmup_result"] = warmup_result
            if not warmup_valid:
                raise RuntimeError(warmup_reason)
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
            if sampler is None:
                sampler = DeviceSampler(
                    case_log_dir / f"{device_prefix}_samples.csv",
                    self.device_sample_interval_seconds,
                    self.platform_paths,
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
            record["peak_device_memory_mb"] = {
                str(device): value
                for device, value in sorted(
                    sampler.peak_device_memory_mb.items()
                )
            }
            if self.platform_name == "ascend_npu":
                record["peak_hbm_mb"] = dict(
                    record["peak_device_memory_mb"]
                )
            if sampler.errors:
                record["diagnostic_warnings"].append(
                    "device sampler errors: "
                    f"{len(sampler.errors)}"
                )
            if (
                self.platform_name == "mlu"
                and self.suite_kind == "formal"
                and {
                    int(device)
                    for device in record["peak_device_memory_mb"]
                    if str(device).isdigit()
                }
                != set(range(8))
            ):
                raise RuntimeError(
                    "formal MLU case has no complete 0..7 device-memory "
                    "sample"
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
            (
                measure_valid,
                measure_reason,
                measure_result,
            ) = self.benchmark_phase_is_valid(
                case,
                measure_dir.parent,
                "measure",
                self.arguments.sample_count,
            )
            record["benchmark_result"] = measure_result
            if not measure_valid:
                raise RuntimeError(measure_reason)
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
            if (
                record["status"] == "passed"
                and isinstance(record["benchmark_result"], dict)
            ):
                record["dit_validation"] = self.validate_dit_evidence(
                    case,
                    case_log_dir / "server.log",
                    record["benchmark_result"],
                )
                if record["dit_validation"]["status"] != "passed":
                    dit_error = (
                        "single-step DiT evidence is incomplete: "
                        + "; ".join(
                            record["dit_validation"]["errors"][:3]
                        )
                    )
                    if (
                        self.suite_kind == "formal"
                        and self.infer_profile_level != "0"
                    ):
                        record["status"] = "failed"
                        record["error"] = dit_error
                        self.log(
                            f"[{index}] {case.case_id}: failed: "
                            f"{dit_error}"
                        )
                    elif self.infer_profile_level != "0":
                        record["diagnostic_warnings"].append(dit_error)
            device_after_path = (
                case_log_dir / f"{device_prefix}_after.txt"
            )
            if record["server_shutdown"]["attempted"]:
                record["device_released"] = self.wait_device_idle(
                    device_after_path,
                    baseline_memory_mb,
                )
            else:
                try:
                    process_ids, memory_used = self.snapshot_device(
                        device_after_path
                    )
                    memory_released = self.memory_is_released(
                        memory_used,
                        baseline_memory_mb,
                    )
                    record["device_released"] = (
                        not process_ids and memory_released
                    )
                except RuntimeError as exception:
                    record["diagnostic_warnings"].append(
                        f"final device snapshot: {exception}"
                    )
                    record["device_released"] = False
            record["npu_released"] = record["device_released"]
            if not record["device_released"]:
                record["status"] = "failed"
                release_error = (
                    "accelerator processes or memory did not release "
                    "after service"
                )
                record["error"] = (
                    f"{record['error']}; {release_error}"
                    if record["error"]
                    else release_error
                )
            record["completed_at"] = utc_now()
            attempt_record.update(
                {
                    "status": record["status"],
                    "completed_at": record["completed_at"],
                    "startup_seconds": record["startup_seconds"],
                    "warmup_exit_code": record["warmup_exit_code"],
                    "warmup_result": record["warmup_result"],
                    "benchmark_exit_code": record[
                        "benchmark_exit_code"
                    ],
                    "benchmark_result": record["benchmark_result"],
                    "dit_validation": record["dit_validation"],
                    "peak_device_memory_mb": record[
                        "peak_device_memory_mb"
                    ],
                    "server_shutdown": record["server_shutdown"],
                    "device_released": record["device_released"],
                    "error": record["error"],
                }
            )
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

        if self.resume:
            if not self.log_root.is_dir():
                raise FileNotFoundError(
                    f"resume log directory does not exist: {self.log_root}"
                )
            if not self.result_root.is_dir():
                raise FileNotFoundError(
                    "resume result directory does not exist: "
                    f"{self.result_root}"
                )
            self.acquire_lock()
            self.load_resume_state()
            planned_case_ids = self.state.get("selected_case_ids")
            if (
                not isinstance(planned_case_ids, list)
                or not all(
                    isinstance(case_id, str)
                    for case_id in planned_case_ids
                )
            ):
                planned_case_ids = []
            if not planned_case_ids and self.manifest_path.is_file():
                try:
                    manifest = json.loads(
                        self.manifest_path.read_text(encoding="utf-8")
                    )
                    manifest_cases = manifest.get("cases", [])
                    planned_case_ids = [
                        item["case_id"]
                        for item in manifest_cases
                        if isinstance(item, dict)
                        and isinstance(item.get("case_id"), str)
                    ]
                except (OSError, json.JSONDecodeError):
                    planned_case_ids = []
            state_case_ids = {
                record.get("case_id")
                for record in self.state.get("cases", [])
                if isinstance(record, dict)
            }
            if not planned_case_ids:
                planned_case_ids = [
                    case.case_id
                    for case in CASES
                    if case.case_id in state_case_ids
                ]
            self.state["selected_case_ids"] = planned_case_ids
            if not self.arguments.only:
                cases = [
                    case
                    for case in CASES
                    if case.case_id in set(planned_case_ids)
                ]
            unknown_resume_cases = {
                case.case_id for case in cases
            } - set(planned_case_ids)
            if unknown_resume_cases:
                raise ValueError(
                    "resume requested cases not present in suite plan: "
                    f"{sorted(unknown_resume_cases)}"
                )
            # Revalidate after restoring the original workload parameters.
            self.validate(cases)
            self.validate_resume_manifest(cases)
            self.persist()
            self.publish_controller_pid()
        else:
            # Detached launchers may create the log directory first so their
            # controller log can be redirected before this process starts.
            self.log_root.mkdir(parents=True, exist_ok=True)
            self.result_root.mkdir(parents=True, exist_ok=False)
            self.acquire_lock()
            self.state["selected_case_ids"] = [
                case.case_id for case in cases
            ]
            self.create_manifest(cases)
            self.state["status"] = "running"
            self.persist()
            self.publish_controller_pid()
        previous_handlers = {
            signum: signal.signal(signum, self.handle_signal)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        exit_code = 0
        try:
            for index, case in enumerate(cases, 1):
                record = self.existing_record(case.case_id)
                if record is not None:
                    valid, reason = self.passed_record_is_valid(
                        case,
                        record,
                    )
                    if valid:
                        self.log(
                            f"[{index}] {case.case_id}: "
                            "skip validated passed case"
                        )
                        continue
                    if record.get("status") == "passed":
                        self.log(
                            f"[{index}] {case.case_id}: "
                            f"retry stale passed case: {reason}"
                        )
                self.run_case(index, case, record)
                record = self.existing_record(case.case_id)
                server_shutdown = (
                    record.get("server_shutdown")
                    if isinstance(record, dict)
                    else None
                )
                cleanup_unsafe = (
                    not isinstance(record, dict)
                    or record.get("device_released") is not True
                    or not isinstance(server_shutdown, dict)
                    or (
                        isinstance(server_shutdown, dict)
                        and (
                            server_shutdown.get("cleanup_complete") is not True
                            or bool(server_shutdown.get("residual_pids"))
                            or server_shutdown.get("fallback_refused")
                            is True
                        )
                    )
                )
                if cleanup_unsafe:
                    exit_code = 1
                    self.log(
                        f"[{index}] {case.case_id}: stopping suite because "
                        "accelerator cleanup was not proven complete"
                    )
                    break
                if record is None or record["status"] != "passed":
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
            records_by_id = {
                record.get("case_id"): record
                for record in self.state["cases"]
                if isinstance(record, dict)
            }
            planned_case_ids = self.state.get(
                "selected_case_ids", []
            )
            planned_records = [
                records_by_id.get(case_id)
                for case_id in planned_case_ids
            ]
            statuses = {
                record.get("status")
                for record in planned_records
                if isinstance(record, dict)
            }
            if self.interrupted:
                self.state["status"] = "interrupted"
            elif any(
                record is None
                or record.get("status") in {"pending", "running"}
                for record in planned_records
            ):
                self.state["status"] = "incomplete"
            elif planned_records and statuses == {"passed"}:
                self.state["status"] = "passed"
            else:
                self.state["status"] = "failed"
            self.state["completed_at"] = utc_now()
            self.persist()
            self.log(
                f"suite completed with status {self.state['status']}"
            )
            if (
                exit_code == 0
                and self.state["status"] != "passed"
            ):
                exit_code = 1
        return exit_code


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start each distributed service on Ascend or MLU, warm it "
            "up, measure samples and archive all logs/results."
        )
    )
    parser.add_argument(
        "--platform",
        choices=tuple(PLATFORMS),
        default="ascend_npu",
        help="Accelerator platform (default: ascend_npu)",
    )
    parser.add_argument("--suite-id", default="")
    parser.add_argument(
        "--resume",
        metavar="SUITE_ID",
        default="",
        help=(
            "Resume an existing suite and skip passed cases only after "
            "revalidating every output artifact"
        ),
    )
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
        "--device-sample-interval-seconds",
        "--npu-sample-interval-seconds",
        dest="device_sample_interval_seconds",
        type=float,
        default=1.0,
        help="Device monitor sampling interval (default: 1 second)",
    )
    parser.add_argument(
        "--memory-release-tolerance-mb",
        type=int,
        default=256,
        help=(
            "Allowed post-service memory above the idle baseline per "
            "device"
        ),
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
