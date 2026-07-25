#!/usr/bin/env python3
"""Run every Ascend offline-inference example sequentially and archive the suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_PATH = Path("/data/wushuo1/Lightx2v-Platform-Run-Examples")
INFER_ROOT = REPO_PATH / "scripts" / "ascend" / "infer"
LOG_ROOT = REPO_PATH / "logs" / "ascend_npu" / "infer"

# Low-cost smoke cases first, then keep the same model together. Riskier,
# longer 720p/offload and LTX-2.3 cases run later.
CASES: tuple[tuple[str, str], ...] = (
    (
        "z_image_turbo_t2i_1664x928",
        "single/run_z_image_turbo_t2i_1664x928.sh",
    ),
    (
        "z_image_turbo_t2i_1664x928_sp2",
        "dist_2/run_z_image_turbo_t2i_1664x928_sp2.sh",
    ),
    (
        "wan21_1_3b_self_forcing_t2v_480p_81f",
        "single/run_wan21_1_3b_self_forcing_t2v_480p_81f.sh",
    ),
    (
        "wan21_1_3b_t2v_480p_81f",
        "single/run_wan21_1_3b_t2v_480p_81f.sh",
    ),
    (
        "wan21_1_3b_t2v_480p_81f_cfg2_sp4",
        "dist_8/run_wan21_1_3b_t2v_480p_81f_cfg2_sp4.sh",
    ),
    (
        "longcat_image_t2i_1344x768",
        "single/run_longcat_image_t2i_1344x768.sh",
    ),
    (
        "longcat_image_t2i_1344x768_cfg2_sp4",
        "dist_8/run_longcat_image_t2i_1344x768_cfg2_sp4.sh",
    ),
    (
        "qwen_image_2512_t2i_1664x928",
        "single/run_qwen_image_2512_t2i_1664x928.sh",
    ),
    (
        "qwen_image_2512_t2i_1664x928_cfg2_sp4",
        "dist_8/run_qwen_image_2512_t2i_1664x928_cfg2_sp4.sh",
    ),
    (
        "flux2_dev_t2i_1344x768",
        "single/run_flux2_dev_t2i_1344x768.sh",
    ),
    (
        "flux2_dev_t2i_1344x768_tp8",
        "dist_8/run_flux2_dev_t2i_1344x768_tp8.sh",
    ),
    (
        "wan22_moe_a14b_t2v_480p_81f",
        "single/run_wan22_moe_a14b_t2v_480p_81f.sh",
    ),
    (
        "wan22_moe_a14b_t2v_480p_81f_cfg2_sp4",
        "dist_8/run_wan22_moe_a14b_t2v_480p_81f_cfg2_sp4.sh",
    ),
    (
        "wan22_moe_a14b_t2v_480p_81f_tp8",
        "dist_8/run_wan22_moe_a14b_t2v_480p_81f_tp8.sh",
    ),
    (
        "wan22_moe_a14b_t2v_720p_81f",
        "single/run_wan22_moe_a14b_t2v_720p_81f.sh",
    ),
    (
        "wan22_moe_a14b_t2v_720p_81f_cfg2_sp4",
        "dist_8/run_wan22_moe_a14b_t2v_720p_81f_cfg2_sp4.sh",
    ),
    (
        "wan22_moe_a14b_t2v_720p_81f_tp8",
        "dist_8/run_wan22_moe_a14b_t2v_720p_81f_tp8.sh",
    ),
    (
        "hunyuan_video_15_t2v_480p_121f",
        "single/run_hunyuan_video_15_t2v_480p_121f.sh",
    ),
    (
        "hunyuan_video_15_t2v_480p_121f_cfg2_sp4",
        "dist_8/run_hunyuan_video_15_t2v_480p_121f_cfg2_sp4.sh",
    ),
    (
        "hunyuan_video_15_t2v_720p_121f",
        "single/run_hunyuan_video_15_t2v_720p_121f.sh",
    ),
    (
        "hunyuan_video_15_t2v_720p_121f_cfg2_sp4",
        "dist_8/run_hunyuan_video_15_t2v_720p_121f_cfg2_sp4.sh",
    ),
    (
        "ltx2_3_22b_dev_s2v_768x512_241f",
        "single/run_ltx2_3_22b_dev_s2v_768x512_241f.sh",
    ),
    (
        "ltx2_3_22b_dev_s2v_768x512_241f_sp8",
        "dist_8/run_ltx2_3_22b_dev_s2v_768x512_241f_sp8.sh",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def npu_process_ids(output: str) -> set[int]:
    """Extract only PIDs from the process table at the end of npu-smi info."""
    in_process_table = False
    process_ids: set[int] = set()
    for line in output.splitlines():
        if "Process id" in line and "Process name" in line:
            in_process_table = True
            continue
        if not in_process_table:
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) >= 4 and fields[2].isdigit():
            process_ids.add(int(fields[2]))
    return process_ids


def parse_npu_snapshot(output: str) -> tuple[set[int], dict[int, int]]:
    """Validate a complete npu-smi report and return process/HBM state."""
    device_ids: set[int] = set()
    process_section_device_ids: set[int] = set()
    hbm_used_mb: dict[int, int] = {}
    current_device: int | None = None
    in_process_table = False

    for line in output.splitlines():
        device_match = re.match(
            r"^\|\s*(\d+)\s+\S+\s+\|\s*(?:OK|Warning|Alarm|Critical)",
            line,
        )
        if device_match:
            current_device = int(device_match.group(1))
            device_ids.add(current_device)
            continue
        if current_device is not None:
            hbm_match = re.search(r"(\d+)\s*/\s*(\d+)\s*\|\s*$", line)
            if hbm_match:
                hbm_used_mb[current_device] = int(hbm_match.group(1))
                current_device = None

        if "Process id" in line and "Process name" in line:
            in_process_table = True
            continue
        if not in_process_table:
            continue
        empty_match = re.search(
            r"No running processes found in NPU\s+(\d+)", line
        )
        if empty_match:
            process_section_device_ids.add(int(empty_match.group(1)))
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) >= 4 and fields[2].isdigit():
            npu_fields = fields[1].split()
            if npu_fields and npu_fields[0].isdigit():
                process_section_device_ids.add(int(npu_fields[0]))

    if not in_process_table:
        raise ValueError("npu-smi process table header was not found")
    if not device_ids or set(hbm_used_mb) != device_ids:
        raise ValueError(
            "npu-smi device/HBM table was incomplete: "
            f"devices={sorted(device_ids)}, hbm={sorted(hbm_used_mb)}"
        )
    if process_section_device_ids != device_ids:
        raise ValueError(
            "npu-smi process table was incomplete: "
            f"devices={sorted(device_ids)}, "
            f"process_sections={sorted(process_section_device_ids)}"
        )
    return npu_process_ids(output), hbm_used_mb


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    session_id: int
    start_time_ticks: int
    run_id: str
    command: str


class SuiteInterrupted(RuntimeError):
    """Raised by the lightweight signal handler to enter orderly shutdown."""


def process_identity(pid: int) -> ProcessIdentity | None:
    proc_path = Path("/proc") / str(pid)
    try:
        stat_text = (proc_path / "stat").read_text(encoding="utf-8")
        close_parenthesis = stat_text.rfind(")")
        fields_after_command = stat_text[close_parenthesis + 2 :].split()
        session_id = int(fields_after_command[3])
        start_time_ticks = int(fields_after_command[19])
        environment = (proc_path / "environ").read_bytes().split(b"\0")
        command = (proc_path / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except (OSError, ValueError, IndexError):
        return None

    run_id = ""
    for item in environment:
        if item.startswith(b"RUN_ID="):
            run_id = item.removeprefix(b"RUN_ID=").decode(
                "utf-8", errors="replace"
            )
            break
    return ProcessIdentity(
        pid=pid,
        session_id=session_id,
        start_time_ticks=start_time_ticks,
        run_id=run_id,
        command=command,
    )


def session_members(session_id: int) -> list[ProcessIdentity]:
    members: list[ProcessIdentity] = []
    for proc_path in Path("/proc").iterdir():
        if not proc_path.name.isdigit():
            continue
        identity = process_identity(int(proc_path.name))
        if identity is not None and identity.session_id == session_id:
            members.append(identity)
    return members


class InferSuite:
    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments
        self.suite_id = arguments.suite_id or (
            f"suite_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_p{os.getpid()}"
        )
        self.suite_root = LOG_ROOT / "suites" / self.suite_id
        self.suite_root.mkdir(parents=True, exist_ok=False)
        self.suite_log_path = self.suite_root / "suite.log"
        self.state_path = self.suite_root / "suite.json"
        self.report_path = self.suite_root / "report.md"
        self.current_process: subprocess.Popen[bytes] | None = None
        self.current_session_id: int | None = None
        self.current_run_id = ""
        self.interrupted = False
        self._signal_interruptible = False
        self._shutdown_in_progress = False
        self._received_signal_number: int | None = None
        self._received_signal_count = 0
        self._shutdown_cleanup_started = False
        self._final_npu_check_started = False
        self.baseline_hbm_used_mb: dict[int, int] = {}
        self.state: dict[str, Any] = {
            "suite_id": self.suite_id,
            "status": "running",
            "started_at_utc": utc_now(),
            "finished_at_utc": None,
            "repo_path": str(REPO_PATH),
            "case_timeout_seconds": arguments.case_timeout_seconds,
            "idle_timeout_seconds": arguments.idle_timeout_seconds,
            "cases": [],
            "shutdown": {
                "required": False,
                "trigger": None,
                "status": "not_required",
                "succeeded": True,
                "signal": None,
                "cleanup": {
                    "attempted": False,
                    "required": False,
                    "session_id": None,
                    "run_id": "",
                    "terminate_owned_session_return": None,
                    "succeeded": True,
                    "error": "",
                },
                "final_npu_check": {
                    "attempted": False,
                    "query_succeeded": None,
                    "idle": None,
                    "process_ids": [],
                    "hbm_used_mb": {},
                    "hbm_baseline_mb": {},
                    "hbm_within_baseline": None,
                    "high_hbm_mb": {},
                    "process_termination_attempted": False,
                    "error": "",
                },
                "errors": [],
            },
        }
        self.save_state()

    def log(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with self.suite_log_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(f"{line}\n")
            file_obj.flush()

    def save_state(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def snapshot_npu(self, label: str) -> tuple[set[int], dict[int, int], str]:
        last_error = ""
        for query_attempt in range(1, 4):
            snapshot_path = (
                self.suite_root / f"npu_{label}_query{query_attempt}.txt"
            )
            try:
                completed = subprocess.run(
                    ["npu-smi", "info"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                output = completed.stdout + completed.stderr
                snapshot_path.write_text(output, encoding="utf-8")
                if (
                    completed.returncode != 0
                    or "dcmi module initialize failed" in output
                ):
                    raise RuntimeError(
                        f"npu-smi exited with {completed.returncode}"
                    )
                process_ids, hbm_used_mb = parse_npu_snapshot(output)
                return process_ids, hbm_used_mb, output
            except (
                OSError,
                RuntimeError,
                ValueError,
                subprocess.TimeoutExpired,
            ) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self.log(
                    f"[NPU] query retry {query_attempt}/3 at {label}: "
                    f"{last_error}"
                )
                if query_attempt < 3:
                    time.sleep(2)
        raise RuntimeError(
            f"npu-smi failed for {label} after 3 attempts: {last_error}"
        )

    @staticmethod
    def identity_is_unchanged(identity: ProcessIdentity, run_id: str) -> bool:
        current = process_identity(identity.pid)
        return (
            current is not None
            and current.session_id == identity.session_id
            and current.start_time_ticks == identity.start_time_ticks
            and current.run_id == run_id
        )

    def terminate_owned_session(self, session_id: int, run_id: str) -> bool:
        self.reap_current_process_if_exited(session_id)
        members = session_members(session_id)
        if not members:
            return True
        foreign = [member for member in members if member.run_id != run_id]
        if foreign:
            self.log(
                "[SAFETY] Refusing cleanup: session "
                f"{session_id} contains process(es) without RUN_ID={run_id}: "
                + ", ".join(
                    f"{member.pid}:{member.command[:80]}" for member in foreign
                )
            )
            return False

        self.log(
            f"[Cleanup] TERM session={session_id}, run_id={run_id}, "
            f"pids={[member.pid for member in members]}"
        )
        for member in sorted(members, key=lambda item: item.pid, reverse=True):
            if self.identity_is_unchanged(member, run_id):
                try:
                    os.kill(member.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            self.reap_current_process_if_exited(session_id)
            if not session_members(session_id):
                return True
            time.sleep(1)

        self.reap_current_process_if_exited(session_id)
        remaining = session_members(session_id)
        foreign = [member for member in remaining if member.run_id != run_id]
        if foreign:
            self.log(
                "[SAFETY] Refusing KILL after session membership changed: "
                + ", ".join(str(member.pid) for member in foreign)
            )
            return False
        self.log(
            f"[Cleanup] KILL session={session_id}, run_id={run_id}, "
            f"pids={[member.pid for member in remaining]}"
        )
        for member in remaining:
            if self.identity_is_unchanged(member, run_id):
                try:
                    os.kill(member.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        time.sleep(2)
        self.reap_current_process_if_exited(session_id)
        return not session_members(session_id)

    def reap_current_process_if_exited(self, session_id: int) -> None:
        process = self.current_process
        if process is not None and process.pid == session_id:
            process.poll()

    def wait_for_session_exit(self, session_id: int, run_id: str) -> bool:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if not session_members(session_id):
                return True
            time.sleep(0.5)
        return self.terminate_owned_session(session_id, run_id)

    def wait_for_npu_idle(
        self,
        label: str,
        owned_session_id: int | None = None,
        owned_run_id: str = "",
    ) -> bool:
        deadline = time.monotonic() + self.arguments.idle_timeout_seconds
        reported_pids: set[int] | None = None
        reported_hbm: dict[int, int] | None = None
        attempt = 0
        while True:
            if self.interrupted:
                return False
            attempt += 1
            process_ids, hbm_used_mb, _ = self.snapshot_npu(
                f"{label}_{attempt:03d}"
            )
            if not process_ids:
                if not self.baseline_hbm_used_mb:
                    self.baseline_hbm_used_mb = dict(hbm_used_mb)
                    self.state["npu_idle_hbm_baseline_mb"] = {
                        str(device): used
                        for device, used in sorted(hbm_used_mb.items())
                    }
                    self.save_state()
                high_hbm = {
                    device: used
                    for device, used in hbm_used_mb.items()
                    if used
                    > self.baseline_hbm_used_mb.get(device, used) + 1024
                }
                if not high_hbm:
                    self.log(
                        f"[NPU] idle at {label}; HBM(MB)="
                        f"{dict(sorted(hbm_used_mb.items()))}"
                    )
                    return True
                if hbm_used_mb != reported_hbm:
                    self.log(
                        f"[NPU] no process but HBM has not returned to baseline "
                        f"at {label}: current={hbm_used_mb}, "
                        f"baseline={self.baseline_hbm_used_mb}"
                    )
                    reported_hbm = dict(hbm_used_mb)

            if process_ids != reported_pids:
                details = []
                for pid in sorted(process_ids):
                    identity = process_identity(pid)
                    if identity is None:
                        details.append(f"{pid}:unavailable")
                    else:
                        details.append(
                            f"{pid}:sid={identity.session_id}:run_id={identity.run_id}:"
                            f"{identity.command[:100]}"
                        )
                self.log(f"[NPU] busy at {label}: " + "; ".join(details))
                reported_pids = process_ids

            if owned_session_id is not None and owned_run_id:
                identities = [process_identity(pid) for pid in process_ids]
                if identities and all(
                    identity is not None
                    and identity.session_id == owned_session_id
                    and identity.run_id == owned_run_id
                    for identity in identities
                ):
                    if not self.terminate_owned_session(
                        owned_session_id, owned_run_id
                    ):
                        return False
                    continue

            if time.monotonic() >= deadline:
                self.log(
                    f"[SAFETY] NPU remained occupied at {label}; "
                    "unknown processes were not terminated"
                )
                return False
            time.sleep(10)

    def record_signal_state(self) -> None:
        if self._received_signal_number is None:
            return
        try:
            signal_name = signal.Signals(self._received_signal_number).name
        except ValueError:
            signal_name = str(self._received_signal_number)
        self.state["shutdown"]["signal"] = {
            "number": self._received_signal_number,
            "name": signal_name,
            "received_count": self._received_signal_count,
        }

    def handle_signal(self, signal_number: int, _frame: Any) -> None:
        """Mark interruption and unwind once; never clean up reentrantly."""
        self.interrupted = True
        self._received_signal_count += 1
        if self._received_signal_number is None:
            self._received_signal_number = signal_number
        if self._shutdown_in_progress or not self._signal_interruptible:
            return
        self._signal_interruptible = False
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = str(signal_number)
        raise SuiteInterrupted(f"received {signal_name}")

    def cleanup_current_session_once(self, trigger: str) -> bool:
        """Clean the exact owned session once and persist the return value."""
        cleanup = self.state["shutdown"]["cleanup"]
        if self._shutdown_cleanup_started:
            return cleanup.get("succeeded") is True
        self._shutdown_cleanup_started = True

        session_id = self.current_session_id
        run_id = self.current_run_id
        required = session_id is not None and bool(run_id)
        cleanup.update(
            {
                "attempted": True,
                "attempted_at_utc": utc_now(),
                "trigger": trigger,
                "required": required,
                "session_id": session_id,
                "run_id": run_id,
                "terminate_owned_session_return": None,
                "succeeded": not required,
                "error": "",
            }
        )
        self.save_state()
        if not required:
            return True

        try:
            cleanup_return = self.terminate_owned_session(session_id, run_id)
        except Exception as exc:
            cleanup["succeeded"] = False
            cleanup["error"] = f"{type(exc).__name__}: {exc}"
            self.log(
                f"[Shutdown] owned-session cleanup raised at {trigger}: "
                f"{cleanup['error']}"
            )
        else:
            cleanup["terminate_owned_session_return"] = cleanup_return
            cleanup["succeeded"] = cleanup_return
            if not cleanup_return:
                cleanup["error"] = (
                    "terminate_owned_session returned False; no unowned "
                    "process was terminated"
                )
                self.log(
                    f"[Shutdown] owned-session cleanup failed at {trigger}: "
                    f"session={session_id}, run_id={run_id}"
                )
        cleanup["finished_at_utc"] = utc_now()
        self.save_state()
        return cleanup["succeeded"] is True

    def record_final_npu_check(self, label: str) -> bool:
        """Perform one interrupt-independent, read-only final NPU/HBM check."""
        result = self.state["shutdown"]["final_npu_check"]
        if self._final_npu_check_started:
            return result.get("idle") is True
        self._final_npu_check_started = True
        result.update(
            {
                "attempted": True,
                "attempted_at_utc": utc_now(),
                "label": label,
                "query_succeeded": False,
                "idle": False,
                "process_ids": [],
                "hbm_used_mb": {},
                "hbm_baseline_mb": {
                    str(device): used
                    for device, used in sorted(
                        self.baseline_hbm_used_mb.items()
                    )
                },
                "hbm_within_baseline": None,
                "high_hbm_mb": {},
                "process_termination_attempted": False,
                "error": "",
            }
        )
        self.save_state()

        try:
            process_ids, hbm_used_mb, _ = self.snapshot_npu(label)
            baseline_devices = set(self.baseline_hbm_used_mb)
            current_devices = set(hbm_used_mb)
            baseline_complete = (
                bool(self.baseline_hbm_used_mb)
                and baseline_devices == current_devices
            )
            high_hbm = (
                {
                    device: used
                    for device, used in hbm_used_mb.items()
                    if used
                    > self.baseline_hbm_used_mb[device] + 1024
                }
                if baseline_complete
                else {}
            )
            hbm_within_baseline = baseline_complete and not high_hbm
            idle = not process_ids and hbm_within_baseline

            process_details: list[dict[str, Any]] = []
            for pid in sorted(process_ids):
                identity = process_identity(pid)
                if identity is None:
                    process_details.append(
                        {"pid": pid, "identity": "unavailable"}
                    )
                else:
                    process_details.append(
                        {
                            "pid": pid,
                            "session_id": identity.session_id,
                            "run_id": identity.run_id,
                            "command": identity.command,
                        }
                    )

            result.update(
                {
                    "query_succeeded": True,
                    "idle": idle,
                    "process_ids": sorted(process_ids),
                    "processes": process_details,
                    "hbm_used_mb": {
                        str(device): used
                        for device, used in sorted(hbm_used_mb.items())
                    },
                    "hbm_baseline_mb": {
                        str(device): used
                        for device, used in sorted(
                            self.baseline_hbm_used_mb.items()
                        )
                    },
                    "hbm_within_baseline": hbm_within_baseline,
                    "high_hbm_mb": {
                        str(device): used
                        for device, used in sorted(high_hbm.items())
                    },
                }
            )
            if process_ids:
                result["error"] = (
                    "NPU process table was not idle; the final check is "
                    "read-only and did not terminate any process"
                )
            elif not baseline_complete:
                result["error"] = (
                    "HBM baseline was unavailable or did not cover the same "
                    "devices"
                )
            elif high_hbm:
                result["error"] = (
                    "NPU process table was idle but HBM had not returned "
                    "within 1024 MB of baseline"
                )
            if idle:
                self.log(
                    f"[Shutdown] final NPU check passed at {label}; "
                    f"HBM(MB)={dict(sorted(hbm_used_mb.items()))}"
                )
            else:
                self.log(
                    f"[Shutdown] final NPU check failed at {label}: "
                    f"{result['error']}"
                )
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            self.log(
                f"[Shutdown] final NPU query failed at {label}: "
                f"{result['error']}"
            )
        result["finished_at_utc"] = utc_now()
        self.save_state()
        return result.get("idle") is True

    def finalize_shutdown(self, trigger: str) -> bool:
        shutdown = self.state["shutdown"]
        shutdown.update(
            {
                "required": True,
                "trigger": trigger,
                "status": "running",
                "succeeded": False,
                "errors": [],
            }
        )
        self.record_signal_state()
        self.save_state()

        cleanup_succeeded = self.cleanup_current_session_once(trigger)
        idle_succeeded = self.record_final_npu_check("shutdown_final")
        errors: list[str] = []
        cleanup_error = shutdown["cleanup"].get("error", "")
        idle_error = shutdown["final_npu_check"].get("error", "")
        if not cleanup_succeeded:
            errors.append(f"cleanup: {cleanup_error or 'failed'}")
        if not idle_succeeded:
            errors.append(f"final_npu_check: {idle_error or 'not idle'}")
        shutdown.update(
            {
                "status": "passed" if not errors else "failed",
                "succeeded": not errors,
                "errors": errors,
                "finished_at_utc": utc_now(),
            }
        )
        self.record_signal_state()
        self.save_state()
        return not errors

    def selected_cases(self) -> list[tuple[int, str, str]]:
        only = set(self.arguments.only or [])
        cases = [
            (index, case_id, relative_script)
            for index, (case_id, relative_script) in enumerate(CASES, start=1)
            if not only or case_id in only
        ]
        missing = only - {case_id for _, case_id, _ in cases}
        if missing:
            raise ValueError(f"unknown --only case(s): {sorted(missing)}")
        return cases

    def run_case(self, index: int, case_id: str, relative_script: str) -> None:
        script_path = INFER_ROOT / relative_script
        run_id = f"{self.suite_id}_{index:02d}"
        controller_log = self.suite_root / f"{index:02d}_{case_id}_controller.log"
        run_record_path = LOG_ROOT / case_id / run_id / "run.json"
        started_epoch = time.time()
        entry: dict[str, Any] = {
            "index": index,
            "case_id": case_id,
            "script": str(script_path),
            "run_id": run_id,
            "status": "running",
            "exit_code": None,
            "started_at_utc": utc_now(),
            "finished_at_utc": None,
            "duration_seconds": None,
            "session_id": None,
            "controller_log": str(controller_log),
            "run_record": str(run_record_path),
            "result": None,
            "run_record_status": None,
            "error": "",
        }
        self.state["cases"].append(entry)
        self.save_state()

        self.log(
            f"[Case {index:02d}/{len(CASES)}] START {case_id} "
            f"script={relative_script} run_id={run_id}"
        )
        if not self.wait_for_npu_idle(f"{index:02d}_before"):
            entry["status"] = "blocked"
            entry["error"] = "NPU was occupied by an unowned process"
            entry["finished_at_utc"] = utc_now()
            entry["duration_seconds"] = round(time.time() - started_epoch, 3)
            self.save_state()
            raise RuntimeError(entry["error"])

        environment = os.environ.copy()
        environment.update(
            {
                "RUN_ID": run_id,
                "LIGHTX2V_INFER_SUITE_ID": self.suite_id,
                "PYTHONUNBUFFERED": "1",
            }
        )
        timed_out = False
        with controller_log.open("w", encoding="utf-8") as controller_file:
            process = subprocess.Popen(
                ["bash", str(script_path)],
                cwd=REPO_PATH,
                env=environment,
                stdout=controller_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.current_process = process
            self.current_session_id = process.pid
            self.current_run_id = run_id
            entry["session_id"] = process.pid
            self.save_state()
            try:
                exit_code = process.wait(
                    timeout=self.arguments.case_timeout_seconds
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                self.log(
                    f"[Case {index:02d}] TIMEOUT after "
                    f"{self.arguments.case_timeout_seconds}s"
                )
                self.terminate_owned_session(process.pid, run_id)
                try:
                    exit_code = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    exit_code = 124

        session_clean = self.wait_for_session_exit(process.pid, run_id)
        npu_clean = self.wait_for_npu_idle(
            f"{index:02d}_after",
            owned_session_id=process.pid,
            owned_run_id=run_id,
        )
        self.current_process = None
        self.current_session_id = None
        self.current_run_id = ""

        record = read_json(run_record_path)
        record_status = record.get("status") if record else None
        result_path = (
            record.get("paths", {}).get("result")
            if isinstance(record, dict)
            else None
        )
        artifact_valid = (
            record.get("artifact", {}).get("valid") is True
            if isinstance(record, dict)
            else False
        )
        passed = (
            not timed_out
            and exit_code == 0
            and record_status == "succeeded"
            and artifact_valid
            and session_clean
            and npu_clean
        )
        entry.update(
            {
                "status": "passed" if passed else ("timeout" if timed_out else "failed"),
                "exit_code": exit_code,
                "finished_at_utc": utc_now(),
                "duration_seconds": round(time.time() - started_epoch, 3),
                "result": result_path,
                "run_record_status": record_status,
                "error": (
                    ""
                    if passed
                    else (
                        f"exit={exit_code}, record_status={record_status}, "
                        f"artifact_valid={artifact_valid}, "
                        f"session_clean={session_clean}, npu_clean={npu_clean}"
                    )
                ),
            }
        )
        self.save_state()
        self.log(
            f"[Case {index:02d}/{len(CASES)}] {entry['status'].upper()} "
            f"{case_id} exit={exit_code} duration={entry['duration_seconds']}s "
            f"run_record={run_record_path}"
        )

    def write_report(self) -> None:
        cases = self.state["cases"]
        passed = sum(case["status"] == "passed" for case in cases)
        failed = sum(case["status"] in {"failed", "timeout", "blocked"} for case in cases)
        shutdown = self.state["shutdown"]
        cleanup = shutdown["cleanup"]
        final_npu = shutdown["final_npu_check"]
        lines = [
            f"# Ascend inference suite {self.suite_id}",
            "",
            f"- Status: `{self.state['status']}`",
            f"- Started: `{self.state['started_at_utc']}`",
            f"- Finished: `{self.state['finished_at_utc']}`",
            f"- Passed: `{passed}`",
            f"- Failed/timeout/blocked: `{failed}`",
            "",
            "| # | Case | Status | Exit | Duration(s) | run.json | Result |",
            "|---:|---|---|---:|---:|---|---|",
        ]
        for case in cases:
            lines.append(
                f"| {case['index']} | `{case['case_id']}` | `{case['status']}` | "
                f"{case['exit_code']} | {case['duration_seconds']} | "
                f"`{case['run_record']}` | `{case['result'] or ''}` |"
            )
        lines.extend(
            [
                "",
                f"Suite log: `{self.suite_log_path}`",
                "",
                "Each case run directory contains its complete `run.log`, "
                "`run.json`, and result. The suite directory contains controller "
                "logs and before/after NPU snapshots.",
                "",
                "## Shutdown verification",
                "",
                f"- Required: `{shutdown['required']}`",
                f"- Trigger: `{shutdown['trigger']}`",
                f"- Status: `{shutdown['status']}`",
                f"- Succeeded: `{shutdown['succeeded']}`",
                f"- Signal: `{shutdown['signal']}`",
                f"- Cleanup attempted: `{cleanup['attempted']}`",
                f"- Cleanup required: `{cleanup['required']}`",
                "- `terminate_owned_session` return: "
                f"`{cleanup['terminate_owned_session_return']}`",
                f"- Cleanup succeeded: `{cleanup['succeeded']}`",
                f"- Cleanup error: `{cleanup['error']}`",
                f"- Final NPU check attempted: `{final_npu['attempted']}`",
                f"- Final NPU query succeeded: `{final_npu['query_succeeded']}`",
                f"- Final NPU idle: `{final_npu['idle']}`",
                f"- Final NPU process IDs: `{final_npu['process_ids']}`",
                f"- Final HBM used (MB): `{final_npu['hbm_used_mb']}`",
                f"- HBM baseline (MB): `{final_npu['hbm_baseline_mb']}`",
                "- HBM within baseline tolerance: "
                f"`{final_npu['hbm_within_baseline']}`",
                f"- Final NPU check error: `{final_npu['error']}`",
                f"- Shutdown errors: `{shutdown['errors']}`",
                "",
            ]
        )
        self.report_path.write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> int:
        caught_exception: Exception | None = None
        try:
            self._signal_interruptible = True
            signal.signal(signal.SIGINT, self.handle_signal)
            signal.signal(signal.SIGTERM, self.handle_signal)
            selected = self.selected_cases()
            self.log(
                f"[Suite] START id={self.suite_id}, cases={len(selected)}, "
                f"case_timeout={self.arguments.case_timeout_seconds}s"
            )
            for index, case_id, relative_script in selected:
                if self.interrupted:
                    break
                self.run_case(index, case_id, relative_script)
            statuses = [case["status"] for case in self.state["cases"]]
            if self.interrupted:
                self.state["status"] = "interrupted"
            elif statuses and all(status == "passed" for status in statuses):
                self.state["status"] = "passed"
            else:
                self.state["status"] = "completed_with_failures"
        except Exception as exc:
            self._signal_interruptible = False
            caught_exception = exc
            for case in self.state["cases"]:
                if case.get("status") == "running":
                    case["status"] = (
                        "interrupted" if self.interrupted else "failed"
                    )
                    case["finished_at_utc"] = utc_now()
                    case["error"] = f"{type(exc).__name__}: {exc}"
            self.log(f"[Suite] ABORTED: {type(exc).__name__}: {exc}")
            self.state["status"] = "interrupted" if self.interrupted else "aborted"
        finally:
            self._signal_interruptible = False
            self._shutdown_in_progress = True
            self.record_signal_state()
            shutdown_required = (
                self.interrupted
                or caught_exception is not None
                or (
                    self.current_session_id is not None
                    and bool(self.current_run_id)
                )
            )
            if shutdown_required:
                if self.interrupted:
                    signal_state = self.state["shutdown"].get("signal")
                    trigger = (
                        f"signal:{signal_state['name']}"
                        if signal_state
                        else "signal"
                    )
                elif caught_exception is not None:
                    trigger = f"exception:{type(caught_exception).__name__}"
                else:
                    trigger = "unexpected_active_session"
                shutdown_succeeded = self.finalize_shutdown(trigger)
                if not shutdown_succeeded and self.state["status"] == "passed":
                    self.state["status"] = "cleanup_failed"
            self.state["finished_at_utc"] = utc_now()
            self.record_signal_state()
            self.save_state()
            self.write_report()
            self.log(
                f"[Suite] FINISH status={self.state['status']} "
                f"report={self.report_path}"
            )
        return 0 if self.state["status"] == "passed" else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-id",
        help="Unique suite ID; defaults to a UTC timestamp plus controller PID.",
    )
    parser.add_argument(
        "--only",
        action="append",
        help="Run only this exact case_id; may be passed more than once.",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=int,
        default=8 * 60 * 60,
        help="Per-case timeout; default: 28800 (8 hours).",
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=int,
        default=30 * 60,
        help="How long to wait for unowned NPU work; default: 1800 seconds.",
    )
    arguments = parser.parse_args(argv)
    if arguments.case_timeout_seconds < 60:
        parser.error("--case-timeout-seconds must be at least 60")
    if arguments.idle_timeout_seconds < 10:
        parser.error("--idle-timeout-seconds must be at least 10")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    suite = InferSuite(arguments)
    return suite.run()


if __name__ == "__main__":
    raise SystemExit(main())
