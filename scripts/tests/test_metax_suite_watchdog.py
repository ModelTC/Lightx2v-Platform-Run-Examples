import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_PATH = Path(__file__).resolve().parents[2]
SUITE_TOOL = REPO_PATH / "scripts" / "metax" / "run_infer_suite.py"


def load_suite_module():
    spec = importlib.util.spec_from_file_location(
        "_metax_suite_watchdog_under_test",
        SUITE_TOOL,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SuiteWatchdogTest(unittest.TestCase):
    def setUp(self):
        self.module = load_suite_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.run_log = self.root / "run.log"
        self.run_record = self.root / "run.json"
        self.result_dir = self.root / "results"
        self.result_dir.mkdir()
        self.processes = []
        self.addCleanup(self._cleanup_processes)

    def _start_python(self, source):
        process = subprocess.Popen(
            [sys.executable, "-c", source],
            start_new_session=True,
        )
        self.processes.append(process)
        return process

    def _wait(
        self,
        process,
        *,
        timeout,
        grace=0.3,
        on_timeout=None,
        queue_timeout=0,
        queue_min_count=3,
    ):
        return self.module.wait_for_case_process(
            process,
            run_log_path=self.run_log,
            run_record_path=self.run_record,
            result_dir=self.result_dir,
            stall_timeout_seconds=timeout,
            queue_retry_timeout_seconds=queue_timeout,
            queue_retry_min_count=queue_min_count,
            poll_interval_seconds=0.02,
            termination_grace_seconds=grace,
            on_timeout=on_timeout,
        )

    def _cleanup_processes(self):
        for process in self.processes:
            if process.poll() is not None:
                continue
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def test_each_watched_file_type_refreshes_stall_deadline(self):
        result_path = self.result_dir / "output.tmp"
        source = (
            "import pathlib,time\n"
            "paths=["
            f"pathlib.Path({str(self.run_log)!r}),"
            f"pathlib.Path({str(self.run_record)!r}),"
            f"pathlib.Path({str(result_path)!r})]\n"
            "for index in range(9):\n"
            " paths[index % len(paths)].write_text(str(index))\n"
            " time.sleep(0.1)\n"
        )
        process = self._start_python(source)

        exit_code, status = self._wait(process, timeout=0.3)

        self.assertEqual(exit_code, 0)
        self.assertFalse(status["timed_out"])
        self.assertEqual(self.run_log.read_text(), "6")
        self.assertEqual(self.run_record.read_text(), "7")
        self.assertEqual(result_path.read_text(), "8")

    def test_timeout_kills_entire_process_group_and_returns_124(self):
        child_pid_path = self.root / "child.pid"
        child_source = "import signal,time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nwhile True: time.sleep(1)\n"
        parent_source = (
            "import os,pathlib,signal,subprocess,sys,time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"child=subprocess.Popen([sys.executable,'-c',{child_source!r}])\n"
            f"pathlib.Path({str(child_pid_path)!r}).write_text("
            "f'{child.pid} {os.getpgid(child.pid)}')\n"
            "while True: time.sleep(1)\n"
        )
        process = self._start_python(parent_source)
        deadline = time.monotonic() + 3
        while not child_pid_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(child_pid_path.is_file())
        child_pid, child_process_group = map(
            int,
            child_pid_path.read_text().split(),
        )
        self.assertGreater(child_pid, 1)
        self.assertEqual(child_process_group, process.pid)
        timeout_updates = []

        exit_code, status = self._wait(
            process,
            timeout=0.2,
            grace=0.2,
            on_timeout=timeout_updates.append,
        )

        self.assertEqual(exit_code, 124)
        self.assertTrue(status["timed_out"])
        self.assertTrue(status["sigterm_sent"])
        self.assertTrue(status["sigkill_sent"])
        self.assertEqual(status["child_return_code"], -signal.SIGKILL)
        self.assertEqual(len(timeout_updates), 1)
        self.assertEqual(timeout_updates[0]["timeout_exit_code"], 124)
        self.assertIsNotNone(process.poll())
        self.assertIn(process.pid, status["process_group_pids_seen"])
        self.assertFalse(
            status["process_group_alive_after_cleanup"],
            status,
        )

    def test_process_group_member_pid_parser(self):
        proc_root = self.root / "proc"
        worker_dir = proc_root / "42"
        other_dir = proc_root / "43"
        worker_dir.mkdir(parents=True)
        other_dir.mkdir()
        self._write_proc_stat(
            proc_root,
            42,
            "worker with ) spaces",
            parent_pid=1,
            process_group=777,
            start_time=4200,
        )
        self._write_proc_stat(
            proc_root,
            43,
            "other",
            parent_pid=1,
            process_group=778,
            start_time=4300,
        )

        self.assertEqual(
            self.module._process_group_member_pids(777, proc_root),
            {42},
        )
        parsed = self.module._proc_stat_for_pid(42, proc_root)
        self.assertEqual(parsed.identity.pid, 42)
        self.assertEqual(parsed.identity.start_time_ticks, 4200)
        self.assertEqual(parsed.parent_pid, 1)
        self.assertEqual(parsed.process_group_id, 777)

    def _write_proc_stat(
        self,
        proc_root,
        pid,
        command,
        *,
        parent_pid,
        process_group,
        start_time,
        state="S",
    ):
        process_dir = proc_root / str(pid)
        process_dir.mkdir(parents=True, exist_ok=True)
        fields = [
            state,
            str(parent_pid),
            str(process_group),
            str(process_group),
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "1",
            "0",
            "0",
            str(start_time),
        ]
        process_dir.joinpath("stat").write_text(
            f"{pid} ({command}) {' '.join(fields)}",
        )

    def test_recursive_descendants_include_setsid_worker_and_exclude_unrelated(self):
        proc_root = self.root / "proc"
        self._write_proc_stat(
            proc_root,
            10,
            "case leader",
            parent_pid=1,
            process_group=10,
            start_time=100,
        )
        self._write_proc_stat(
            proc_root,
            11,
            "torchrun agent",
            parent_pid=10,
            process_group=10,
            start_time=110,
        )
        self._write_proc_stat(
            proc_root,
            12,
            "rank 0 (setsid)",
            parent_pid=11,
            process_group=12,
            start_time=120,
        )
        self._write_proc_stat(
            proc_root,
            13,
            "unrelated setsid",
            parent_pid=1,
            process_group=13,
            start_time=130,
        )
        root_identity = self.module._ProcessIdentity(10, 100)

        descendants = self.module._descendant_processes(
            root_identity,
            {},
            proc_root,
        )

        self.assertEqual(
            {(identity.pid, depth) for identity, depth in descendants.items()},
            {(10, 0), (11, 1), (12, 2)},
        )

        # Once sampled, an owned setsid worker stays attributable after its
        # torchrun parent exits/reparents it, and its later children remain in
        # the same guarded tree.
        (proc_root / "11" / "stat").unlink()
        self._write_proc_stat(
            proc_root,
            12,
            "rank 0 (reparented)",
            parent_pid=1,
            process_group=12,
            start_time=120,
        )
        self._write_proc_stat(
            proc_root,
            14,
            "rank helper",
            parent_pid=12,
            process_group=14,
            start_time=140,
        )
        descendants = self.module._descendant_processes(
            root_identity,
            descendants,
            proc_root,
        )
        self.assertEqual(
            {(identity.pid, depth) for identity, depth in descendants.items()},
            {(10, 0), (12, 2), (14, 3)},
        )

    def test_signal_guard_rejects_pid_reuse_and_orders_deepest_first(self):
        proc_root = self.root / "proc"
        self._write_proc_stat(
            proc_root,
            11,
            "owned parent",
            parent_pid=1,
            process_group=10,
            start_time=110,
        )
        self._write_proc_stat(
            proc_root,
            12,
            "reused unrelated",
            parent_pid=1,
            process_group=12,
            start_time=999,
        )
        self._write_proc_stat(
            proc_root,
            14,
            "owned leaf",
            parent_pid=11,
            process_group=14,
            start_time=140,
        )
        self._write_proc_stat(
            proc_root,
            15,
            "untracked unrelated",
            parent_pid=1,
            process_group=15,
            start_time=150,
        )
        tracked = {
            self.module._ProcessIdentity(11, 110): 1,
            self.module._ProcessIdentity(12, 120): 2,
            self.module._ProcessIdentity(14, 140): 2,
        }
        calls = []

        signaled, errors = self.module._signal_matching_processes(
            tracked,
            signal.SIGTERM,
            proc_path=proc_root,
            signal_process=lambda pid, signum: calls.append((pid, signum)),
        )

        self.assertEqual(errors, [])
        self.assertEqual(calls, [(14, signal.SIGTERM), (11, signal.SIGTERM)])
        self.assertEqual([identity.pid for identity in signaled], [14, 11])
        self.assertNotIn(12, [pid for pid, _signum in calls])
        self.assertNotIn(15, [pid for pid, _signum in calls])

    def test_transient_queue_retry_recovery_is_not_terminated(self):
        marker = "[MXKW][E] [mxkwCreateQueueBlock][Hint]ioctl create queue block timeout, gpu_id:1 type:21. Retrying."
        source = (
            "import pathlib,time\n"
            f"path=pathlib.Path({str(self.run_log)!r})\n"
            f"marker={marker!r}\n"
            "with path.open('a') as output:\n"
            " for _ in range(3): output.write(marker+'\\n'); output.flush()\n"
            " time.sleep(0.1)\n"
            " output.write('business progress resumed\\n'); output.flush()\n"
            " time.sleep(0.1)\n"
            " for _ in range(3): output.write(marker+'\\n'); output.flush()\n"
            " time.sleep(0.1)\n"
        )
        process = self._start_python(source)

        exit_code, status = self._wait(
            process,
            timeout=2,
            queue_timeout=0.2,
        )

        self.assertEqual(exit_code, 0)
        self.assertFalse(status["timed_out"])
        self.assertEqual(status["queue_retry"]["total_retry_count"], 6)

    def test_run_record_progress_resets_queue_retry_incident(self):
        result_path = self.result_dir / "partial-output.bin"
        marker = "[MXKW][E] [mxkwCreateQueueBlock][Hint]ioctl create queue block timeout, gpu_id:1 type:21. Retrying."
        source = (
            "import pathlib,time\n"
            f"log_path=pathlib.Path({str(self.run_log)!r})\n"
            f"record_path=pathlib.Path({str(self.run_record)!r})\n"
            f"result_path=pathlib.Path({str(result_path)!r})\n"
            f"marker={marker!r}\n"
            "with log_path.open('a') as output:\n"
            " for _ in range(3): output.write(marker+'\\n'); output.flush()\n"
            " time.sleep(0.12)\n"
            ' record_path.write_text(\'{"status":"running"}\')\n'
            " for _ in range(3): output.write(marker+'\\n'); output.flush()\n"
            " time.sleep(0.12)\n"
            " result_path.write_bytes(b'partial')\n"
            " time.sleep(0.15)\n"
        )
        process = self._start_python(source)

        exit_code, status = self._wait(
            process,
            timeout=2,
            queue_timeout=0.2,
        )

        self.assertEqual(exit_code, 0)
        self.assertFalse(status["timed_out"])
        self.assertEqual(status["queue_retry"]["total_retry_count"], 6)
        self.assertEqual(status["queue_retry"]["incident_retry_count"], 0)

    def test_persistent_queue_retry_stops_then_times_out(self):
        marker = "[MXKW][E] [mxkwCreateQueueBlock][Hint]ioctl create queue block timeout, gpu_id:1 type:21. Retrying."
        source = (
            "import pathlib,signal,time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"path=pathlib.Path({str(self.run_log)!r})\n"
            f"marker={marker!r}\n"
            "with path.open('a') as output:\n"
            " for _ in range(3): output.write(marker+'\\n'); output.flush()\n"
            " while True: time.sleep(1)\n"
        )
        unrelated = self._start_python("import time; time.sleep(10)")
        process = self._start_python(source)

        exit_code, status = self._wait(
            process,
            timeout=2,
            grace=0.1,
            queue_timeout=0.2,
        )

        self.assertEqual(exit_code, 124)
        self.assertTrue(status["timed_out"])
        self.assertEqual(status["timeout_reason"], "metax_queue_retry_hang")
        self.assertEqual(status["queue_retry"]["incident_retry_count"], 3)
        self.assertGreaterEqual(status["queue_retry_hang_seconds"], 0.2)
        self.assertEqual(status["child_return_code"], -signal.SIGKILL)
        self.assertEqual(status["surviving_owned_processes"], [])
        self.assertIsNone(unrelated.poll())

    def test_zero_disables_watchdog(self):
        process = self._start_python("import time; time.sleep(0.4)")

        exit_code, status = self._wait(process, timeout=0)

        self.assertEqual(exit_code, 0)
        self.assertFalse(status["timed_out"])

    def test_normal_exit_preserves_child_return_code(self):
        process = self._start_python("raise SystemExit(7)")

        exit_code, status = self._wait(process, timeout=1)

        self.assertEqual(exit_code, 7)
        self.assertFalse(status["timed_out"])

    def test_user_interrupt_is_forwarded_without_watchdog_relabeling(self):
        process = self._start_python("import time; time.sleep(10)")
        self.module.ACTIVE_PROCESS = process
        self.addCleanup(setattr, self.module, "ACTIVE_PROCESS", None)
        self.addCleanup(setattr, self.module, "INTERRUPTED_SIGNAL", None)

        self.module.handle_signal(signal.SIGINT, None)
        exit_code, status = self._wait(process, timeout=0.1)

        self.assertEqual(self.module.INTERRUPTED_SIGNAL, signal.SIGINT)
        self.assertEqual(exit_code, -signal.SIGTERM)
        self.assertFalse(status["timed_out"])


if __name__ == "__main__":
    unittest.main()
