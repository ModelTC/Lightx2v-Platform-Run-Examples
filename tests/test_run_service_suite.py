import argparse
import json
import signal
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_PATH / "scripts"))

import run_service_suite as controller  # noqa: E402


def arguments(**overrides):
    values = {
        "suite_id": "unit_service_suite",
        "only": None,
        "sample_count": 10,
        "warmup_count": 1,
        "concurrency": 1,
        "poll_interval_seconds": 0.5,
        "startup_timeout_seconds": 1800.0,
        "task_timeout_seconds": 7200.0,
        "case_timeout_seconds": 86400.0,
        "npu_sample_interval_seconds": 1.0,
        "port": 8000,
        "metric_port": 8001,
        "master_port": 29500,
        "fail_fast": False,
        "dry_run": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class ServiceSuitePlanTest(unittest.TestCase):
    def test_current_distributed_services_are_exactly_covered(self):
        expected_ids = {path.name.removeprefix("start_server_").removesuffix(".sh") for path in controller.SERVER_ROOT.glob("start_server_*.sh")}
        planned_ids = {case.case_id for case in controller.CASES}

        self.assertEqual(12, len(controller.CASES))
        self.assertEqual(expected_ids, planned_ids)
        self.assertEqual(
            {"t2i": 4, "t2v": 7, "s2v": 1},
            {task_kind: sum(case.task_kind == task_kind for case in controller.CASES) for task_kind in ("t2i", "t2v", "s2v")},
        )

    def test_all_case_files_and_parallel_sizes_validate(self):
        suite = controller.ServiceSuite(arguments())
        suite.validate(controller.CASES)

        for case in controller.CASES:
            with self.subTest(case=case.case_id):
                paths = controller.case_paths(case)
                self.assertTrue(all(path.is_file() for path in paths.values()))
                config = json.loads(paths["model_config"].read_text(encoding="utf-8"))
                self.assertIn("parallel", config)

    def test_commands_separate_warmup_and_ten_sample_measurement(self):
        suite = controller.ServiceSuite(arguments())
        for case in controller.CASES:
            with self.subTest(case=case.case_id):
                warmup = suite.benchmark_command(
                    case,
                    phase="warmup",
                    limit=1,
                    run_dir=Path("/tmp/warmup"),
                )
                measure = suite.benchmark_command(
                    case,
                    phase="measure",
                    limit=10,
                    run_dir=Path("/tmp/measure"),
                )
                self.assertEqual("1", warmup[warmup.index("--limit") + 1])
                self.assertEqual("10", measure[measure.index("--limit") + 1])
                self.assertEqual("warmup", warmup[warmup.index("--phase") + 1])
                self.assertEqual("measure", measure[measure.index("--phase") + 1])
                self.assertEqual(
                    "1",
                    measure[measure.index("--concurrency") + 1],
                )

    def test_ltx_uses_s2v_dataset_with_ten_audio_samples(self):
        case = next(case for case in controller.CASES if case.task_kind == "s2v")
        data_path = controller.case_paths(case)["data"]
        rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(10, len(rows))
        self.assertEqual(
            [f"audios/{index:02d}.mp3" for index in range(1, 11)],
            [row["audio_path"] for row in rows],
        )
        manifest = controller.ServiceSuite(arguments()).case_manifest(case)
        self.assertEqual(10, len(manifest["data_dependencies"]))
        self.assertTrue(all(len(dependency["sha256"]) == 64 for dependency in manifest["data_dependencies"]))

    def test_busy_ports_are_rejected_in_configuration(self):
        suite = controller.ServiceSuite(arguments(metric_port=8000))
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            suite.validate(controller.CASES)

    def test_stop_client_terminates_only_owned_process_group(self):
        suite = controller.ServiceSuite(arguments())
        process = mock.Mock()
        process.pid = 4242
        process.poll.return_value = None
        process.wait.side_effect = [
            subprocess.TimeoutExpired("client", 10),
            0,
        ]
        suite.active_client_process = process

        with mock.patch.object(controller.os, "killpg") as killpg:
            suite.stop_client()

        self.assertEqual(
            [
                mock.call(4242, signal.SIGTERM),
                mock.call(4242, signal.SIGKILL),
            ],
            killpg.call_args_list,
        )
        self.assertIsNone(suite.active_client_process)

    def test_stop_server_cleans_all_processes_with_exact_run_id(self):
        suite = controller.ServiceSuite(arguments())
        process = mock.Mock()
        process.pid = 4242
        process.poll.return_value = 0
        suite.active_process = process
        suite.active_server_run_id = "unit_service_suite:case"

        with (
            mock.patch.object(
                suite,
                "signal_owned_server",
                return_value=[4242, 4343],
            ) as signal_owned,
            mock.patch.object(suite, "wait_owned_server_exit", return_value=True),
            mock.patch.object(suite, "owned_server_pids", return_value=[]),
        ):
            result = suite.stop_server()

        signal_owned.assert_called_once_with("unit_service_suite:case", signal.SIGTERM)
        self.assertEqual([4242, 4343], result["term_pids"])
        self.assertTrue(result["graceful"])
        self.assertEqual([], result["residual_pids"])
        self.assertEqual("", suite.active_server_run_id)


if __name__ == "__main__":
    unittest.main()
