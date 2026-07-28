#!/usr/bin/env python3
"""Accelerator-free contract tests for the MetaX Server scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path("/data/Lightx2v-Platform-Run-Examples")
SERVER = REPO / "scripts" / "metax" / "server"
EXPECTED_CASES = {
    "z_image_turbo_t2i_1664x928_sp2",
    "flux2_dev_t2i_1344x768_tp8",
    "longcat_image_t2i_1344x768_cfg2_sp4",
    "qwen_image_2512_t2i_1664x928_cfg2_sp4",
    "wan21_1_3b_t2v_480p_81f_cfg2_sp4",
    "wan22_moe_a14b_t2v_480p_81f_cfg2_sp4",
    "wan22_moe_a14b_t2v_480p_81f_tp8",
    "wan22_moe_a14b_t2v_720p_81f_cfg2_sp4",
    "wan22_moe_a14b_t2v_720p_81f_tp8",
    "hunyuan_video_15_t2v_480p_121f_cfg2_sp4",
    "hunyuan_video_15_t2v_720p_121f_cfg2_sp4",
    "ltx2_3_22b_dev_s2v_768x512_241f_sp8",
}


def load_runner():
    name = "_test_metax_server_runner"
    spec = importlib.util.spec_from_file_location(
        name, SERVER / "run_service_suite.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def shell_assignments(path: Path) -> dict[str, str]:
    assignments = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if not name.replace("_", "").isalnum():
            continue
        assignments[name] = value.strip().strip("\"'")
    return assignments


RUNNER = load_runner()


def mx_fixture(*, process: bool = False, devices: int = 8) -> str:
    lines = [
        "mx-smi version: 2.3.1",
        "| Process: |",
        (
            "|  0  4242 python 1024 MiB |"
            if process
            else "|  no process found  |"
        ),
    ]
    for device in range(devices):
        lines.extend(
            [
                (
                    f"| {device}     MetaX C500 | {device}           Off | "
                    "0000:00:00.0 | 0% Disabled |"
                ),
                (
                    "| 61W / 350W | 33C P0 | "
                    "858/65536 MiB | Available |"
                ),
            ]
        )
    return "\n".join(lines)


class MetaXServerScriptsTest(unittest.TestCase):
    def test_fatal_server_patterns_cover_metax_oom(self):
        joined = b"\n".join(RUNNER.FATAL_SERVER_LOG_PATTERNS)
        self.assertIn(b"torch.OutOfMemoryError:", joined)
        self.assertIn(b"CUDA out of memory.", joined)

    def test_case_matrix_matches_ascend_contract(self):
        case_ids = {case.case_id for case in RUNNER.common.CASES}
        self.assertEqual(case_ids, EXPECTED_CASES)
        ascend_starters = {
            path.name.removeprefix("start_server_").removesuffix(".sh")
            for path in (
                REPO / "scripts" / "ascend" / "server" / "dist"
            ).glob("start_server_*.sh")
        }
        self.assertEqual(case_ids, ascend_starters)
        starters = {
            path.name.removeprefix("start_server_").removesuffix(".sh")
            for path in (SERVER / "dist").glob("start_server_*.sh")
        }
        self.assertEqual(starters, case_ids)

    def test_every_case_uses_existing_metax_inputs(self):
        for case in RUNNER.common.CASES:
            paths = RUNNER.common.case_paths(case, RUNNER.METAX_PATHS)
            for path in paths.values():
                self.assertTrue(path.is_file(), path)
            config = json.loads(paths["model_config"].read_text())
            parallel = config["parallel"]
            tensor = int(parallel.get("tensor_p_size", 1))
            effective = (
                tensor
                if tensor > 1
                else int(parallel.get("cfg_p_size", 1))
                * int(parallel.get("seq_p_size", 1))
            )
            self.assertEqual(effective, case.world_size, case.case_id)
            starter = paths["service_script"].read_text()
            model_line = next(
                line
                for line in starter.splitlines()
                if line.startswith("model_path=")
            )
            self.assertTrue(Path(model_line.split("=", 1)[1]).exists())
            self.assertIn(
                "scripts/metax/server/metax_server_runtime.sh", starter
            )

    def test_server_starters_reuse_reviewed_infer_case_settings(self):
        compared_fields = (
            "model_path",
            "config_path",
            "case_id",
            "model_cls",
            "task",
            "world_size",
            "parallel_strategy",
        )
        for case in RUNNER.common.CASES:
            paths = RUNNER.common.case_paths(case, RUNNER.METAX_PATHS)
            starter = shell_assignments(paths["service_script"])
            infer = shell_assignments(paths["source_script"])
            for field in compared_fields:
                self.assertEqual(
                    starter.get(field),
                    infer.get(field),
                    f"{case.case_id}: {field}",
                )
            visible_devices = starter["visible_devices"].split(",")
            self.assertEqual(len(visible_devices), case.world_size)
            self.assertEqual(
                visible_devices,
                [str(index) for index in range(case.world_size)],
            )

    def test_mx_smi_idle_snapshot_requires_all_eight_cards(self):
        processes, memory = RUNNER.parse_mx_smi_snapshot(mx_fixture())
        self.assertEqual(processes, set())
        self.assertEqual(set(memory), set(range(8)))
        self.assertEqual(set(memory.values()), {858})

    def test_mx_smi_process_snapshot_is_fail_closed(self):
        processes, _ = RUNNER.parse_mx_smi_snapshot(
            mx_fixture(process=True)
        )
        self.assertEqual(processes, {4242})
        actual_process_columns = mx_fixture().replace(
            "|  no process found  |",
            (
                "|  0                  4075889         python3.12"
                "                   29760          |"
            ),
        )
        processes, _ = RUNNER.parse_mx_smi_snapshot(
            actual_process_columns
        )
        self.assertEqual(processes, {4075889})
        with self.assertRaisesRegex(ValueError, "device set is incomplete"):
            RUNNER.parse_mx_smi_snapshot(mx_fixture(devices=7))
        broken = mx_fixture().replace("|  no process found  |", "")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            RUNNER.parse_mx_smi_snapshot(broken)

    def test_runtime_and_report_contract_are_level_zero(self):
        runtime = (SERVER / "metax_server_runtime.sh").read_text()
        report = (SERVER / "aggregate_service_reports.py").read_text()
        self.assertGreaterEqual(
            runtime.count("export PROFILING_DEBUG_LEVEL=0"), 2
        )
        self.assertIn("EXPECTED_PROFILE_LEVEL = 0", report)
        self.assertIn("E2E P50(s)", report)
        self.assertIn("E2E P90(s)", report)
        self.assertIn("E2E Min(s)", report)
        self.assertIn("E2E Max(s)", report)
        self.assertNotIn("DiT P50", report)
        self.assertIn(
            'REPO_PATH / "results" / "metax" / "server"', report
        )
        self.assertIn("server log does not prove effective", report)

    def test_report_core_uses_repository_metax_roots(self):
        report_core = RUNNER.SERVER_PATH / "_service_report_core.py"
        name = "_test_metax_server_report_core"
        spec = importlib.util.spec_from_file_location(name, report_core)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self.assertEqual(module.REPO_PATH, REPO)
        self.assertEqual(
            module.result_root("metax"),
            REPO / "results" / "metax" / "server",
        )
        self.assertEqual(
            module.log_root("metax"),
            REPO / "logs" / "metax" / "server",
        )

    def test_private_core_is_used_instead_of_mutable_public_controller(self):
        runner = (SERVER / "run_service_suite.py").read_text()
        reporter = (SERVER / "aggregate_service_reports.py").read_text()
        self.assertIn("import _service_suite_core as common", runner)
        self.assertIn("import _service_report_core as common", reporter)
        self.assertNotIn("import run_service_suite as common", runner)
        self.assertNotIn(
            "import aggregate_service_reports as common", reporter
        )

    def test_pinned_shared_client_hashes_match(self):
        for path, expected in RUNNER.PINNED_SHARED_FILES.items():
            self.assertEqual(RUNNER.common.sha256_file(path), expected, path)
        RUNNER.validate_pinned_shared_files()

    def test_git_provenance_works_across_checkout_ownership(self):
        for repository in (REPO, RUNNER.LIGHTX2V_PATH):
            state = RUNNER.common.git_state(repository)
            self.assertTrue(state["commands_succeeded"], repository)
            self.assertRegex(state["revision"], r"^[0-9a-f]{40}$")
            self.assertIn("status_sha256", state)

    def test_smoke_and_formal_launchers_have_disjoint_contracts(self):
        smoke = (SERVER / "run_smoke_detached.sh").read_text()
        formal = (SERVER / "run_all_detached.sh").read_text()
        report = (SERVER / "aggregate_service_reports.py").read_text()
        self.assertIn(
            'smoke_cases="z_image_turbo_t2i_1664x928_sp2"', smoke
        )
        self.assertIn(
            'metax_server_launch_new diagnostic "${suite_id}"', smoke
        )
        self.assertIn(
            'metax_server_launch_new formal "${suite_id}" ""', formal
        )
        self.assertIn(
            '--source-suite must be a formal suite',
            (SERVER / "_service_report_core.py").read_text(),
        )
        self.assertIn('"smoke_results_included": False', report)

    def test_type21_retry_watchdog_contract(self):
        marker = (
            b"[mxkwCreateQueueBlock][Hint]ioctl create queue block timeout "
            b"type:21. Retrying.\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "server.log"
            log_path.write_bytes(marker * 12)
            suite = object.__new__(RUNNER.MetaXServiceSuite)
            suite.active_server_log_path = log_path
            watch = RUNNER.offline_helpers._QueueRetryWatch()
            self.assertFalse(suite._queue_retry_timed_out(watch, 100.0))
            self.assertTrue(suite._queue_retry_timed_out(watch, 401.0))
            with log_path.open("ab") as file_obj:
                file_obj.write(b"business progress\n")
            self.assertFalse(suite._queue_retry_timed_out(watch, 402.0))

    def test_metax_lifecycle_isolated_and_named(self):
        runner = (SERVER / "run_service_suite.py").read_text()
        core = (SERVER / "_service_suite_core.py").read_text()
        report_core = (SERVER / "_service_report_core.py").read_text()
        self.assertIn('logs" / "metax" / "infer" / "suites"', runner)
        self.assertIn("metax_shared_memory_errors()", runner)
        self.assertIn('"metax": "metax"', core)
        self.assertIn("metax_samples.csv", report_core)


if __name__ == "__main__":
    unittest.main()
