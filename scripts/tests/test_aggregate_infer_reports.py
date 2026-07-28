from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AGGREGATE_PATH = REPOSITORY_ROOT / "scripts" / "aggregate_infer_reports.py"
SPEC = importlib.util.spec_from_file_location(
    "aggregate_infer_reports_under_test",
    AGGREGATE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {AGGREGATE_PATH}")
aggregate_reports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate_reports)


class CrossPlatformAggregateTests(unittest.TestCase):
    def test_metax_attempt_history_and_platform_title(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aggregate-report-test-") as directory:
            root = Path(directory)
            suite_dir = root / "metax_20260727T000000Z"
            run_dir = root / "run"
            suite_dir.mkdir()
            run_dir.mkdir()
            run_record_path = run_dir / "run.json"
            run_record = {
                "run_id": "metax_case_01",
                "status": "succeeded",
                "benchmark": {
                    "case_id": "z_image_turbo_t2i_1664x928",
                    "task": "t2i",
                },
                "artifact": {
                    "valid": True,
                    "format": "png",
                    "sha256": "a" * 64,
                    "validation": {
                        "schema_version": "1.0",
                        "validator": "run_record.validate_artifact",
                        "checked": True,
                        "method": "Pillow.Image.verify + RGB extrema",
                        "errors": [],
                    },
                },
                "paths": {
                    "run_log": str(run_dir / "run.log"),
                    "run_record": str(run_record_path),
                    "result": str(run_dir / "output.png"),
                },
                "timing": {
                    "started_at_utc": "2026-07-27T00:00:00Z",
                    "finished_at_utc": "2026-07-27T00:01:00Z",
                    "duration_seconds": 60,
                },
                "exit": {"child_exit_code": 0, "error": ""},
                "metrics": {
                    "wall_seconds": 60,
                    "profile_seconds": {"dit": 9.0},
                    "dit_seconds_per_step": 1.0,
                    "dit_first_step_seconds": 1.2,
                    "dit_median_step_seconds": 0.9,
                    "dit_p95_step_seconds": 1.1,
                },
                "errors": [],
            }
            run_record_path.write_text(
                json.dumps(run_record),
                encoding="utf-8",
            )
            suite = {
                "suite_id": suite_dir.name,
                "platform": "metax_cuda",
                "status": "success",
                "cases": [
                    {
                        "case_id": "z_image_turbo_t2i_1664x928",
                        "status": "success",
                        "attempts": [
                            {
                                "run_id": "metax_case_01",
                                "run_record": str(run_record_path),
                                "exit_code": 0,
                            }
                        ],
                    }
                ],
            }
            (suite_dir / "suite.json").write_text(
                json.dumps(suite),
                encoding="utf-8",
            )

            discovered = aggregate_reports.discover_suite_directories([], root)
            report = aggregate_reports.aggregate(
                discovered,
                platform_key="metax",
            )
            markdown = aggregate_reports.render_markdown(report)

        self.assertEqual(discovered, [suite_dir])
        self.assertEqual(report["summary"]["passed"], 1)
        self.assertEqual(
            report["cases"][0]["selected_success"]["run_id"],
            "metax_case_01",
        )
        self.assertIn("# MetaX C500 Offline Inference Final Report", markdown)
        self.assertIn("First DiT step (s)", markdown)
        self.assertIn("End-to-end (s)", markdown)
        self.assertNotIn("P50 step", markdown)
        self.assertNotIn("P95 step", markdown)
        self.assertIn("1.2", markdown)


if __name__ == "__main__":
    unittest.main()
