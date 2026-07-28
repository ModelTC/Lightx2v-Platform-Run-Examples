import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_PATH = Path(__file__).resolve().parents[2]
SUITE_TOOL = REPO_PATH / "scripts" / "metax" / "run_infer_suite.py"


def load_suite_module():
    spec = importlib.util.spec_from_file_location(
        "_metax_suite_recovery_under_test",
        SUITE_TOOL,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ValidationOnlyRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.module = load_suite_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.artifact_path = root / "output.mp4"
        self.artifact_path.write_bytes(b"reviewed-artifact")
        self.artifact_sha256 = hashlib.sha256(self.artifact_path.read_bytes()).hexdigest()
        self.record_path = root / "run.json"
        self.case_id = "reviewed_case"
        self.run_id = "reviewed_run"
        validation_error = "ffmpeg: error while loading shared libraries: libfreetype.so.6: cannot open shared object file"
        self.record = {
            "run_id": self.run_id,
            "status": "failed",
            "benchmark": {
                "case_id": self.case_id,
                "task": "t2v",
                "target": {},
            },
            "exit": {
                "child_exit_code": 0,
                "wrapper_exit_code": 0,
                "error": "",
            },
            "artifact": {
                "path": str(self.artifact_path),
                "format": "mp4",
                "size_bytes": self.artifact_path.stat().st_size,
                "sha256": self.artifact_sha256,
                "valid": False,
                "validation": {
                    "method": "ffmpeg",
                    "errors": [validation_error],
                },
            },
            "errors": [validation_error],
        }
        self.original_record_bytes = (json.dumps(self.record, ensure_ascii=False, indent=2) + "\n").encode()
        self.record_path.write_bytes(self.original_record_bytes)
        self.case = {
            "case_id": self.case_id,
            "attempts": [
                {
                    "run_id": self.run_id,
                    "exit_code": 74,
                    "run_record": str(self.record_path),
                }
            ],
        }

        def validate_artifact(path, result_format, target, expected_audio):
            self.assertEqual(path, self.artifact_path)
            self.assertEqual(result_format, "mp4")
            self.assertEqual(target, {})
            self.assertFalse(expected_audio)
            return {
                "path": str(path),
                "format": result_format,
                "size_bytes": path.stat().st_size,
                "sha256": self.artifact_sha256,
                "valid": True,
                "validation": {
                    "method": "test-full-decode",
                    "errors": [],
                },
            }

        self.fake_validator = SimpleNamespace(validate_artifact=validate_artifact)

    def recovery_patches(self, allowlist=None, denylist=None):
        if allowlist is None:
            allowlist = {(self.case_id, self.run_id): self.artifact_sha256}
        if denylist is None:
            denylist = set()
        return (
            mock.patch.object(
                self.module,
                "VALIDATION_ONLY_RECOVERY_ALLOWLIST",
                allowlist,
            ),
            mock.patch.object(
                self.module,
                "NON_RECOVERABLE_VALIDATION_ONLY_RUNS",
                denylist,
            ),
            mock.patch.object(
                self.module,
                "RUN_RECORD_MODULE",
                self.fake_validator,
            ),
        )

    def test_exact_allowlist_recovers_and_archives_idempotently(self):
        patches = self.recovery_patches()
        with patches[0], patches[1], patches[2]:
            recovered, reason, recovery = self.module.recover_validation_only_failure(self.case)
            self.assertTrue(recovered, reason)
            archive_path = self.record_path.with_name(self.module.VALIDATION_RECOVERY_ARCHIVE_NAME)
            self.assertEqual(
                archive_path.read_bytes(),
                self.original_record_bytes,
            )
            updated = json.loads(self.record_path.read_bytes())
            self.assertEqual(updated["status"], "succeeded")
            self.assertTrue(updated["artifact"]["valid"])
            self.assertEqual(updated["errors"], [])
            self.assertEqual(
                recovery["previous_run_record_sha256"],
                hashlib.sha256(self.original_record_bytes).hexdigest(),
            )

            recovered, reason, repeated = self.module.recover_validation_only_failure(self.case)
            self.assertTrue(recovered, reason)
            self.assertEqual(repeated, recovery)
            valid, reason = self.module.validate_successful_case(self.case)
            self.assertTrue(valid, reason)
            profiled, profile_reason = self.module.validate_successful_case(
                self.case,
                require_dit_profile=True,
            )
            self.assertFalse(profiled)
            self.assertIn("metrics", profile_reason)

    def test_non_allowlisted_attempt_is_rejected(self):
        patches = self.recovery_patches(allowlist={})
        with patches[0], patches[1], patches[2]:
            recovered, reason, _ = self.module.recover_validation_only_failure(self.case)
        self.assertFalse(recovered)
        self.assertIn("not in the exact", reason)
        self.assertEqual(self.record_path.read_bytes(), self.original_record_bytes)

    def test_allowlisted_identifier_with_wrong_digest_is_rejected(self):
        patches = self.recovery_patches(allowlist={(self.case_id, self.run_id): "0" * 64})
        with patches[0], patches[1], patches[2]:
            recovered, reason, _ = self.module.recover_validation_only_failure(self.case)
        self.assertFalse(recovered)
        self.assertIn("reviewed allowlisted digest", reason)
        self.assertEqual(self.record_path.read_bytes(), self.original_record_bytes)

    def test_explicit_numerical_path_denylist_wins(self):
        key = (self.case_id, self.run_id)
        patches = self.recovery_patches(denylist={key})
        with patches[0], patches[1], patches[2]:
            recovered, reason, _ = self.module.recover_validation_only_failure(self.case)
        self.assertFalse(recovered)
        self.assertIn("numerical path", reason)
        self.assertEqual(self.record_path.read_bytes(), self.original_record_bytes)


if __name__ == "__main__":
    unittest.main()
