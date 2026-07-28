import importlib.util
import tempfile
import unittest
from pathlib import Path

REPO_PATH = Path(__file__).resolve().parents[2]
SUITE_TOOL = REPO_PATH / "scripts" / "metax" / "run_infer_suite.py"


def load_suite_module():
    spec = importlib.util.spec_from_file_location(
        "_metax_suite_locks_under_test",
        SUITE_TOOL,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ControllerLockTest(unittest.TestCase):
    def setUp(self):
        self.module = load_suite_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.suite_root = Path(self.temp_dir.name) / "suites"

    def test_different_suite_is_rejected_while_gpu_lock_is_held(self):
        first_suite = self.suite_root / "first"
        second_suite = self.suite_root / "second"

        with self.module.hold_controller_locks(
            self.suite_root,
            first_suite,
            "first",
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "global controller lock is already held",
            ):
                with self.module.hold_controller_locks(
                    self.suite_root,
                    second_suite,
                    "second",
                ):
                    self.fail("a second suite acquired the MetaX GPU lock")

    def test_released_gpu_lock_allows_another_suite(self):
        first_suite = self.suite_root / "first"
        second_suite = self.suite_root / "second"

        with self.module.hold_controller_locks(
            self.suite_root,
            first_suite,
            "first",
        ):
            pass

        with self.module.hold_controller_locks(
            self.suite_root,
            second_suite,
            "second",
        ):
            self.assertTrue((self.suite_root / self.module.METAX_GPU_LOCK_NAME).is_file())

    def test_exception_releases_both_controller_locks(self):
        first_suite = self.suite_root / "first"
        second_suite = self.suite_root / "second"

        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with self.module.hold_controller_locks(
                self.suite_root,
                first_suite,
                "first",
            ):
                raise RuntimeError("test failure")

        with self.module.hold_controller_locks(
            self.suite_root,
            second_suite,
            "second",
        ):
            pass

    def test_same_suite_lock_is_rejected(self):
        suite_dir = self.suite_root / "same"
        first_lock = self.module.acquire_suite_lock(suite_dir)

        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "suite controller lock is already held",
            ):
                with self.module.hold_controller_locks(
                    self.suite_root,
                    suite_dir,
                    "same",
                ):
                    self.fail("a second controller acquired the same suite lock")
        finally:
            first_lock.close()

        with self.module.hold_controller_locks(
            self.suite_root,
            suite_dir,
            "same",
        ):
            pass

    def test_active_legacy_controller_in_other_suite_is_rejected(self):
        legacy_suite = self.suite_root / "legacy"
        new_suite = self.suite_root / "new"
        legacy_lock = self.module.acquire_suite_lock(legacy_suite)

        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "active MetaX suite controller.*without the global GPU lock",
            ) as raised:
                with self.module.hold_controller_locks(
                    self.suite_root,
                    new_suite,
                    "new",
                ):
                    self.fail("a new suite ignored an active legacy controller")
            self.assertIn(
                str(legacy_suite / "controller.lock"),
                str(raised.exception),
            )
        finally:
            legacy_lock.close()

        with self.module.hold_controller_locks(
            self.suite_root,
            new_suite,
            "new",
        ):
            pass


if __name__ == "__main__":
    unittest.main()
