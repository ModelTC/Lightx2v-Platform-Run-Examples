import importlib.util
from pathlib import Path


def load_suite_runner():
    script = Path(__file__).resolve().parents[1] / "metax" / "run_infer_suite.py"
    spec = importlib.util.spec_from_file_location(
        "_metax_suite_cooldown_under_test",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_release_cooldown_remaining_is_bounded():
    module = load_suite_runner()

    assert module.gpu_release_cooldown_remaining(100, 109, 30) == 21
    assert module.gpu_release_cooldown_remaining(100, 130, 30) == 0
    assert module.gpu_release_cooldown_remaining(100, 200, 0) == 0


def test_only_distributed_cases_require_cooldown():
    module = load_suite_runner()

    assert module.case_requires_gpu_release_cooldown("dist_2")
    assert module.case_requires_gpu_release_cooldown("dist_8")
    assert not module.case_requires_gpu_release_cooldown("single")


def test_gpu_release_wait_uses_only_remaining_interval():
    module = load_suite_runner()
    clock = [109.0]
    sleeps = []

    def monotonic():
        return clock[0]

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    waited = module.wait_for_gpu_release_cooldown(
        100.0,
        30.0,
        monotonic=monotonic,
        sleep=sleep,
    )

    assert waited == 21
    assert sum(sleeps) == 21
    assert max(sleeps) <= 1
