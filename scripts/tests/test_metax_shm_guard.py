import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def load_module(name, relative_path):
    repo = Path(__file__).resolve().parents[2]
    script = repo / relative_path
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def test_preflight_reports_low_shared_memory_with_stale_mccl_files(
    monkeypatch,
    tmp_path,
):
    module = load_module(
        "_metax_shm_preflight_under_test",
        "scripts/preflight_infer.py",
    )
    (tmp_path / "mccl-stale").write_bytes(b"x" * 17)
    (tmp_path / "unrelated").write_bytes(b"y" * 23)
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda _path: SimpleNamespace(f_bavail=100, f_frsize=4096),
    )

    errors = module.metax_shared_memory_errors(
        tmp_path,
        minimum_available_bytes=1024 * 1024,
    )

    assert len(errors) == 1
    assert "only 0.4 MiB is available" in errors[0]
    assert "1 mccl-* file(s)" in errors[0]


def test_preflight_accepts_sufficient_shared_memory(monkeypatch, tmp_path):
    module = load_module(
        "_metax_shm_preflight_sufficient_under_test",
        "scripts/preflight_infer.py",
    )
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda _path: SimpleNamespace(f_bavail=256, f_frsize=4096),
    )

    assert (
        module.metax_shared_memory_errors(
            tmp_path,
            minimum_available_bytes=1024 * 1024,
        )
        == []
    )


def test_suite_quarantines_only_owned_new_regular_mccl_files(tmp_path):
    module = load_module(
        "_metax_shm_suite_under_test",
        "scripts/metax/run_infer_suite.py",
    )
    shm = tmp_path / "shm"
    quarantine = tmp_path / "quarantine"
    shm.mkdir()
    old_file = shm / "mccl-00000029-old"
    old_file.write_bytes(b"old")
    before = module.snapshot_mccl_shm_files(shm)
    new_file = shm / "mccl-0000002a-new"
    new_file.write_bytes(b"new payload")
    unowned_file = shm / "mccl-0000002b-other"
    unowned_file.write_bytes(b"other job")
    unrelated = shm / "other"
    unrelated.write_bytes(b"keep")

    result = module.quarantine_new_mccl_shm_files(
        before,
        quarantine,
        {42},
        shm,
    )

    assert result["errors"] == []
    assert result["created_file_count"] == 2
    assert result["eligible_file_count"] == 1
    assert result["quarantined_file_count"] == 1
    assert result["quarantined_bytes"] == len(b"new payload")
    assert result["skipped_unowned_files"] == ["mccl-0000002b-other"]
    assert old_file.read_bytes() == b"old"
    assert unowned_file.read_bytes() == b"other job"
    assert unrelated.read_bytes() == b"keep"
    assert not new_file.exists()
    assert (quarantine / "mccl-0000002a-new").read_bytes() == b"new payload"


def test_suite_refuses_mccl_file_when_owned_pid_was_reused(tmp_path):
    module = load_module(
        "_metax_shm_suite_pid_reuse_under_test",
        "scripts/metax/run_infer_suite.py",
    )
    shm = tmp_path / "shm"
    proc = tmp_path / "proc"
    quarantine = tmp_path / "quarantine"
    shm.mkdir()
    process_dir = proc / "42"
    process_dir.mkdir(parents=True)
    fields = [
        "S",
        "1",
        "42",
        "42",
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
        "200",
    ]
    process_dir.joinpath("stat").write_text(f"42 (reused process) {' '.join(fields)}")
    reused_file = shm / "mccl-0000002a-reused"
    reused_file.write_bytes(b"other job")

    result = module.quarantine_new_mccl_shm_files(
        {},
        quarantine,
        {42: {100}},
        shm,
        proc,
    )

    assert result["eligible_file_count"] == 0
    assert result["quarantined_file_count"] == 0
    assert result["skipped_reused_pid_files"] == [reused_file.name]
    assert reused_file.read_bytes() == b"other job"
    assert not quarantine.exists()
