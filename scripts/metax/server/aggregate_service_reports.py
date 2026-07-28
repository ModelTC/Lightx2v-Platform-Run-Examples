#!/usr/bin/env python3
"""Build the MetaX Level-0 formal service benchmark report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_PATH = Path("/data/Lightx2v-Platform-Run-Examples")
SCRIPTS_PATH = REPO_PATH / "scripts"
PLATFORM = "metax"
PLATFORM_LABEL = "MetaX C500"
EXPECTED_PROFILE_LEVEL = 0
PINNED_SERVICE_COMMON_SHA256 = (
    "a0cdc4a3f5cf9d025247065601155d689103fa41bb1bf2eb54f202f2f492ed39"
)

if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))
SERVER_PATH = SCRIPTS_PATH / "metax" / "server"
if str(SERVER_PATH) not in sys.path:
    sys.path.insert(0, str(SERVER_PATH))

import _service_report_core as common  # noqa: E402

common.PLATFORM_LABELS[PLATFORM] = PLATFORM_LABEL


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate formal MetaX Level-0 service suites. Diagnostic "
            "smoke suites are trace-only and never selected for metrics."
        )
    )
    parser.add_argument("--source-suite", action="append", required=True)
    parser.add_argument("--diagnostic-suite", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _suite_path(value: str) -> Path:
    return common.suite_path(value, PLATFORM)


def _validate_formal(summary: dict[str, Any], path: Path) -> None:
    common.validate_suite_contract(summary, path, PLATFORM)
    parameters = summary.get("parameters", {})
    if parameters.get("profiling_debug_level") != EXPECTED_PROFILE_LEVEL:
        raise ValueError(
            f"{path}: profiling_debug_level "
            f"{parameters.get('profiling_debug_level')!r} != 0"
        )
    manifest = common.read_json(path / "manifest.json")
    profiling = manifest.get("profiling")
    if not isinstance(profiling, dict) or profiling.get("level") != 0:
        raise ValueError(f"{path}: manifest does not prove profiling level 0")


def _collect_diagnostics(values: list[str]) -> list[dict[str, Any]]:
    diagnostics = []
    for value in values:
        path = _suite_path(value)
        summary = common.read_json(path / "summary.json")
        if summary.get("suite_kind") != "diagnostic":
            raise ValueError(
                f"{path}: diagnostic suite_kind is "
                f"{summary.get('suite_kind')!r}"
            )
        diagnostics.append(
            {
                "suite_id": path.name,
                "status": summary.get("status"),
                "summary_json": str(path / "summary.json"),
                "summary_markdown": str(path / "summary.md"),
            }
        )
    return diagnostics


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# {PLATFORM_LABEL} 服务化推理性能报告",
        "",
        f"- 生成时间：`{report['generated_at']}`",
        "- 测试口径：1 次预热（不计入统计）+ 10 个正式样本，并发 1",
        "- Profiling：Level 0（关闭）；本报告不统计逐步 DiT",
        "- 延迟：客户端端到端时延，P50/P90 仅由 10 个成功正式样本计算",
        f"- 总体结果：`{report['status']}`，"
        f"{report['cases_passed']}/{report['cases_expected']} 个配置通过，"
        f"{report['artifacts_verified']}/{report['artifacts_expected']} "
        "个正式产物通过 SHA256 与媒体校验",
        "",
        "| 配置 | 任务 | 卡数 | 并行 | 状态 | E2E P50(s) | E2E P90(s) | "
        "E2E Avg(s) | E2E Min(s) | E2E Max(s) | 吞吐/分 | 启动(s) | "
        "预热(s) | 峰值显存(MiB) | 媒体 |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|",
    ]
    for case in report["cases"]:
        latency = case["latency_seconds"]
        validation = case["artifact_validation"]
        memory = case["peak_device_memory_mb"]["max"]
        lines.append(
            f"| {case['case_id']} | {case['task_kind']} | "
            f"{case['world_size']} | {case['parallel_strategy']} | "
            f"{case['status']} | {latency['p50_s']:.4f} | "
            f"{latency['p90_s']:.4f} | {latency['avg_s']:.4f} | "
            f"{latency['min_s']:.4f} | {latency['max_s']:.4f} | "
            f"{case['throughput_per_minute']:.6f} | "
            f"{case['startup_seconds']:.2f} | "
            f"{case['warmup_seconds']:.2f} | "
            f"{common.format_number(memory, digits=0)} | "
            f"{validation['media_verified']}/"
            f"{validation['artifacts_expected']} |"
        )
    lines.extend(["", "## 结果追溯", ""])
    for case in report["cases"]:
        source = case["source"]
        lines.extend(
            [
                f"### {case['case_id']}",
                "",
                f"- 正式结果：`{source['result']}`",
                f"- 请求明细：`{source['requests']}`",
                f"- 正式产物：`{source['outputs']}`",
                f"- 服务日志：`{source['server_log']}`",
                f"- 客户端日志：`{source['client_log']}`",
                f"- 显存采样：`{source['device_samples']}`",
                f"- 来源正式批次：`{source['suite_id']}`",
                "",
            ]
        )
    lines.extend(["## 诊断批次（不参与统计）", ""])
    if report["diagnostic_suites"]:
        for diagnostic in report["diagnostic_suites"]:
            lines.append(
                f"- `{diagnostic['suite_id']}`："
                f"`{diagnostic['status']}`，"
                f"`{diagnostic['summary_markdown']}`"
            )
    else:
        lines.append("- 无。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    arguments = parse_arguments()
    service_common = SCRIPTS_PATH / "service_benchmark_common.py"
    if common.sha256_file(service_common) != PINNED_SERVICE_COMMON_SHA256:
        raise RuntimeError(
            "shared service_benchmark_common.py changed; re-review before "
            "aggregating MetaX results"
        )
    selected: dict[str, dict[str, Any]] = {}
    sources = [_suite_path(value) for value in arguments.source_suite]
    for source in sources:
        summary = common.read_json(source / "summary.json")
        _validate_formal(summary, source)
        for raw_case in summary.get("cases", []):
            if (
                isinstance(raw_case, dict)
                and raw_case.get("status") == "passed"
                and raw_case.get("case_id") in common.EXPECTED_CASES
            ):
                case_id = raw_case["case_id"]
                if case_id in selected:
                    raise ValueError(f"duplicate passed case: {case_id}")
                case = common.collect_case(
                    raw_case, source_suite=source, platform=PLATFORM
                )
                server_log = Path(case["source"]["server_log"])
                if "profiling_debug_level: 0" not in server_log.read_text(
                    encoding="utf-8", errors="replace"
                ):
                    raise ValueError(
                        f"{case_id}: server log does not prove effective "
                        "profiling level 0"
                    )
                validation = case["artifact_validation"]
                case["status"] = (
                    "passed"
                    if validation["status"] == "passed"
                    else "failed_validation"
                )
                case.pop("dit_step_latency_seconds", None)
                case["source"].pop("mlu_samples", None)
                case["source"]["metax_samples"] = case["source"][
                    "device_samples"
                ]
                selected[case_id] = case

    missing = [
        case_id
        for case_id in common.EXPECTED_CASES
        if case_id not in selected
    ]
    cases = [
        selected[case_id]
        for case_id in common.EXPECTED_CASES
        if case_id in selected
    ]
    expected_artifacts = (
        len(common.EXPECTED_CASES) * common.MEASURED_REQUESTS
    )
    artifacts_verified = sum(
        case["artifact_validation"]["artifacts_verified"]
        for case in cases
    )
    media_verified = sum(
        case["artifact_validation"]["media_verified"] for case in cases
    )
    failed_validation = [
        case["case_id"] for case in cases if case["status"] != "passed"
    ]
    cases_passed = sum(case["status"] == "passed" for case in cases)
    complete = (
        not missing
        and not failed_validation
        and cases_passed == len(common.EXPECTED_CASES)
        and artifacts_verified == expected_artifacts
        and media_verified == expected_artifacts
    )
    report = {
        "schema_version": "1.0",
        "generated_at": common.utc_now(),
        "platform": PLATFORM,
        "platform_label": PLATFORM_LABEL,
        "status": "passed" if complete else "incomplete",
        "test_contract": {
            "warmup_samples": 1,
            "measured_samples": 10,
            "concurrency": 1,
            "profiling_debug_level": 0,
            "percentiles": ["p50", "p90"],
            "smoke_results_included": False,
            "cases": list(common.EXPECTED_CASES),
        },
        "cases_expected": len(common.EXPECTED_CASES),
        "cases_passed": cases_passed,
        "missing_cases": missing,
        "failed_validation_cases": failed_validation,
        "artifacts_expected": expected_artifacts,
        "artifacts_verified": artifacts_verified,
        "media_verified": media_verified,
        "source_suites": [str(path) for path in sources],
        "diagnostic_suites": _collect_diagnostics(
            arguments.diagnostic_suite
        ),
        "cases": cases,
    }
    output = arguments.output_dir.resolve()
    allowed_root = (
        REPO_PATH / "results" / "metax" / "server"
    ).resolve()
    if output != allowed_root and allowed_root not in output.parents:
        raise ValueError(
            f"--output-dir must stay under {allowed_root}, got {output}"
        )
    common.atomic_write(
        output / "final_report.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    common.atomic_write(
        output / "final_report.md", markdown_report(report)
    )
    print(
        f"Wrote {output}: {cases_passed}/"
        f"{len(common.EXPECTED_CASES)} passed"
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
