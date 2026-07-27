#!/usr/bin/env python3
"""Merge successful service suites into one traceable benchmark report."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from service_benchmark_common import sha256_file

REPO_PATH = Path("/data/wushuo1/Lightx2v-Platform-Run-Examples")
RESULT_ROOT = REPO_PATH / "results" / "ascend_npu" / "server"
EXPECTED_CASES = (
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
)
REMEDIATIONS = (
    {
        "scope": "服务测试清理",
        "problem": (
            "失败用例可能残留服务进程，在下一用例开始前继续占用 NPU 显存。"
        ),
        "resolution": (
            "按精确的 suite run ID 追踪并终止该批次拥有的服务子进程，"
            "随后检查 NPU 显存。"
        ),
    },
    {
        "scope": "Ascend 64GB 上的 LTX2.3 S2V",
        "problem": (
            "全量权重 241 帧配置会在 Gemma 编码阶段显存不足，"
            "或在 VAE 分块解码时保留计算图。"
        ),
        "resolution": (
            "采用 Gemma 分阶段 CPU offload，移除未使用的视觉与 logits 路径，"
            "在 inference mode 下执行 VAE 分块解码，并启用 torch_npu "
            "expandable-segments 分配器。正式结果仍使用全量权重、241 帧、"
            "30 个扩散步，且 double_precision_rope=false。"
        ),
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file_obj:
            file_obj.write(text)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def suite_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = RESULT_ROOT / path
    return path.resolve(strict=True)


def throughput_per_minute(throughput: dict[str, Any]) -> float:
    for key, value in throughput.items():
        if key.endswith("_per_minute") and isinstance(value, (int, float)):
            return float(value)
    raise ValueError(f"missing per-minute throughput: {throughput}")


def verify_case_artifacts(case: dict[str, Any]) -> dict[str, Any]:
    benchmark = case["benchmark_result"]
    failures: list[str] = []
    total_bytes = 0
    verified = 0
    for request in benchmark.get("request_results", []):
        artifact_path = Path(request.get("artifact_path", ""))
        expected_sha = request.get("artifact_sha256")
        expected_bytes = request.get("artifact_bytes")
        if not request.get("ok"):
            failures.append(f"{request.get('task_id')}: request is not successful")
            continue
        if not artifact_path.is_file():
            failures.append(f"{artifact_path}: missing")
            continue
        actual_bytes = artifact_path.stat().st_size
        if actual_bytes <= 0 or actual_bytes != expected_bytes:
            failures.append(
                f"{artifact_path}: size {actual_bytes} != {expected_bytes}"
            )
            continue
        actual_sha = sha256_file(artifact_path)
        if actual_sha != expected_sha:
            failures.append(
                f"{artifact_path}: sha256 {actual_sha} != {expected_sha}"
            )
            continue
        verified += 1
        total_bytes += actual_bytes
    return {
        "status": "passed" if not failures and verified == 10 else "failed",
        "artifacts_verified": verified,
        "artifact_bytes": total_bytes,
        "failures": failures,
    }


def collect_case(
    case: dict[str, Any],
    *,
    source_suite: Path,
) -> dict[str, Any]:
    benchmark = case.get("benchmark_result")
    if case.get("status") != "passed" or not isinstance(benchmark, dict):
        raise ValueError(f"{case.get('case_id')}: selected case is not passed")
    if (
        benchmark.get("status") != "passed"
        or benchmark.get("requests_total") != 10
        or benchmark.get("requests_success") != 10
        or benchmark.get("requests_failed") != 0
    ):
        raise ValueError(
            f"{case.get('case_id')}: benchmark is not a clean 10/10 pass"
        )

    result_dir = Path(case["result_dir"])
    log_dir = Path(case["log_dir"])
    warmup = read_json(result_dir / "warmup" / "result.json")
    if (
        warmup.get("status") != "passed"
        or warmup.get("requests_success") != 1
        or warmup.get("requests_failed") != 0
    ):
        raise ValueError(f"{case.get('case_id')}: warm-up is not a clean pass")

    artifact_validation = verify_case_artifacts(case)
    if artifact_validation["status"] != "passed":
        raise ValueError(
            f"{case.get('case_id')}: artifact verification failed: "
            f"{artifact_validation['failures']}"
        )

    peak_by_device = case.get("peak_hbm_mb", {})
    latency = benchmark["end_to_end_latency"]
    return {
        "case_id": case["case_id"],
        "task_kind": case["task_kind"],
        "world_size": case["world_size"],
        "parallel_strategy": case["parallel_strategy"],
        "status": "passed",
        "requests": {
            "warmup": 1,
            "measured": benchmark["requests_total"],
            "success": benchmark["requests_success"],
            "failed": benchmark["requests_failed"],
        },
        "startup_seconds": case["startup_seconds"],
        "warmup_seconds": warmup["wall_time_s"],
        "latency_seconds": latency,
        "throughput_per_minute": throughput_per_minute(
            benchmark["throughput"]
        ),
        "peak_hbm_mb": {
            "max": max(peak_by_device.values()),
            "by_device": peak_by_device,
        },
        "artifact_validation": artifact_validation,
        "source": {
            "suite_id": source_suite.name,
            "suite_summary": str(source_suite / "summary.json"),
            "suite_manifest": str(source_suite / "manifest.json"),
            "result": str(result_dir / "measure" / "result.json"),
            "requests": str(result_dir / "measure" / "requests.jsonl"),
            "outputs": str(result_dir / "measure" / "outputs"),
            "server_log": str(log_dir / "server.log"),
            "client_log": str(log_dir / "client.log"),
            "warmup_log": str(log_dir / "warmup.log"),
            "npu_samples": str(log_dir / "npu_samples.csv"),
        },
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Ascend NPU 服务化推理性能报告",
        "",
        f"- 生成时间：`{report['generated_at']}`",
        "- 测试口径：1 次预热（不计入统计）+ 10 个正式样本，并发 1",
        f"- 总体结果：`{report['status']}`，"
        f"{report['cases_passed']}/{report['cases_expected']} 个配置通过，"
        f"{report['artifacts_verified']} 个正式产物通过 SHA256 复核",
        "- 延迟：客户端端到端时延，P50/P90 仅由 10 个成功正式样本计算",
        "",
        "| 配置 | 任务 | 卡数 | 并行 | P50(s) | P90(s) | Avg(s) | Min(s) | Max(s) | 吞吐/分 | 启动(s) | 预热(s) | 峰值HBM(MB) |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        latency = case["latency_seconds"]
        lines.append(
            f"| {case['case_id']} | {case['task_kind']} | "
            f"{case['world_size']} | {case['parallel_strategy']} | "
            f"{latency['p50_s']:.4f} | {latency['p90_s']:.4f} | "
            f"{latency['avg_s']:.4f} | {latency['min_s']:.4f} | "
            f"{latency['max_s']:.4f} | "
            f"{case['throughput_per_minute']:.6f} | "
            f"{case['startup_seconds']:.2f} | "
            f"{case['warmup_seconds']:.2f} | "
            f"{case['peak_hbm_mb']['max']} |"
        )

    lines.extend(["", "## 测试中修复", ""])
    for remediation in report["remediations"]:
        lines.extend(
            [
                f"- `{remediation['scope']}`：{remediation['problem']} "
                f"{remediation['resolution']}",
            ]
        )

    lines.extend(["", "## 结果追溯", ""])
    for case in report["cases"]:
        source = case["source"]
        lines.extend(
            [
                f"### {case['case_id']}",
                "",
                f"- 原始结果：`{source['result']}`",
                f"- 请求明细：`{source['requests']}`",
                f"- 正式产物：`{source['outputs']}`",
                f"- 服务日志：`{source['server_log']}`",
                f"- 客户端日志：`{source['client_log']}`",
                f"- NPU 采样：`{source['npu_samples']}`",
                f"- 来源批次：`{source['suite_id']}`",
                "",
            ]
        )

    lines.extend(["## 问题诊断批次", ""])
    for diagnostic in report["diagnostic_suites"]:
        lines.append(
            f"- `{diagnostic['suite_id']}`：`{diagnostic['status']}`，"
            f"`{diagnostic['summary_markdown']}`"
        )
    lines.extend(
        [
            "",
            "这些失败/中断批次不参与性能统计，但保留了每次修复前的日志、"
            "配置快照和环境信息。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite", action="append", required=True)
    parser.add_argument("--diagnostic-suite", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected: dict[str, dict[str, Any]] = {}
    source_suites = [suite_path(value) for value in args.source_suite]
    for source_suite in source_suites:
        summary = read_json(source_suite / "summary.json")
        for raw_case in summary.get("cases", []):
            if (
                raw_case.get("status") == "passed"
                and raw_case.get("case_id") in EXPECTED_CASES
            ):
                case_id = raw_case["case_id"]
                if case_id in selected:
                    raise ValueError(f"duplicate passed case: {case_id}")
                selected[case_id] = collect_case(
                    raw_case, source_suite=source_suite
                )

    missing = [case_id for case_id in EXPECTED_CASES if case_id not in selected]
    diagnostics = []
    for value in args.diagnostic_suite:
        path = suite_path(value)
        summary = read_json(path / "summary.json")
        diagnostics.append(
            {
                "suite_id": path.name,
                "status": summary.get("status"),
                "summary_json": str(path / "summary.json"),
                "summary_markdown": str(path / "summary.md"),
                "log_root": summary.get("log_root"),
            }
        )

    cases = [selected[case_id] for case_id in EXPECTED_CASES if case_id in selected]
    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": "passed" if not missing else "incomplete",
        "test_contract": {
            "warmup_samples": 1,
            "measured_samples": 10,
            "concurrency": 1,
            "percentiles": ["p50", "p90"],
            "p99_measured": False,
        },
        "cases_expected": len(EXPECTED_CASES),
        "cases_passed": len(cases),
        "missing_cases": missing,
        "artifacts_verified": sum(
            case["artifact_validation"]["artifacts_verified"] for case in cases
        ),
        "remediations": REMEDIATIONS,
        "source_suites": [str(path) for path in source_suites],
        "diagnostic_suites": diagnostics,
        "cases": cases,
    }

    output_dir = args.output_dir.resolve()
    atomic_write(
        output_dir / "final_report.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write(output_dir / "final_report.md", markdown_report(report))
    print(output_dir / "final_report.md")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
