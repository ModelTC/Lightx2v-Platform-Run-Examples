#!/usr/bin/env python3
"""Benchmark LightX2V asynchronous T2V and S2V service endpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.service_benchmark_common import (
    atomic_write_json,
    benchmark_output_dir,
    load_requests,
    post_async_task,
    run_benchmark,
    safe_filename_part,
    sha256_file,
    utc_now,
)

REPO_PATH = Path("/data/wushuo1/Lightx2v-Platform-Run-Examples")
DEFAULT_DATA_PATH = REPO_PATH / "data" / "t2v_10.jsonl"
DEFAULT_RESULT_ROOT = REPO_PATH / "results"
DEFAULT_SOURCE_SCRIPT = (
    REPO_PATH
    / "scripts"
    / "ascend"
    / "infer"
    / "dist_8"
    / "run_wan21_1_3b_t2v_480p_81f_cfg2_sp4.sh"
)
DEFAULT_SERVICE_SCRIPT = (
    REPO_PATH
    / "scripts"
    / "ascend"
    / "server"
    / "dist"
    / "start_server_wan21_1_3b_t2v_480p_81f_cfg2_sp4.sh"
)
DEFAULT_MODEL_CONFIG = (
    REPO_PATH
    / "configs"
    / "ascend_npu"
    / "dist_8"
    / "wan21_1_3b_t2v_480p_81f_cfg2_sp4.json"
)
ALLOWED_FIELDS = {
    "prompt",
    "seed",
    "negative_prompt",
    "audio_path",
}


def file_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LightX2V async T2V/S2V service with traceable outputs."
        )
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="LightX2V server base URL",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=(
            "JSONL data; rows require prompt/seed and may contain audio_path"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of distinct input rows to measure",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the selected input rows",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent client workers",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=0.5,
        help="Task status polling interval",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout for submit/status requests",
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=float,
        default=7200.0,
        help="Maximum wait time for one generated video",
    )
    parser.add_argument(
        "--platform",
        default="ascend_npu",
        help="Platform recorded in the result",
    )
    parser.add_argument(
        "--case-id",
        default="wan21_1_3b_t2v_480p_81f_cfg2_sp4",
        help="Stable benchmark case identifier",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Suite/run identifier; generated when omitted",
    )
    parser.add_argument(
        "--phase",
        choices=("warmup", "measure"),
        default="measure",
        help="Phase label recorded in request results",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Root used when --run-dir is omitted",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Exact directory for result.json, requests.jsonl and outputs",
    )
    parser.add_argument(
        "--source-script",
        type=Path,
        default=DEFAULT_SOURCE_SCRIPT,
        help="Matching offline inference script",
    )
    parser.add_argument(
        "--service-script",
        type=Path,
        default=DEFAULT_SERVICE_SCRIPT,
        help="Service startup script under test",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Model config loaded by the service",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional extra copy of result.json",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    if arguments.poll_interval_seconds <= 0:
        raise ValueError("--poll-interval-seconds must be > 0")
    if arguments.request_timeout_seconds <= 0:
        raise ValueError("--request-timeout-seconds must be > 0")
    if arguments.task_timeout_seconds <= 0:
        raise ValueError("--task-timeout-seconds must be > 0")

    for path_name in (
        "data",
        "source_script",
        "service_script",
        "model_config",
    ):
        path = getattr(arguments, path_name)
        if not path.is_file():
            raise FileNotFoundError(f"{path_name} does not exist: {path}")

    payloads = load_requests(
        arguments.data,
        limit=arguments.limit,
        repeat=arguments.repeat,
        allowed_fields=ALLOWED_FIELDS,
    )
    run_id = arguments.run_id or safe_filename_part(utc_now())
    output_dir = benchmark_output_dir(
        run_dir=arguments.run_dir,
        result_root=arguments.result_root,
        platform=arguments.platform,
        case_id=arguments.case_id,
        run_id=run_id,
        phase=arguments.phase,
    )
    create_url = f"{arguments.url.rstrip('/')}/v1/tasks/video/"
    run_config = {
        "started_at": utc_now(),
        "platform": arguments.platform,
        "case_id": arguments.case_id,
        "run_id": run_id,
        "phase": arguments.phase,
        "task_kind": "s2v"
        if any("audio_path" in payload for payload in payloads)
        else "t2v",
        "request_mode": "async",
        "create_endpoint": create_url,
        "status_endpoint_template": (
            f"{arguments.url.rstrip('/')}/v1/tasks/{{task_id}}/status"
        ),
        "request_count": len(payloads),
        "concurrency": arguments.concurrency,
        "limit": arguments.limit,
        "repeat": arguments.repeat,
        "poll_interval_seconds": arguments.poll_interval_seconds,
        "request_timeout_seconds": arguments.request_timeout_seconds,
        "task_timeout_seconds": arguments.task_timeout_seconds,
        "benchmark_script": file_record(Path(__file__)),
        "source_script": file_record(arguments.source_script),
        "service_script": file_record(arguments.service_script),
        "model_config": file_record(arguments.model_config),
        "data": file_record(arguments.data),
        "output_dir": str(output_dir.resolve()),
    }

    def request_runner(
        phase: str,
        index: int,
        payload: dict[str, object],
        output_path: Path,
        task_id: str,
    ):
        return post_async_task(
            phase=phase,
            index=index,
            payload=payload,
            create_url=create_url,
            base_url=arguments.url,
            request_timeout_seconds=arguments.request_timeout_seconds,
            poll_interval_seconds=arguments.poll_interval_seconds,
            task_timeout_seconds=arguments.task_timeout_seconds,
            output_path=output_path,
            task_id=task_id,
        )

    print(f"create_endpoint: {create_url}")
    print(f"case_id:         {arguments.case_id}")
    print(f"phase:           {arguments.phase}")
    print(f"requests:        {len(payloads)}")
    print(f"concurrency:     {arguments.concurrency}")
    print(f"result_dir:      {output_dir}")
    print("")
    summary = run_benchmark(
        task_kind=run_config["task_kind"],
        payloads=payloads,
        case_id=arguments.case_id,
        run_id=run_id,
        phase=arguments.phase,
        concurrency=arguments.concurrency,
        output_dir=output_dir,
        request_runner=request_runner,
        run_config=run_config,
    )
    if arguments.output_json is not None:
        atomic_write_json(arguments.output_json, summary)
        print(f"extra_result_json: {arguments.output_json}")
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
