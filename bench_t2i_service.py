#!/usr/bin/env python3
"""Benchmark the LightX2V synchronous T2I service endpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlencode

from scripts.service_benchmark_common import (
    atomic_write_json,
    benchmark_output_dir,
    load_requests,
    post_sync_image,
    run_benchmark,
    safe_filename_part,
    sha256_file,
    utc_now,
)

REPO_PATH = Path("/data/wushuo1/Lightx2v-Platform-Run-Examples")
DEFAULT_DATA_PATH = REPO_PATH / "data" / "t2i_100.jsonl"
DEFAULT_RESULT_ROOT = REPO_PATH / "results"
DEFAULT_SOURCE_SCRIPT = (
    REPO_PATH
    / "scripts"
    / "ascend"
    / "infer"
    / "dist_8"
    / "run_flux2_dev_t2i_1344x768_tp8.sh"
)
DEFAULT_SERVICE_SCRIPT = (
    REPO_PATH
    / "scripts"
    / "ascend"
    / "server"
    / "dist"
    / "start_server_flux2_dev_t2i_1344x768_tp8.sh"
)
DEFAULT_MODEL_CONFIG = (
    REPO_PATH
    / "configs"
    / "ascend_npu"
    / "dist_8"
    / "flux2_dev_t2i_1344x768_tp8.json"
)
ALLOWED_FIELDS = {"prompt", "seed", "negative_prompt"}


def file_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LightX2V T2I sync service with traceable outputs."
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
        help="JSONL data; rows require prompt and seed",
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
        "--timeout-seconds",
        type=int,
        default=3600,
        help="Server-side sync task timeout",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=0.5,
        help="Server-side sync status polling interval",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=3660.0,
        help="HTTP timeout for one complete sync request",
    )
    parser.add_argument(
        "--platform",
        default="ascend_npu",
        help="Platform recorded in the result",
    )
    parser.add_argument(
        "--case-id",
        default="flux2_dev_t2i_1344x768_tp8",
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
    if arguments.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0")
    if arguments.poll_interval_seconds <= 0:
        raise ValueError("--poll-interval-seconds must be > 0")
    if arguments.request_timeout_seconds <= 0:
        raise ValueError("--request-timeout-seconds must be > 0")

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
    query = urlencode(
        {
            "timeout_seconds": arguments.timeout_seconds,
            "poll_interval_seconds": arguments.poll_interval_seconds,
        }
    )
    endpoint = (
        f"{arguments.url.rstrip('/')}/v1/tasks/image/sync?{query}"
    )
    run_config = {
        "started_at": utc_now(),
        "platform": arguments.platform,
        "case_id": arguments.case_id,
        "run_id": run_id,
        "phase": arguments.phase,
        "task_kind": "t2i",
        "request_mode": "sync",
        "endpoint": endpoint,
        "request_count": len(payloads),
        "concurrency": arguments.concurrency,
        "limit": arguments.limit,
        "repeat": arguments.repeat,
        "timeout_seconds": arguments.timeout_seconds,
        "poll_interval_seconds": arguments.poll_interval_seconds,
        "request_timeout_seconds": arguments.request_timeout_seconds,
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
        return post_sync_image(
            phase=phase,
            index=index,
            payload=payload,
            endpoint=endpoint,
            request_timeout_seconds=arguments.request_timeout_seconds,
            output_path=output_path,
            task_id=task_id,
        )

    print(f"endpoint:      {endpoint}")
    print(f"case_id:       {arguments.case_id}")
    print(f"phase:         {arguments.phase}")
    print(f"requests:      {len(payloads)}")
    print(f"concurrency:   {arguments.concurrency}")
    print(f"result_dir:    {output_dir}")
    print("")
    summary = run_benchmark(
        task_kind="t2i",
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
