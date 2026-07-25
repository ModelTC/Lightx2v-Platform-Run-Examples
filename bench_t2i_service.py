#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

DEFAULT_DATA_PATH = Path("/data/nvme1/wushuo/LightX2V-TestSpeed/data/t2i_100.jsonl")
DEFAULT_RESULT_ROOT = Path("/data/nvme1/wushuo/LightX2V-TestSpeed/result")
DEFAULT_SOURCE_SCRIPT = Path("/data/nvme1/wushuo/LightX2V/scripts/platforms/nvidia/infer_flux2_dev.sh")
DEFAULT_SERVICE_SCRIPT = Path("/data/nvme1/wushuo/LightX2V-TestSpeed/scripts/nvidia/start_server_flux2_dev.sh")
DEFAULT_MODEL_CONFIG = Path("/data/nvme1/wushuo/LightX2V/configs/platforms/nvidia/single/flux2_dev.json")
NO_PROXY_OPENER = build_opener(ProxyHandler({}))


@dataclass
class RequestResult:
    index: int
    ok: bool
    latency_s: float
    status_code: int | None
    bytes_received: int
    error: str
    submit_latency_s: float = 0.0
    task_id: str = ""
    save_result_path: str = ""


def load_requests(path: Path, limit: int = 0, repeat: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            keys = set(row)
            if keys != {"prompt", "seed"}:
                raise ValueError(f"{path}:{lineno} expected only prompt/seed, got keys={sorted(keys)}")
            rows.append(row)

    if limit > 0:
        rows = rows[:limit]
    if repeat <= 0:
        raise ValueError("--repeat must be > 0")
    return rows * repeat


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def http_json(method: str, url: str, payload: dict[str, Any] | None, timeout_s: float) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with NO_PROXY_OPENER.open(request, timeout=timeout_s) as response:
            data = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(data) if data else {}
            return response.getcode(), parsed
    except HTTPError as exc:
        data = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            parsed = {"detail": data}
        return exc.code, parsed


def post_sync_image(
    index: int,
    payload: dict[str, Any],
    endpoint: str,
    request_timeout_s: float,
    save_images_dir: Path | None,
) -> RequestResult:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with NO_PROXY_OPENER.open(request, timeout=request_timeout_s) as response:
            data = response.read()
            latency_s = time.perf_counter() - started
            status_code = response.getcode()
            content_type = response.headers.get("Content-Type", "")
            ok = status_code == 200 and "image/png" in content_type
            error = "" if ok else f"unexpected response: status={status_code}, content_type={content_type!r}, preview={data[:200]!r}"
            if ok and save_images_dir is not None:
                save_images_dir.mkdir(parents=True, exist_ok=True)
                (save_images_dir / f"t2i_{index:06d}.png").write_bytes(data)
            return RequestResult(index, ok, latency_s, status_code, len(data), error)
    except HTTPError as exc:
        data = exc.read()
        latency_s = time.perf_counter() - started
        return RequestResult(index, False, latency_s, exc.code, len(data), data[:500].decode("utf-8", errors="replace"))
    except URLError as exc:
        latency_s = time.perf_counter() - started
        return RequestResult(index, False, latency_s, None, 0, str(exc.reason))
    except Exception as exc:
        latency_s = time.perf_counter() - started
        return RequestResult(index, False, latency_s, None, 0, repr(exc))


def post_async_image_task(
    index: int,
    payload: dict[str, Any],
    base_url: str,
    request_timeout_s: float,
    poll_interval_s: float,
    task_timeout_s: float,
) -> RequestResult:
    started = time.perf_counter()
    create_url = f"{base_url.rstrip('/')}/v1/tasks/image/"
    status_code: int | None = None
    task_id = ""
    save_result_path = ""

    try:
        create_started = time.perf_counter()
        status_code, response = http_json("POST", create_url, payload, request_timeout_s)
        submit_latency_s = time.perf_counter() - create_started
        if status_code >= 400:
            return RequestResult(index, False, time.perf_counter() - started, status_code, 0, json.dumps(response, ensure_ascii=False), submit_latency_s, "", "")

        task_id = str(response.get("task_id", ""))
        save_result_path = str(response.get("save_result_path", ""))
        if not task_id:
            return RequestResult(index, False, time.perf_counter() - started, status_code, 0, f"missing task_id in response: {response}", submit_latency_s, "", save_result_path)

        deadline = started + task_timeout_s
        last_status: dict[str, Any] = {}
        while time.perf_counter() < deadline:
            status_url = f"{base_url.rstrip('/')}/v1/tasks/{task_id}/status"
            poll_status_code, last_status = http_json("GET", status_url, None, request_timeout_s)
            status = str(last_status.get("status", ""))
            if poll_status_code >= 400:
                return RequestResult(index, False, time.perf_counter() - started, poll_status_code, 0, json.dumps(last_status, ensure_ascii=False), submit_latency_s, task_id, save_result_path)
            if status == "completed":
                save_result_path = str(last_status.get("save_result_path", save_result_path) or "")
                return RequestResult(index, True, time.perf_counter() - started, poll_status_code, 0, "", submit_latency_s, task_id, save_result_path)
            if status in {"failed", "cancelled"}:
                error = last_status.get("error") or json.dumps(last_status, ensure_ascii=False)
                return RequestResult(index, False, time.perf_counter() - started, poll_status_code, 0, str(error), submit_latency_s, task_id, save_result_path)
            time.sleep(poll_interval_s)

        return RequestResult(
            index,
            False,
            time.perf_counter() - started,
            status_code,
            0,
            f"task timed out after {task_timeout_s}s; last_status={last_status}",
            submit_latency_s,
            task_id,
            save_result_path,
        )
    except URLError as exc:
        return RequestResult(index, False, time.perf_counter() - started, status_code, 0, str(exc.reason), 0.0, task_id, save_result_path)
    except Exception as exc:
        return RequestResult(index, False, time.perf_counter() - started, status_code, 0, repr(exc), 0.0, task_id, save_result_path)


def safe_filename_part(value: str) -> str:
    cleaned = []
    for char in value.strip():
        if char.isalnum() or char in {"-", "_", "."}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "run"


def build_run_config(args: argparse.Namespace, endpoint: str, request_count: int, timestamp: str) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "platform": args.platform,
        "benchmark_script": str(Path(__file__).resolve()),
        "source_script": str(args.source_script),
        "service_script": str(args.service_script),
        "model_config": str(args.model_config),
        "data": str(args.data),
        "url": args.url,
        "request_mode": args.request_mode,
        "endpoint": endpoint if args.request_mode == "sync" else "",
        "create_endpoint": f"{args.url.rstrip('/')}/v1/tasks/image/" if args.request_mode == "async" else "",
        "status_endpoint_template": f"{args.url.rstrip('/')}/v1/tasks/{{task_id}}/status" if args.request_mode == "async" else "",
        "request_count": request_count,
        "concurrency": args.concurrency,
        "limit": args.limit,
        "repeat": args.repeat,
        "timeout_seconds": args.timeout_seconds,
        "poll_interval_seconds": args.poll_interval_seconds,
        "request_timeout_seconds": args.request_timeout_seconds,
        "task_timeout_seconds": args.task_timeout_seconds,
        "save_images_dir": str(args.save_images_dir) if args.save_images_dir else None,
    }


def default_result_path(args: argparse.Namespace, timestamp: str, request_count: int) -> Path:
    platform_dir = args.result_root / safe_filename_part(args.platform)
    data_name = safe_filename_part(args.data.stem)
    config_name = safe_filename_part(args.model_config.stem)
    filename = f"{timestamp}_{data_name}_{config_name}_c{args.concurrency}_n{request_count}.json"
    return platform_dir / filename


def summarize(
    results: list[RequestResult],
    total_wall_s: float,
    first_success_s: float | None,
    first_success_request_latency_s: float | None,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    ok_results = [r for r in results if r.ok]
    failed_results = [r for r in results if not r.ok]
    latencies = sorted(r.latency_s for r in ok_results)
    success_count = len(ok_results)

    if latencies:
        latency_summary = {
            "p50_s": percentile(latencies, 50),
            "p90_s": percentile(latencies, 90),
            "p95_s": percentile(latencies, 95),
            "p99_s": percentile(latencies, 99),
            "avg_s": statistics.fmean(latencies),
            "max_s": max(latencies),
            "min_s": min(latencies),
        }
    else:
        latency_summary = {
            "p50_s": 0.0,
            "p90_s": 0.0,
            "p95_s": 0.0,
            "p99_s": 0.0,
            "avg_s": 0.0,
            "max_s": 0.0,
            "min_s": 0.0,
        }

    images_per_second = success_count / total_wall_s if total_wall_s > 0 else 0.0
    return {
        "run_config": run_config,
        "requests_total": len(results),
        "requests_success": success_count,
        "requests_failed": len(failed_results),
        "wall_time_s": total_wall_s,
        "end_to_end_latency": latency_summary,
        "first_image_time_s": first_success_s,
        "first_success_request_latency_s": first_success_request_latency_s,
        "throughput": {
            "images_per_second": images_per_second,
            "images_per_minute": images_per_second * 60.0,
        },
        "failed_samples": [asdict(r) for r in failed_results[:10]],
        "request_results": [asdict(r) for r in results],
    }


def print_summary(summary: dict[str, Any]) -> None:
    latency = summary["end_to_end_latency"]
    throughput = summary["throughput"]
    print("=== T2I Service Benchmark Summary ===")
    print(f"requests_total:   {summary['requests_total']}")
    print(f"requests_success: {summary['requests_success']}")
    print(f"requests_failed:  {summary['requests_failed']}")
    print(f"wall_time_s:      {summary['wall_time_s']:.4f}")
    print("")
    print("End-to-end latency, successful requests only:")
    print(f"  p50_s: {latency['p50_s']:.4f}")
    print(f"  p90_s: {latency['p90_s']:.4f}")
    print(f"  p95_s: {latency['p95_s']:.4f}")
    print(f"  p99_s: {latency['p99_s']:.4f}")
    print(f"  avg_s: {latency['avg_s']:.4f}")
    print(f"  max_s: {latency['max_s']:.4f}")
    print("")
    first_image_time = summary["first_image_time_s"]
    first_request_latency = summary["first_success_request_latency_s"]
    print("First image / first response:")
    print(f"  time_to_first_success_s: {first_image_time:.4f}" if first_image_time is not None else "  time_to_first_success_s: N/A")
    print(f"  first_success_request_latency_s: {first_request_latency:.4f}" if first_request_latency is not None else "  first_success_request_latency_s: N/A")
    print("")
    print("Throughput:")
    print(f"  images_per_second: {throughput['images_per_second']:.6f}")
    print(f"  images_per_minute: {throughput['images_per_minute']:.6f}")

    if summary["failed_samples"]:
        print("")
        print("Failed samples, first 10:")
        for item in summary["failed_samples"]:
            print(json.dumps(item, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark LightX2V T2I sync image service.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="LightX2V server base URL")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="JSONL data path; each row must contain prompt and seed")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent client workers")
    parser.add_argument("--limit", type=int, default=0, help="Use only first N rows; 0 means all rows")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the loaded data N times")
    parser.add_argument("--timeout-seconds", type=int, default=600, help="Server sync timeout_seconds query parameter")
    parser.add_argument("--poll-interval-seconds", type=float, default=0.5, help="Server poll_interval_seconds query parameter")
    parser.add_argument("--request-timeout-seconds", type=float, default=660.0, help="HTTP client timeout per request")
    parser.add_argument("--task-timeout-seconds", type=float, default=3600.0, help="Max wait time for one async image task")
    parser.add_argument("--request-mode", choices=("sync", "async"), default="sync", help="Use sync PNG response or async task polling for image requests")
    parser.add_argument("--save-images-dir", type=Path, default=None, help="Optional directory to save returned PNG files in sync mode")
    parser.add_argument("--platform", default="nvidia", help="Platform name used under result/{platform}")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT, help="Root directory for benchmark result files")
    parser.add_argument("--source-script", type=Path, default=DEFAULT_SOURCE_SCRIPT, help="Original inference script represented by this benchmark run")
    parser.add_argument("--service-script", type=Path, default=DEFAULT_SERVICE_SCRIPT, help="Service startup script used for this benchmark run")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG, help="Model config JSON loaded by the service script")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional extra path to write benchmark summary JSON")
    args = parser.parse_args()

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0")
    if args.poll_interval_seconds <= 0:
        raise ValueError("--poll-interval-seconds must be > 0")
    if args.request_timeout_seconds <= 0:
        raise ValueError("--request-timeout-seconds must be > 0")
    if args.task_timeout_seconds <= 0:
        raise ValueError("--task-timeout-seconds must be > 0")

    payloads = load_requests(args.data, limit=args.limit, repeat=args.repeat)
    if not payloads:
        raise ValueError(f"No request payloads loaded from {args.data}")

    query = urlencode(
        {
            "timeout_seconds": args.timeout_seconds,
            "poll_interval_seconds": args.poll_interval_seconds,
        }
    )
    endpoint = f"{args.url.rstrip('/')}/v1/tasks/image/sync?{query}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_config = build_run_config(args, endpoint, len(payloads), timestamp)
    result_path = default_result_path(args, timestamp, len(payloads))

    print(f"request_mode: {args.request_mode}")
    if args.request_mode == "sync":
        print(f"endpoint:     {endpoint}")
    else:
        print(f"create_endpoint: {run_config['create_endpoint']}")
        print(f"status_endpoint: {run_config['status_endpoint_template']}")
    print(f"data:         {args.data}")
    print(f"requests:     {len(payloads)}")
    print(f"concurrency:  {args.concurrency}")
    print(f"platform:     {args.platform}")
    print(f"result_json:  {result_path}")
    print("", flush=True)

    results: list[RequestResult] = []
    first_success_s: float | None = None
    first_success_request_latency_s: float | None = None
    bench_started = time.perf_counter()

    def run_one(index: int, payload: dict[str, Any]) -> RequestResult:
        print(f"[start] index={index}", flush=True)
        if args.request_mode == "sync":
            return post_sync_image(index, payload, endpoint, args.request_timeout_seconds, args.save_images_dir)
        return post_async_image_task(index, payload, args.url, args.request_timeout_seconds, args.poll_interval_seconds, args.task_timeout_seconds)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_index = {executor.submit(run_one, index, payload): index for index, payload in enumerate(payloads)}
        for completed, future in enumerate(concurrent.futures.as_completed(future_to_index), 1):
            result = future.result()
            results.append(result)
            if result.ok and first_success_s is None:
                first_success_s = time.perf_counter() - bench_started
                first_success_request_latency_s = result.latency_s
            status = "ok" if result.ok else "fail"
            task_part = f" task_id={result.task_id}" if result.task_id else ""
            save_part = f" save_result_path={result.save_result_path}" if result.save_result_path else ""
            print(f"[{completed}/{len(payloads)}] {status} index={result.index}{task_part} latency_s={result.latency_s:.4f} bytes={result.bytes_received}{save_part}", flush=True)
            if result.error:
                print(f"[{completed}/{len(payloads)}] error index={result.index}: {result.error}", flush=True)

    total_wall_s = time.perf_counter() - bench_started
    results.sort(key=lambda r: r.index)
    summary = summarize(results, total_wall_s, first_success_s, first_success_request_latency_s, run_config)
    print("")
    print_summary(summary)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved result JSON to: {result_path}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved extra summary JSON to: {args.output_json}")

    return 0 if summary["requests_failed"] == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
