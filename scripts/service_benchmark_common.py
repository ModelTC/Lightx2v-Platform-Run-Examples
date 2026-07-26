#!/usr/bin/env python3
"""Shared client-side measurement logic for LightX2V service benchmarks."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

NO_PROXY_OPENER = build_opener(ProxyHandler({}))
SCHEMA_VERSION = "1.0"
PATH_FIELDS = {
    "audio_path",
    "image_path",
    "image_mask_path",
    "last_frame_path",
    "video_path",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def safe_filename_part(value: str) -> str:
    cleaned = [
        character
        if character.isalnum() or character in {"-", "_", "."}
        else "_"
        for character in value.strip()
    ]
    return "".join(cleaned).strip("_") or "run"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
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


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    atomic_write_text(path, text)


def resolve_payload_paths(
    payload: dict[str, Any], data_path: Path
) -> dict[str, Any]:
    resolved = dict(payload)
    for field in PATH_FIELDS:
        raw_value = resolved.get(field)
        if not isinstance(raw_value, str) or not raw_value:
            continue
        if raw_value.startswith(("http://", "https://", "data:")):
            continue
        candidate = Path(raw_value)
        if not candidate.is_absolute():
            candidate = data_path.parent / candidate
        resolved[field] = str(candidate.resolve())
    return resolved


def load_requests(
    path: Path,
    *,
    limit: int,
    repeat: int,
    allowed_fields: set[str],
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("--limit must be > 0")
    if repeat <= 0:
        raise ValueError("--repeat must be > 0")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            missing = {"prompt", "seed"} - set(row)
            unknown = set(row) - allowed_fields
            if missing:
                raise ValueError(
                    f"{path}:{line_number} missing required fields "
                    f"{sorted(missing)}"
                )
            if unknown:
                raise ValueError(
                    f"{path}:{line_number} has unsupported fields "
                    f"{sorted(unknown)}"
                )
            if not isinstance(row["prompt"], str):
                raise ValueError(
                    f"{path}:{line_number} prompt must be a string"
                )
            if isinstance(row["seed"], bool) or not isinstance(
                row["seed"], int
            ):
                raise ValueError(
                    f"{path}:{line_number} seed must be an integer"
                )
            resolved = resolve_payload_paths(row, path)
            for field in PATH_FIELDS & set(resolved):
                value = resolved[field]
                if (
                    isinstance(value, str)
                    and value
                    and not value.startswith(("http://", "https://", "data:"))
                    and not Path(value).is_file()
                ):
                    raise ValueError(
                        f"{path}:{line_number} {field} does not exist: "
                        f"{value}"
                    )
            rows.append(resolved)
            if len(rows) >= limit:
                break

    if not rows:
        raise ValueError(f"No request payloads loaded from {path}")
    if len(rows) < limit:
        raise ValueError(
            f"{path} contains only {len(rows)} usable rows, "
            f"but --limit={limit}"
        )
    return rows * repeat


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Return a linearly interpolated percentile, matching the old benchmark."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def parse_server_processing_seconds(
    start_time: Any, end_time: Any
) -> float | None:
    if not isinstance(start_time, str) or not isinstance(end_time, str):
        return None
    try:
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)
    except ValueError:
        return None
    duration = (end - start).total_seconds()
    return duration if duration >= 0 else None


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout_seconds: float,
) -> tuple[int, dict[str, Any]]:
    body = (
        None
        if payload is None
        else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    )
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with NO_PROXY_OPENER.open(
            request, timeout=timeout_seconds
        ) as response:
            data = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(data) if data else {}
            if not isinstance(parsed, dict):
                parsed = {"response": parsed}
            return response.getcode(), parsed
    except HTTPError as exception:
        data = exception.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            parsed = {"detail": data}
        if not isinstance(parsed, dict):
            parsed = {"response": parsed}
        return exception.code, parsed


@dataclass
class RequestResult:
    phase: str
    index: int
    task_id: str
    request_payload: dict[str, Any]
    ok: bool
    latency_s: float
    submit_latency_s: float
    status_code: int | None
    poll_count: int
    bytes_received: int
    server_start_time: str
    server_end_time: str
    server_processing_s: float | None
    save_result_path: str
    artifact_path: str
    artifact_bytes: int
    artifact_sha256: str
    started_at: str
    completed_at: str
    error: str


def make_task_id(
    run_id: str, case_id: str, phase: str, index: int
) -> str:
    raw = f"{run_id}_{case_id}_{phase}_{index:03d}"
    return safe_filename_part(raw)[:180]


def artifact_metadata(path: Path) -> tuple[int, str]:
    if not path.is_file():
        return 0, ""
    return path.stat().st_size, sha256_file(path)


def post_sync_image(
    *,
    phase: str,
    index: int,
    payload: dict[str, Any],
    endpoint: str,
    request_timeout_seconds: float,
    output_path: Path,
    task_id: str,
) -> RequestResult:
    request_payload = dict(payload)
    request_payload["task_id"] = task_id
    request_payload["save_result_path"] = str(output_path)
    body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = utc_now()
    started = time.perf_counter()
    status_code: int | None = None
    response_bytes = b""
    error = ""
    try:
        with NO_PROXY_OPENER.open(
            request, timeout=request_timeout_seconds
        ) as response:
            response_bytes = response.read()
            status_code = response.getcode()
            content_type = response.headers.get("Content-Type", "")
            if status_code == 200 and "image/png" in content_type:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(response_bytes)
            else:
                error = (
                    "unexpected response: "
                    f"status={status_code}, content_type={content_type!r}, "
                    f"preview={response_bytes[:200]!r}"
                )
    except HTTPError as exception:
        status_code = exception.code
        response_bytes = exception.read()
        error = response_bytes[:1000].decode(
            "utf-8", errors="replace"
        )
    except URLError as exception:
        error = str(exception.reason)
    except Exception as exception:
        error = repr(exception)

    latency_s = time.perf_counter() - started
    artifact_bytes, artifact_sha256 = artifact_metadata(output_path)
    ok = (
        not error
        and status_code == 200
        and artifact_bytes > 0
        and bool(artifact_sha256)
    )
    if not ok and not error:
        error = f"missing or empty output artifact: {output_path}"
    return RequestResult(
        phase=phase,
        index=index,
        task_id=task_id,
        request_payload=request_payload,
        ok=ok,
        latency_s=latency_s,
        submit_latency_s=latency_s,
        status_code=status_code,
        poll_count=0,
        bytes_received=len(response_bytes),
        server_start_time="",
        server_end_time="",
        server_processing_s=None,
        save_result_path=str(output_path),
        artifact_path=str(output_path),
        artifact_bytes=artifact_bytes,
        artifact_sha256=artifact_sha256,
        started_at=started_at,
        completed_at=utc_now(),
        error=error,
    )


def post_async_task(
    *,
    phase: str,
    index: int,
    payload: dict[str, Any],
    create_url: str,
    base_url: str,
    request_timeout_seconds: float,
    poll_interval_seconds: float,
    task_timeout_seconds: float,
    output_path: Path,
    task_id: str,
) -> RequestResult:
    request_payload = dict(payload)
    request_payload["task_id"] = task_id
    request_payload["save_result_path"] = str(output_path)
    started_at = utc_now()
    started = time.perf_counter()
    status_code: int | None = None
    submit_latency_s = 0.0
    poll_count = 0
    server_start_time = ""
    server_end_time = ""
    response_save_path = ""
    error = ""

    try:
        submit_started = time.perf_counter()
        status_code, response = http_json(
            "POST",
            create_url,
            request_payload,
            request_timeout_seconds,
        )
        submit_latency_s = time.perf_counter() - submit_started
        if status_code >= 400:
            error = json.dumps(response, ensure_ascii=False)
        elif str(response.get("task_id", "")) != task_id:
            error = (
                f"task_id mismatch: requested={task_id!r}, "
                f"response={response!r}"
            )
        else:
            response_save_path = str(response.get("save_result_path", ""))

        deadline = started + task_timeout_seconds
        last_status: dict[str, Any] = {}
        while not error and time.perf_counter() < deadline:
            status_url = (
                f"{base_url.rstrip('/')}/v1/tasks/{task_id}/status"
            )
            poll_status_code, last_status = http_json(
                "GET", status_url, None, request_timeout_seconds
            )
            poll_count += 1
            status_code = poll_status_code
            status = str(last_status.get("status", ""))
            if poll_status_code >= 400:
                error = json.dumps(last_status, ensure_ascii=False)
                break
            if status == "completed":
                response_save_path = str(
                    last_status.get(
                        "save_result_path", response_save_path
                    )
                    or ""
                )
                server_start_time = str(
                    last_status.get("start_time", "") or ""
                )
                server_end_time = str(
                    last_status.get("end_time", "") or ""
                )
                break
            if status in {"failed", "cancelled"}:
                error = str(
                    last_status.get("error")
                    or json.dumps(last_status, ensure_ascii=False)
                )
                server_start_time = str(
                    last_status.get("start_time", "") or ""
                )
                server_end_time = str(
                    last_status.get("end_time", "") or ""
                )
                break
            time.sleep(poll_interval_seconds)
        else:
            if not error:
                error = (
                    f"task timed out after {task_timeout_seconds}s; "
                    f"last_status={last_status}"
                )
    except URLError as exception:
        error = str(exception.reason)
    except Exception as exception:
        error = repr(exception)

    latency_s = time.perf_counter() - started
    artifact_bytes, artifact_sha256 = artifact_metadata(output_path)
    ok = (
        not error
        and status_code is not None
        and status_code < 400
        and artifact_bytes > 0
        and bool(artifact_sha256)
    )
    if not ok and not error:
        error = f"missing or empty output artifact: {output_path}"
    return RequestResult(
        phase=phase,
        index=index,
        task_id=task_id,
        request_payload=request_payload,
        ok=ok,
        latency_s=latency_s,
        submit_latency_s=submit_latency_s,
        status_code=status_code,
        poll_count=poll_count,
        bytes_received=0,
        server_start_time=server_start_time,
        server_end_time=server_end_time,
        server_processing_s=parse_server_processing_seconds(
            server_start_time, server_end_time
        ),
        save_result_path=response_save_path,
        artifact_path=str(output_path),
        artifact_bytes=artifact_bytes,
        artifact_sha256=artifact_sha256,
        started_at=started_at,
        completed_at=utc_now(),
        error=error,
    )


def distribution(values: Sequence[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    if not sorted_values:
        return {
            "p50_s": 0.0,
            "p90_s": 0.0,
            "avg_s": 0.0,
            "min_s": 0.0,
            "max_s": 0.0,
        }
    return {
        "p50_s": percentile(sorted_values, 50),
        "p90_s": percentile(sorted_values, 90),
        "avg_s": statistics.fmean(sorted_values),
        "min_s": min(sorted_values),
        "max_s": max(sorted_values),
    }


def summarize(
    *,
    results: Sequence[RequestResult],
    total_wall_s: float,
    first_success_s: float | None,
    run_config: dict[str, Any],
    unit_name: str,
) -> dict[str, Any]:
    successful = [result for result in results if result.ok]
    failed = [result for result in results if not result.ok]
    server_processing = [
        result.server_processing_s
        for result in successful
        if result.server_processing_s is not None
    ]
    throughput_per_second = (
        len(successful) / total_wall_s if total_wall_s > 0 else 0.0
    )
    valid = len(successful) == len(results) and bool(results)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if valid else "failed",
        "run_config": run_config,
        "requests_total": len(results),
        "requests_success": len(successful),
        "requests_failed": len(failed),
        "success_rate": (
            len(successful) / len(results) if results else 0.0
        ),
        "wall_time_s": total_wall_s,
        "end_to_end_latency": distribution(
            [result.latency_s for result in successful]
        ),
        "submit_latency": distribution(
            [result.submit_latency_s for result in successful]
        ),
        "server_processing_latency": (
            distribution(server_processing) if server_processing else None
        ),
        "time_to_first_success_s": first_success_s,
        "throughput": {
            f"{unit_name}_per_second": throughput_per_second,
            f"{unit_name}_per_minute": throughput_per_second * 60.0,
        },
        "failed_samples": [asdict(result) for result in failed],
        "request_results": [asdict(result) for result in results],
    }


def print_summary(summary: dict[str, Any], title: str) -> None:
    latency = summary["end_to_end_latency"]
    throughput = summary["throughput"]
    print(f"=== {title} ===")
    print(f"status:           {summary['status']}")
    print(f"requests_total:   {summary['requests_total']}")
    print(f"requests_success: {summary['requests_success']}")
    print(f"requests_failed:  {summary['requests_failed']}")
    print(f"success_rate:     {summary['success_rate']:.2%}")
    print(f"wall_time_s:      {summary['wall_time_s']:.4f}")
    print("")
    print("End-to-end latency, successful requests only:")
    print(f"  p50_s: {latency['p50_s']:.4f}")
    print(f"  p90_s: {latency['p90_s']:.4f}")
    print(f"  avg_s: {latency['avg_s']:.4f}")
    print(f"  min_s: {latency['min_s']:.4f}")
    print(f"  max_s: {latency['max_s']:.4f}")
    print("")
    first_success = summary["time_to_first_success_s"]
    print(
        f"time_to_first_success_s: {first_success:.4f}"
        if first_success is not None
        else "time_to_first_success_s: N/A"
    )
    for name, value in throughput.items():
        print(f"{name}: {value:.6f}")


def benchmark_output_dir(
    *,
    run_dir: Path | None,
    result_root: Path,
    platform: str,
    case_id: str,
    run_id: str,
    phase: str,
) -> Path:
    if run_dir is not None:
        return run_dir
    return (
        result_root
        / safe_filename_part(platform)
        / "server"
        / safe_filename_part(run_id)
        / safe_filename_part(case_id)
        / safe_filename_part(phase)
    )


def run_benchmark(
    *,
    task_kind: str,
    payloads: Sequence[dict[str, Any]],
    case_id: str,
    run_id: str,
    phase: str,
    concurrency: int,
    output_dir: Path,
    request_runner: Callable[
        [str, int, dict[str, Any], Path, str], RequestResult
    ],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    if concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "outputs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    extension = ".png" if task_kind == "t2i" else ".mp4"

    results: list[RequestResult] = []
    first_success_s: float | None = None
    benchmark_started = time.perf_counter()

    def run_one(index: int, payload: dict[str, Any]) -> RequestResult:
        task_id = make_task_id(run_id, case_id, phase, index)
        output_path = artifacts_dir / f"{index:03d}_{task_id}{extension}"
        print(
            f"[start] phase={phase} index={index} task_id={task_id}",
            flush=True,
        )
        return request_runner(
            phase, index, payload, output_path, task_id
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:
        future_to_index = {
            executor.submit(run_one, index, payload): index
            for index, payload in enumerate(payloads)
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_to_index), 1
        ):
            result = future.result()
            results.append(result)
            if result.ok and first_success_s is None:
                first_success_s = (
                    time.perf_counter() - benchmark_started
                )
            status = "ok" if result.ok else "fail"
            print(
                f"[{completed}/{len(payloads)}] {status} "
                f"index={result.index} task_id={result.task_id} "
                f"latency_s={result.latency_s:.4f} "
                f"submit_latency_s={result.submit_latency_s:.4f} "
                f"artifact={result.artifact_path}",
                flush=True,
            )
            if result.error:
                print(
                    f"[{completed}/{len(payloads)}] "
                    f"error index={result.index}: {result.error}",
                    flush=True,
                )

    total_wall_s = time.perf_counter() - benchmark_started
    results.sort(key=lambda result: result.index)
    unit_name = "images" if task_kind == "t2i" else "videos"
    summary = summarize(
        results=results,
        total_wall_s=total_wall_s,
        first_success_s=first_success_s,
        run_config=run_config,
        unit_name=unit_name,
    )
    result_path = output_dir / "result.json"
    requests_path = output_dir / "requests.jsonl"
    atomic_write_json(result_path, summary)
    write_jsonl(
        requests_path,
        [asdict(result) for result in results],
    )
    print_summary(
        summary,
        f"{task_kind.upper()} Service Benchmark Summary",
    )
    print(f"result_json:  {result_path}")
    print(f"requests_log: {requests_path}")
    return summary
