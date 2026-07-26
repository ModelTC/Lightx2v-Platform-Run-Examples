import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import service_benchmark_common as benchmark


def successful_result(index: int, latency: float) -> benchmark.RequestResult:
    return benchmark.RequestResult(
        phase="measure",
        index=index,
        task_id=f"task-{index}",
        request_payload={"prompt": f"prompt-{index}", "seed": index},
        ok=True,
        latency_s=latency,
        submit_latency_s=0.01,
        status_code=200,
        poll_count=1,
        bytes_received=0,
        server_start_time="2026-01-01T00:00:00",
        server_end_time="2026-01-01T00:00:01",
        server_processing_s=1.0,
        save_result_path=f"{index}.mp4",
        artifact_path=f"{index}.mp4",
        artifact_bytes=100,
        artifact_sha256="a" * 64,
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:01Z",
        error="",
    )


class ServiceBenchmarkCommonTest(unittest.TestCase):
    def test_percentiles_keep_only_requested_p50_and_p90(self):
        summary = benchmark.summarize(
            results=[successful_result(index, float(index + 1)) for index in range(10)],
            total_wall_s=100.0,
            first_success_s=1.0,
            run_config={"case_id": "case"},
            unit_name="videos",
        )

        latency = summary["end_to_end_latency"]
        self.assertEqual(
            set(latency),
            {"p50_s", "p90_s", "avg_s", "min_s", "max_s"},
        )
        self.assertAlmostEqual(5.5, latency["p50_s"])
        self.assertAlmostEqual(9.1, latency["p90_s"])
        self.assertNotIn("p95_s", latency)
        self.assertNotIn("p99_s", latency)
        self.assertEqual("passed", summary["status"])
        self.assertEqual(6.0, summary["throughput"]["videos_per_minute"])

    def test_any_failed_request_invalidates_the_case(self):
        results = [successful_result(0, 1.0)]
        failed = successful_result(1, 2.0)
        failed.ok = False
        failed.error = "failed"
        results.append(failed)

        summary = benchmark.summarize(
            results=results,
            total_wall_s=2.0,
            first_success_s=1.0,
            run_config={},
            unit_name="images",
        )

        self.assertEqual("failed", summary["status"])
        self.assertEqual(1, summary["requests_failed"])
        self.assertEqual(0.5, summary["success_rate"])

    def test_s2v_relative_audio_paths_resolve_from_dataset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audio_dir = root / "audios"
            audio_dir.mkdir()
            (audio_dir / "01.mp3").write_bytes(b"audio")
            data_path = root / "s2v.jsonl"
            data_path.write_text(
                json.dumps(
                    {
                        "prompt": "speaker",
                        "seed": 42,
                        "audio_path": "audios/01.mp3",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            payloads = benchmark.load_requests(
                data_path,
                limit=1,
                repeat=1,
                allowed_fields={"prompt", "seed", "audio_path"},
            )

        self.assertEqual(
            str((audio_dir / "01.mp3").resolve()),
            payloads[0]["audio_path"],
        )

    def test_run_benchmark_writes_result_requests_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "measure"

            def fake_runner(phase, index, _payload, output_path, task_id):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"\x89PNG\r\n\x1a\nmock")
                result = successful_result(index, float(index + 1))
                result.phase = phase
                result.task_id = task_id
                result.artifact_path = str(output_path)
                result.artifact_bytes = output_path.stat().st_size
                result.artifact_sha256 = benchmark.sha256_file(output_path)
                return result

            summary = benchmark.run_benchmark(
                task_kind="t2i",
                payloads=[
                    {"prompt": "one", "seed": 1},
                    {"prompt": "two", "seed": 2},
                ],
                case_id="case",
                run_id="suite",
                phase="measure",
                concurrency=1,
                output_dir=output_dir,
                request_runner=fake_runner,
                run_config={"case_id": "case"},
            )

            saved_summary = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
            request_lines = (output_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual("passed", summary["status"])
        self.assertEqual("passed", saved_summary["status"])
        self.assertEqual(2, len(request_lines))
        first_request = json.loads(request_lines[0])
        self.assertEqual(
            {"prompt": "prompt-0", "seed": 0},
            first_request["request_payload"],
        )

    def test_async_result_keeps_task_timing_and_validates_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "output.mp4"
            output_path.write_bytes(b"mock-mp4")
            responses = [
                (
                    200,
                    {
                        "task_id": "task-1",
                        "save_result_path": str(output_path),
                    },
                ),
                (200, {"task_id": "task-1", "status": "processing"}),
                (
                    200,
                    {
                        "task_id": "task-1",
                        "status": "completed",
                        "start_time": "2026-01-01T00:00:01",
                        "end_time": "2026-01-01T00:00:03.500000",
                        "save_result_path": "output.mp4",
                    },
                ),
            ]
            with (
                mock.patch.object(benchmark, "http_json", side_effect=responses),
                mock.patch.object(benchmark.time, "sleep"),
            ):
                result = benchmark.post_async_task(
                    phase="measure",
                    index=0,
                    payload={"prompt": "speaker", "seed": 42},
                    create_url="http://127.0.0.1/v1/tasks/video/",
                    base_url="http://127.0.0.1",
                    request_timeout_seconds=30.0,
                    poll_interval_seconds=0.5,
                    task_timeout_seconds=60.0,
                    output_path=output_path,
                    task_id="task-1",
                )

        self.assertTrue(result.ok)
        self.assertEqual(2, result.poll_count)
        self.assertEqual(2.5, result.server_processing_s)
        self.assertEqual("task-1", result.request_payload["task_id"])
        self.assertEqual(str(output_path), result.artifact_path)


if __name__ == "__main__":
    unittest.main()
