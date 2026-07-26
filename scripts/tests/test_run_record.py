from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUN_RECORD_PATH = REPOSITORY_ROOT / "scripts" / "run_record.py"
SPEC = importlib.util.spec_from_file_location("run_record_under_test", RUN_RECORD_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {RUN_RECORD_PATH}")
run_record = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_record)


class MP4PixelValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import av
        except ImportError as exc:
            raise unittest.SkipTest(f"PyAV is unavailable: {exc}") from exc
        cls.av = av

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="run-record-test-")
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_video(self, name: str, fill: str, frames: int = 4) -> Path:
        path = self.directory / name
        with self.av.open(str(path), mode="w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=8)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = "yuv420p"
            for frame_index in range(frames):
                frame = self.av.VideoFrame(64, 64, "rgb24")
                plane = frame.planes[0]
                rows = bytearray()
                for row_index in range(64):
                    if fill == "black":
                        row = bytes(64 * 3)
                    elif fill == "white":
                        row = bytes([255]) * (64 * 3)
                    else:
                        row = bytes(
                            value
                            for column_index in range(64)
                            for value in (
                                (column_index * 4 + frame_index) % 256,
                                (row_index * 4 + frame_index) % 256,
                                (column_index + row_index + frame_index) % 256,
                            )
                        )
                    rows.extend(row)
                    rows.extend(bytes(plane.line_size - len(row)))
                plane.update(bytes(rows))
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return path

    def test_unusable_ffmpeg_falls_back_to_full_pyav_decode(self) -> None:
        path = self.write_video("color.mp4", "color")
        ffmpeg_check = {
            "command": ["/usr/bin/ffmpeg", "-version"],
            "return_code": 127,
            "stdout": "",
            "stderr": "error while loading shared libraries: libfreetype.so.6",
            "error": "",
        }
        with (
            mock.patch.object(
                run_record.shutil,
                "which",
                side_effect=lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
            ),
            mock.patch.object(
                run_record,
                "run_command",
                return_value=ffmpeg_check,
            ),
        ):
            artifact = run_record.validate_artifact(
                path,
                "mp4",
                {"width": 64, "height": 64, "frames": 4},
            )

        self.assertTrue(artifact["valid"])
        details = artifact["validation"]["pixel_validation"]
        self.assertEqual(
            details["method"],
            "PyAV full decode at 64x64 RGB24",
        )
        self.assertEqual(details["decoded_frames"], 4)
        self.assertEqual(details["errors"], [])
        self.assertIn("libfreetype.so.6", details["fallback_reason"])

    def test_uniform_black_and_white_videos_are_rejected(self) -> None:
        for fill in ("black", "white"):
            with self.subTest(fill=fill):
                path = self.write_video(f"{fill}.mp4", fill)
                with mock.patch.object(
                    run_record.shutil,
                    "which",
                    return_value=None,
                ):
                    artifact = run_record.validate_artifact(
                        path,
                        "mp4",
                        {"width": 64, "height": 64, "frames": 4},
                    )

                self.assertFalse(artifact["valid"])
                details = artifact["validation"]["pixel_validation"]
                self.assertEqual(details["decoded_frames"], 4)
                self.assertEqual(
                    details["errors"],
                    [f"MP4 pixels are uniformly {fill}"],
                )

    def test_pixel_and_metadata_frame_counts_must_match(self) -> None:
        path = self.write_video("color.mp4", "color")
        pixel_validation = {
            **run_record.mp4_pixel_validation_record("test decoder"),
            "decoded_frames": 3,
            "channel_extrema": [[0, 255], [0, 255], [0, 255]],
        }
        with (
            mock.patch.object(
                run_record.shutil,
                "which",
                return_value=None,
            ),
            mock.patch.object(
                run_record,
                "inspect_mp4_rgb_extrema",
                return_value=pixel_validation,
            ),
        ):
            details = run_record.validate_mp4(path, 64, 64, 4)

        self.assertIn(
            "MP4 pixel decoder produced 3 frames, metadata decoder produced 4",
            details["errors"],
        )

    def test_working_ffmpeg_decode_error_is_not_masked_by_pyav(self) -> None:
        path = self.write_video("color.mp4", "color")
        ffmpeg_check = {
            "command": ["/usr/bin/ffmpeg", "-version"],
            "return_code": 0,
            "stdout": "ffmpeg version test",
            "stderr": "",
            "error": "",
        }
        failed_decode = subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=1,
            stdout=b"",
            stderr=b"corrupt media",
        )
        with (
            mock.patch.object(
                run_record.shutil,
                "which",
                return_value="/usr/bin/ffmpeg",
            ),
            mock.patch.object(
                run_record,
                "run_command",
                return_value=ffmpeg_check,
            ),
            mock.patch.object(
                run_record.subprocess,
                "run",
                return_value=failed_decode,
            ),
            mock.patch.object(
                run_record,
                "inspect_mp4_rgb_extrema_pyav",
            ) as pyav_fallback,
        ):
            details = run_record.inspect_mp4_rgb_extrema(path)

        self.assertEqual(details["errors"], ["corrupt media"])
        pyav_fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
