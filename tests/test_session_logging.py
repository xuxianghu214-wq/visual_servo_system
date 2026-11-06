from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from visual_servo_system.logging.optional_video_recorder import OptionalVideoRecorder
from visual_servo_system.logging.session_csv_logger import LoggerCloseError, SessionCsvLogger


def test_session_logger_creates_unique_manifest_and_all_streams(tmp_path):
    first = SessionCsvLogger(
        tmp_path,
        camera_model="BFS-U3-13Y3C",
        camera_serial="16276663",
        config_snapshot={"marker_id": 0},
        wall_clock=lambda: "20261022_120000",
    )
    second = SessionCsvLogger(
        tmp_path,
        camera_model="BFS-U3-13Y3C",
        camera_serial="16276663",
        config_snapshot={},
        wall_clock=lambda: "20261022_120000",
    )
    assert first.session_directory != second.session_directory
    manifest = json.loads((first.session_directory / "session_manifest.json").read_text("utf-8"))
    assert manifest["camera_model"] == "BFS-U3-13Y3C"
    assert manifest["camera_serial"] == "16276663"
    assert manifest["config_snapshot"]["marker_id"] == 0
    for stream in ("vision", "ekf", "prediction", "mpc", "robot_state", "safety_events", "timing"):
        assert (first.session_directory / f"{stream}.csv").exists()
    with (first.session_directory / "vision.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        header = next(csv.reader(stream))
    assert "receive_timestamp_s" in header
    assert "timestamp_source" in header
    assert "timestamp_reliable" in header
    assert "timestamp_uncertainty_s" in header
    first.close()
    second.close()


def test_logger_flushes_thread_safe_rows(tmp_path):
    logger = SessionCsvLogger(tmp_path, config_snapshot={})
    logger.write("timing", {"host_timestamp_s": 1.0, "stage": "vision", "duration_ms": 2.5})
    with (logger.session_directory / "timing.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["stage"] == "vision"
    assert float(rows[0]["duration_ms"]) == pytest.approx(2.5)
    with pytest.raises(KeyError):
        logger.write("not_a_stream", {"host_timestamp_s": 1.0})
    logger.close()


class _FakeWriter:
    def __init__(self, *_args):
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(np.array(frame, copy=True))

    def release(self):
        self.released = True


def test_optional_video_is_off_by_default_and_checks_size(tmp_path):
    created: list[_FakeWriter] = []

    def factory(*args):
        writer = _FakeWriter(*args)
        created.append(writer)
        return writer

    recorder = OptionalVideoRecorder(
        tmp_path / "preview.mp4",
        frame_size=(8, 6),
        fps=20.0,
        enabled=False,
        writer_factory=factory,
    )
    recorder.write(np.zeros((6, 8, 3), dtype=np.uint8))
    assert created == []
    recorder.set_enabled(True)
    recorder.write(np.zeros((6, 8, 3), dtype=np.uint8))
    assert len(created[0].frames) == 1
    with pytest.raises(ValueError, match="resolution"):
        recorder.write(np.zeros((5, 8, 3), dtype=np.uint8))
    recorder.close()
    assert created[0].released


def test_video_reenable_creates_numbered_segments_without_overwrite(tmp_path):
    created_paths = []
    writers = []

    def factory(path, *_args):
        created_paths.append(path)
        writer = _FakeWriter()
        writers.append(writer)
        return writer

    recorder = OptionalVideoRecorder(
        tmp_path / "preview.mp4",
        frame_size=(8, 6),
        fps=20.0,
        writer_factory=factory,
    )
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    recorder.set_enabled(True)
    recorder.write(frame)
    recorder.set_enabled(False)
    recorder.set_enabled(True)
    recorder.write(frame)
    recorder.close()

    assert [str(path).replace("\\", "/").split("/")[-1] for path in created_paths] == [
        "preview_segment_001.mp4",
        "preview_segment_002.mp4",
    ]
    assert all(writer.released for writer in writers)


def test_csv_close_is_best_effort_and_retries_only_failed_stream(tmp_path):
    logger = SessionCsvLogger(tmp_path, config_snapshot={})
    original = logger._files["vision"]

    class FailCloseOnce:
        def __init__(self, stream):
            self.stream = stream
            self.close_calls = 0

        def flush(self):
            self.stream.flush()

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise OSError("vision close blocked")
            self.stream.close()

    flaky = FailCloseOnce(original)
    logger._files["vision"] = flaky

    with pytest.raises(LoggerCloseError, match="vision close blocked"):
        logger.close()
    assert not logger.closed
    # 其他成功流已经完成，重试时不得重复 flush/close 它们。
    assert all(
        stream.closed
        for name, stream in logger._files.items()
        if name != "vision"
    )

    logger.close()
    assert logger.closed
    assert flaky.close_calls == 2
