"""一次试验一个目录的线程安全 CSV 日志。

文件名对应数据的产生环节，表头中直接写明 SI 单位。日志器只写入调用者
明确提供的值；缺失值保持为空，不会伪造测量。
"""

from __future__ import annotations

import csv
import json
import platform
import threading
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

_COMMON = ["host_timestamp_s", "valid", "quality", "reason"]
STREAM_SCHEMAS: dict[str, list[str]] = {
    "vision": _COMMON
    + [
        "frame_sequence",
        "marker_id",
        "position_x_m",
        "position_y_m",
        "position_z_m",
        "reprojection_error_px",
        "pixel_area_px2",
        "receive_timestamp_s",
        "timestamp_source",
        "timestamp_reliable",
        "timestamp_uncertainty_s",
    ],
    "ekf": _COMMON
    + [
        "position_x_m",
        "position_y_m",
        "position_z_m",
        "linear_speed_mps",
        "angular_speed_rad_s",
        "measurement_age_s",
    ],
    "prediction": _COMMON + ["horizon_s", "sample_count"],
    "mpc": _COMMON
    + ["solver_status", "solve_time_s", "linear_speed_mps", "angular_speed_rad_s"],
    "robot_state": _COMMON
    + [
        "sequence_number",
        "robot_mode",
        "x_m",
        "y_m",
        "z_m",
        "receive_timestamp_s",
        "timestamp_source",
        "timestamp_reliable",
        "timestamp_uncertainty_s",
    ],
    "safety_events": _COMMON + ["state", "allow_motion", "generation"],
    "timing": ["host_timestamp_s", "stage", "duration_ms", "latency_ms", "reason"],
}


class LoggerCloseError(RuntimeError):
    """一个或多个 CSV 流未能关闭；可再次调用 ``close`` 重试。"""


class SessionCsvLogger:
    """持续 flush 的多流 CSV 记录器。"""

    def __init__(
        self,
        base_directory: str | Path,
        *,
        camera_model: str = "BFS-U3-13Y3C",
        camera_serial: str = "16276663",
        config_snapshot: dict[str, Any],
        wall_clock: Callable[[], str] | None = None,
        software_version: str = "0.1.0",
    ) -> None:
        timestamp = (wall_clock or (lambda: datetime.now().strftime("%Y%m%d_%H%M%S")))()
        root = Path(base_directory)
        root.mkdir(parents=True, exist_ok=True)
        # 随机后缀避免一秒内多次开始试验覆盖旧数据。
        self.session_directory = root / f"{timestamp}_{uuid.uuid4().hex[:8]}"
        self.session_directory.mkdir(parents=False, exist_ok=False)
        manifest = {
            "created_local": timestamp,
            "camera_model": camera_model,
            "camera_serial": camera_serial,
            "software_version": software_version,
            "python_version": platform.python_version(),
            "config_snapshot": config_snapshot,
        }
        (self.session_directory / "session_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._lock = threading.RLock()
        self._files: dict[str, Any] = {}
        self._writers: dict[str, csv.DictWriter] = {}
        self._closed = False
        self._closed_streams: set[str] = set()
        for name, fields in STREAM_SCHEMAS.items():
            stream = (self.session_directory / f"{name}.csv").open(
                "w", encoding="utf-8-sig", newline=""
            )
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            stream.flush()
            self._files[name] = stream
            self._writers[name] = writer

    def write(self, stream_name: str, values: dict[str, Any]) -> None:
        """向指定数据流写一行；每行立即 flush，断电时尽量保留已到达数据。"""

        with self._lock:
            if self._closed:
                raise RuntimeError("session logger is closed")
            if stream_name not in self._writers:
                raise KeyError(f"unknown log stream: {stream_name}")
            if stream_name in self._closed_streams:
                raise RuntimeError(f"log stream is already closed: {stream_name}")
            self._writers[stream_name].writerow(values)
            self._files[stream_name].flush()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            failures: list[str] = []
            for name, stream in self._files.items():
                if name in self._closed_streams:
                    continue
                try:
                    stream.flush()
                except Exception as error:
                    failures.append(f"{name} flush: {error}")
                    continue
                try:
                    stream.close()
                except Exception as error:
                    failures.append(f"{name} close: {error}")
                else:
                    self._closed_streams.add(name)
            self._closed = len(self._closed_streams) == len(self._files)
            if failures:
                raise LoggerCloseError("；".join(failures))

    def __enter__(self) -> SessionCsvLogger:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
