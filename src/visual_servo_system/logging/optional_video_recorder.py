"""默认关闭的试验视频记录器。"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class OptionalVideoRecorder:
    def __init__(
        self,
        output_path: str | Path,
        *,
        frame_size: tuple[int, int],
        fps: float,
        enabled: bool = False,
        writer_factory: Callable[..., Any] = cv2.VideoWriter,
    ) -> None:
        width, height = frame_size
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ValueError("frame_size must be positive integer (width, height)")
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be finite and positive")
        self.output_path = Path(output_path)
        self.frame_size = (width, height)
        self.fps = float(fps)
        self._enabled = bool(enabled)
        self._factory = writer_factory
        self._writer: Any | None = None
        self._lock = threading.RLock()
        self._next_segment_index = 1
        self._active_output_path: Path | None = None

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        with self._lock:
            self._enabled = enabled
            if not enabled:
                self._release_locked()

    def _ensure_open_locked(self) -> None:
        if self._writer is not None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            segment_path = self.output_path.with_name(
                f"{self.output_path.stem}_segment_{self._next_segment_index:03d}"
                f"{self.output_path.suffix}"
            )
            self._next_segment_index += 1
            if not segment_path.exists():
                break
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = self._factory(
            str(segment_path), fourcc, self.fps, self.frame_size, True
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"cannot open video writer: {segment_path}")
        self._writer = writer
        self._active_output_path = segment_path

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            if not self._enabled:
                return
            image = np.asarray(frame)
            width, height = self.frame_size
            if image.shape[:2] != (height, width):
                raise ValueError(
                    f"video frame resolution {image.shape[:2][::-1]} does not match "
                    f"configured resolution {self.frame_size}"
                )
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError("video frame must be uint8 Mono8 or BGR8")
            self._ensure_open_locked()
            self._writer.write(np.ascontiguousarray(image))

    def _release_locked(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._active_output_path = None

    def close(self) -> None:
        with self._lock:
            self._release_locked()
