"""把设备时钟稳定映射到主机单调时钟。

设备时间和主机 ``time.monotonic`` 通常有不同的零点与轻微频率偏差。映射器使用
成对斜率中位数估计频率，再用最小时延样本估计偏移；USB/网络排队只会让接收时间
变晚，因此最小时延边界比普通均值更接近真实采样时刻。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ClockMappingEstimate:
    """一次设备时间映射结果，所有时间量单位均为秒。"""

    host_timestamp_s: float
    reliable: bool
    uncertainty_s: float
    residual_rms_s: float
    sample_count: int
    reset_count: int


class DeviceClockMapper:
    """在线仿射时钟映射，设备时钟回退时自动丢弃旧代次。"""

    def __init__(
        self,
        *,
        min_samples: int = 4,
        max_samples: int = 64,
        max_uncertainty_s: float = 0.005,
        min_clock_rate: float = 0.98,
        max_clock_rate: float = 1.02,
    ) -> None:
        if type(min_samples) is not int or min_samples < 2:
            raise ValueError("min_samples must be an integer >= 2")
        if type(max_samples) is not int or max_samples < min_samples:
            raise ValueError("max_samples must be >= min_samples")
        values = (max_uncertainty_s, min_clock_rate, max_clock_rate)
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("clock mapper limits must be finite and positive")
        if min_clock_rate >= max_clock_rate:
            raise ValueError("min_clock_rate must be smaller than max_clock_rate")
        self.min_samples = min_samples
        self.max_uncertainty_s = float(max_uncertainty_s)
        self.min_clock_rate = float(min_clock_rate)
        self.max_clock_rate = float(max_clock_rate)
        self._samples: deque[tuple[float, float]] = deque(maxlen=max_samples)
        self._last_device_s: float | None = None
        self._reset_count = 0

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def reset_count(self) -> int:
        return self._reset_count

    def reset(self) -> None:
        self._samples.clear()
        self._last_device_s = None
        self._reset_count += 1

    @staticmethod
    def _finite_non_negative(value: float, name: str) -> float:
        result = float(value)
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result

    def observe(self, device_timestamp_s: float, receive_host_s: float) -> ClockMappingEstimate:
        """加入一个设备采样时刻及其主机接收时刻，并返回当前映射。"""

        device = self._finite_non_negative(device_timestamp_s, "device_timestamp_s")
        receive = self._finite_non_negative(receive_host_s, "receive_host_s")
        if self._last_device_s is not None and device <= self._last_device_s:
            self.reset()
        self._last_device_s = device
        self._samples.append((device, receive))

        if len(self._samples) == 1:
            return ClockMappingEstimate(
                receive,
                False,
                float("inf"),
                0.0,
                1,
                self._reset_count,
            )

        samples = np.asarray(self._samples, dtype=np.float64)
        device_values = samples[:, 0]
        host_values = samples[:, 1]
        slopes: list[float] = []
        for left in range(len(samples) - 1):
            device_delta = device_values[left + 1 :] - device_values[left]
            host_delta = host_values[left + 1 :] - host_values[left]
            valid = device_delta > 0.0
            slopes.extend((host_delta[valid] / device_delta[valid]).tolist())
        rate = float(np.median(slopes))
        rate_plausible = self.min_clock_rate <= rate <= self.max_clock_rate
        offset = float(np.min(host_values - rate * device_values))
        mapped_values = rate * device_values + offset
        delays = np.maximum(host_values - mapped_values, 0.0)
        residual_rms = float(np.sqrt(np.mean(delays**2)))
        uncertainty = float(np.max(delays))
        mapped = min(float(rate * device + offset), receive)
        reliable = bool(
            len(samples) >= self.min_samples
            and rate_plausible
            and uncertainty <= self.max_uncertainty_s
        )
        return ClockMappingEstimate(
            mapped,
            reliable,
            uncertainty,
            residual_rms,
            len(samples),
            self._reset_count,
        )
