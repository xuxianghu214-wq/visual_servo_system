"""把设备时钟稳定映射到主机单调时钟。

每次输入都先分类为新样本、重复、延迟、乱序或真实时钟重置。
重复和乱序样本不进入拟合，也不会清空已建立的时钟映射。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class ClockSampleStatus(StrEnum):
    """设备时钟样本的完整分类。"""

    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    DELAYED = "delayed"
    OUT_OF_ORDER = "out_of_order"
    RESET = "reset"


@dataclass(frozen=True, slots=True)
class ClockMappingEstimate:
    """一次设备时间映射结果，所有时间量单位均为秒。"""

    host_timestamp_s: float
    reliable: bool
    uncertainty_s: float
    residual_rms_s: float
    sample_count: int
    reset_count: int
    status: ClockSampleStatus = ClockSampleStatus.ACCEPTED
    transport_delay_s: float = 0.0


class DeviceClockMapper:
    """在线仿射时钟映射，仅对足够大的时钟回退重建时基。"""

    def __init__(
        self,
        *,
        min_samples: int = 4,
        max_samples: int = 64,
        max_uncertainty_s: float = 0.005,
        min_clock_rate: float = 0.98,
        max_clock_rate: float = 1.02,
        reset_backjump_s: float = 1.0,
    ) -> None:
        if type(min_samples) is not int or min_samples < 2:
            raise ValueError("min_samples must be an integer >= 2")
        if type(max_samples) is not int or max_samples < min_samples:
            raise ValueError("max_samples must be >= min_samples")
        values = (
            max_uncertainty_s,
            min_clock_rate,
            max_clock_rate,
            reset_backjump_s,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("clock mapper limits must be finite and positive")
        if min_clock_rate >= max_clock_rate:
            raise ValueError("min_clock_rate must be smaller than max_clock_rate")
        self.min_samples = min_samples
        self.max_uncertainty_s = float(max_uncertainty_s)
        self.min_clock_rate = float(min_clock_rate)
        self.max_clock_rate = float(max_clock_rate)
        self.reset_backjump_s = float(reset_backjump_s)
        self._samples: deque[tuple[float, float]] = deque(maxlen=max_samples)
        self._last_device_s: float | None = None
        self._reset_count = 0
        self._duplicate_count = 0
        self._delayed_count = 0
        self._out_of_order_count = 0

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def reset_count(self) -> int:
        return self._reset_count

    @property
    def duplicate_count(self) -> int:
        return self._duplicate_count

    @property
    def delayed_count(self) -> int:
        return self._delayed_count

    @property
    def out_of_order_count(self) -> int:
        return self._out_of_order_count

    def reset(self) -> None:
        """显式开始新时钟代次，并计入真实重置计数。"""

        self._samples.clear()
        self._last_device_s = None
        self._reset_count += 1

    @staticmethod
    def _finite_non_negative(value: float, name: str) -> float:
        result = float(value)
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result

    def _fit(self) -> tuple[float, float, float, float, bool]:
        samples = np.asarray(self._samples, dtype=np.float64)
        device_values = samples[:, 0]
        host_values = samples[:, 1]
        device_delta = device_values[np.newaxis, :] - device_values[:, np.newaxis]
        host_delta = host_values[np.newaxis, :] - host_values[:, np.newaxis]
        upper = np.triu(np.ones(device_delta.shape, dtype=bool), k=1)
        valid = upper & (device_delta > 0.0)
        slopes = host_delta[valid] / device_delta[valid]
        rate = float(np.median(slopes))
        rate_plausible = self.min_clock_rate <= rate <= self.max_clock_rate
        # 单向接收只能使用最小排队边界估计偏移；无法观测绝对网络时延。
        offset = float(np.min(host_values - rate * device_values))
        mapped_values = rate * device_values + offset
        delays = np.maximum(host_values - mapped_values, 0.0)
        residual_rms = float(np.sqrt(np.mean(delays**2)))
        uncertainty = float(np.max(delays))
        return rate, offset, residual_rms, uncertainty, rate_plausible

    def _estimate_without_append(
        self,
        device: float,
        receive: float,
        status: ClockSampleStatus,
    ) -> ClockMappingEstimate:
        if len(self._samples) < 2:
            host = self._samples[-1][1] if self._samples else receive
            return ClockMappingEstimate(
                min(host, receive),
                False,
                float("inf"),
                0.0,
                len(self._samples),
                self._reset_count,
                status,
                max(0.0, receive - host),
            )
        rate, offset, residual_rms, uncertainty, rate_plausible = self._fit()
        predicted = rate * device + offset
        mapped = min(float(predicted), receive)
        transport_delay = max(0.0, receive - mapped)
        reliable = bool(
            status is ClockSampleStatus.ACCEPTED
            and len(self._samples) >= self.min_samples
            and rate_plausible
            and uncertainty <= self.max_uncertainty_s
        )
        return ClockMappingEstimate(
            mapped,
            reliable,
            uncertainty,
            residual_rms,
            len(self._samples),
            self._reset_count,
            status,
            transport_delay,
        )

    def observe(self, device_timestamp_s: float, receive_host_s: float) -> ClockMappingEstimate:
        """分类样本，只把新且未过度延迟的样本加入拟合。"""

        device = self._finite_non_negative(device_timestamp_s, "device_timestamp_s")
        receive = self._finite_non_negative(receive_host_s, "receive_host_s")
        status = ClockSampleStatus.ACCEPTED

        if self._last_device_s is not None:
            if device == self._last_device_s:
                self._duplicate_count += 1
                return self._estimate_without_append(
                    device, receive, ClockSampleStatus.DUPLICATE
                )
            if device < self._last_device_s:
                backjump = self._last_device_s - device
                if backjump < self.reset_backjump_s:
                    self._out_of_order_count += 1
                    return self._estimate_without_append(
                        device, receive, ClockSampleStatus.OUT_OF_ORDER
                    )
                self.reset()
                status = ClockSampleStatus.RESET

        if status is ClockSampleStatus.ACCEPTED and len(self._samples) >= self.min_samples:
            preliminary = self._estimate_without_append(
                device, receive, ClockSampleStatus.ACCEPTED
            )
            if preliminary.transport_delay_s > self.max_uncertainty_s:
                self._delayed_count += 1
                return ClockMappingEstimate(
                    preliminary.host_timestamp_s,
                    False,
                    preliminary.uncertainty_s,
                    preliminary.residual_rms_s,
                    preliminary.sample_count,
                    preliminary.reset_count,
                    ClockSampleStatus.DELAYED,
                    preliminary.transport_delay_s,
                )

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
                status,
                0.0,
            )
        estimate = self._estimate_without_append(device, receive, status)
        if status is ClockSampleStatus.RESET:
            return ClockMappingEstimate(
                estimate.host_timestamp_s,
                False,
                estimate.uncertainty_s,
                estimate.residual_rms_s,
                estimate.sample_count,
                estimate.reset_count,
                status,
                estimate.transport_delay_s,
            )
        return estimate
