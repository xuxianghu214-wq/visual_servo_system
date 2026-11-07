"""UR 控制器时钟与网络异常的明确模拟序列。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RobotReplayConfig:
    frequency_hz: float = 125.0
    duration_s: float = 1.0
    controller_start_s: float = 100.0
    host_offset_s: float = 10.0
    fixed_delay_s: float = 0.001
    jitter_s: float = 0.0
    drift_ppm: float = 0.0
    drop_indices: tuple[int, ...] = ()
    duplicate_indices: tuple[int, ...] = ()
    reorder_pairs: tuple[tuple[int, int], ...] = ()
    delay_indices: tuple[int, ...] = ()
    injected_delay_s: float = 0.0
    reset_index: int | None = None
    reset_to_s: float = 0.0
    random_seed: int = 20261107

    def __post_init__(self) -> None:
        values = (
            self.frequency_hz,
            self.duration_s,
            self.controller_start_s,
            self.host_offset_s,
            self.fixed_delay_s,
            self.jitter_s,
            self.drift_ppm,
            self.injected_delay_s,
            self.reset_to_s,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("robot replay values must be finite")
        if self.frequency_hz <= 0.0 or self.duration_s <= 0.0:
            raise ValueError("frequency_hz and duration_s must be positive")
        if min(
            self.controller_start_s,
            self.host_offset_s,
            self.fixed_delay_s,
            self.jitter_s,
            self.injected_delay_s,
            self.reset_to_s,
        ) < 0.0:
            raise ValueError("robot replay timestamps and delays must be non-negative")


@dataclass(frozen=True, slots=True)
class SimulatedRtdeSample:
    source_index: int
    controller_timestamp_s: float
    receive_host_s: float
    simulated: bool = True
    provenance: str = "synthetic_rtde_for_logic_test_only"


def generate_robot_replay(
    config: RobotReplayConfig | None = None,
) -> Iterator[SimulatedRtdeSample]:
    """生成不包含任何真实 UR 性能结论的时钟逻辑测试序列。"""

    config = RobotReplayConfig() if config is None else config
    count = int(round(config.frequency_hz * config.duration_s))
    period_s = 1.0 / config.frequency_hz
    rate = 1.0 + config.drift_ppm * 1e-6
    drop_indices = set(config.drop_indices)
    duplicate_indices = set(config.duplicate_indices)
    delayed_indices = set(config.delay_indices)
    rng = np.random.default_rng(config.random_seed)

    arrivals: list[tuple[float, int, float]] = []
    for index in range(count):
        if index in drop_indices:
            continue
        if config.reset_index is not None and index >= config.reset_index:
            controller_s = config.reset_to_s + (index - config.reset_index) * period_s * rate
        else:
            controller_s = config.controller_start_s + index * period_s * rate
        jitter = float(rng.uniform(-config.jitter_s, config.jitter_s))
        arrival_s = (
            config.host_offset_s
            + index * period_s
            + config.fixed_delay_s
            + jitter
            + (config.injected_delay_s if index in delayed_indices else 0.0)
        )
        arrivals.append((arrival_s, index, controller_s))
        if index in duplicate_indices:
            arrivals.append((arrival_s + period_s * 0.25, index, controller_s))

    for left_identity, right_identity in config.reorder_pairs:
        left = next(
            (position for position, item in enumerate(arrivals) if item[1] == left_identity),
            None,
        )
        right = next(
            (position for position, item in enumerate(arrivals) if item[1] == right_identity),
            None,
        )
        if left is not None and right is not None:
            left_arrival, left_index, left_clock = arrivals[left]
            right_arrival, right_index, right_clock = arrivals[right]
            arrivals[left] = (right_arrival, left_index, left_clock)
            arrivals[right] = (left_arrival, right_index, right_clock)

    for arrival_s, index, controller_s in sorted(arrivals, key=lambda item: item[0]):
        yield SimulatedRtdeSample(
            source_index=index,
            controller_timestamp_s=float(controller_s),
            receive_host_s=float(arrival_s),
        )
