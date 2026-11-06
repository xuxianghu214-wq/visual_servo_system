"""按主机单调时钟保存并插值机器人位姿。"""

from __future__ import annotations

import bisect
import threading
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from visual_servo_system.geometry.pose_transforms import validate_transform
from visual_servo_system.models.data_types import RobotStateSample


def _slerp_quaternion_xyzw(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """最短路径四元数 SLERP；``fraction`` 可超出 0～1 以支持短时外推。"""

    q0 = np.array(first, dtype=np.float64, copy=True)
    q1 = np.array(second, dtype=np.float64, copy=True)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + fraction * (q1 - q0)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    result = (
        np.sin((1.0 - fraction) * angle) / np.sin(angle) * q0
        + np.sin(fraction * angle) / np.sin(angle) * q1
    )
    return result / np.linalg.norm(result)


class RobotPoseHistory:
    """固定容量的 ``RobotStateSample`` 时间序列。

    区间内平移和速度做线性插值，旋转用四元数 SLERP。区间外仅允许
    ``max_extrapolation_s`` 内的短时外推，超过窗口立即抛出 ``LookupError``，防止视觉
    线程错误地把陈旧机器人位姿与新图像配对。
    """

    def __init__(self, capacity: int, max_extrapolation_s: float) -> None:
        if capacity < 2:
            raise ValueError("capacity must be at least 2")
        if not np.isfinite(max_extrapolation_s) or max_extrapolation_s < 0.0:
            raise ValueError("max_extrapolation_s must be finite and non-negative")
        self._samples: deque[RobotStateSample] = deque(maxlen=capacity)
        self._max_extrapolation_s = float(max_extrapolation_s)
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)

    def add(self, sample: RobotStateSample) -> None:
        """追加一个状态；时间戳必须严格递增。"""

        with self._lock:
            if self._samples and sample.host_timestamp_s <= self._samples[-1].host_timestamp_s:
                raise ValueError("robot state timestamps must be strictly increasing")
            self._samples.append(sample)

    def clear(self) -> None:
        """原子清空旧 TCP 授权代次的全部位姿。"""

        with self._lock:
            self._samples.clear()

    def snapshot(self) -> tuple[RobotStateSample, ...]:
        """原子地复制当前有界历史；返回的元组不受后续追加/淘汰影响。"""

        with self._lock:
            return tuple(self._samples)

    def interpolate(self, host_timestamp_s: float) -> RobotStateSample:
        """返回查询时刻的状态，允许在配置窗口内进行线性/SLERP 外推。"""

        query = float(host_timestamp_s)
        if not np.isfinite(query) or query < 0.0:
            raise ValueError("host_timestamp_s must be finite and non-negative")
        with self._lock:
            return self._interpolate_locked(query)

    def _interpolate_locked(self, query: float) -> RobotStateSample:
        """在 ``self._lock`` 已持有时完成一次一致插值。"""

        samples = self.snapshot()  # RLock 允许同一线程重入，并取得稳定快照。
        if not samples:
            raise LookupError("pose history is empty")

        oldest = samples[0].host_timestamp_s
        newest = samples[-1].host_timestamp_s
        if query < oldest - self._max_extrapolation_s or query > newest + self._max_extrapolation_s:
            raise LookupError("query is outside the configured extrapolation window")
        if len(samples) == 1:
            source = samples[0]
            extrapolation_s = abs(query - source.host_timestamp_s)
            return RobotStateSample(
                host_timestamp_s=query,
                T_base_flange=source.T_base_flange,
                tcp_speed=source.tcp_speed,
                joint_positions_rad=source.joint_positions_rad,
                sequence_number=source.sequence_number,
                robot_mode=source.robot_mode,
                valid=source.valid and source.timestamp_reliable,
                quality=source.quality,
                receive_timestamp_s=max(query, source.receive_timestamp_s),
                timestamp_source=f"interpolated_{source.timestamp_source}",
                timestamp_reliable=source.timestamp_reliable,
                timestamp_uncertainty_s=(
                    source.timestamp_uncertainty_s + extrapolation_s
                ),
            )

        timestamps = [sample.host_timestamp_s for sample in samples]
        right_index = bisect.bisect_right(timestamps, query)
        if right_index == 0:
            first, second = samples[0], samples[1]
        elif right_index >= len(samples):
            first, second = samples[-2], samples[-1]
        else:
            first, second = samples[right_index - 1], samples[right_index]

        interval = second.host_timestamp_s - first.host_timestamp_s
        fraction = (query - first.host_timestamp_s) / interval
        first_transform = first.T_base_flange
        second_transform = second.T_base_flange
        translation = first_transform[:3, 3] + fraction * (
            second_transform[:3, 3] - first_transform[:3, 3]
        )
        first_quaternion = Rotation.from_matrix(first_transform[:3, :3]).as_quat()
        second_quaternion = Rotation.from_matrix(second_transform[:3, :3]).as_quat()
        quaternion = _slerp_quaternion_xyzw(first_quaternion, second_quaternion, fraction)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        transform[:3, 3] = translation
        transform = validate_transform(transform)

        def interpolate_array(name: str) -> np.ndarray:
            first_value = getattr(first, name)
            second_value = getattr(second, name)
            return first_value + fraction * (second_value - first_value)

        nearest = first if fraction <= 0.5 else second
        quality = float(np.clip(first.quality + fraction * (second.quality - first.quality), 0, 1))
        endpoints_reliable = first.timestamp_reliable and second.timestamp_reliable
        if endpoints_reliable and first.timestamp_source == second.timestamp_source:
            timestamp_source = f"interpolated_{first.timestamp_source}"
        else:
            timestamp_source = "interpolated_mixed"
        # 区间内用半个 RTDE 样本间隔作为配准界；外推时再加上
        # 超出最近端点的时间，防止不确定度被低估。
        extrapolation_s = max(first.host_timestamp_s - query, query - second.host_timestamp_s, 0.0)
        uncertainty_s = (
            max(first.timestamp_uncertainty_s, second.timestamp_uncertainty_s)
            + 0.5 * interval
            + extrapolation_s
        )
        return RobotStateSample(
            host_timestamp_s=query,
            T_base_flange=transform,
            tcp_speed=interpolate_array("tcp_speed"),
            joint_positions_rad=interpolate_array("joint_positions_rad"),
            sequence_number=nearest.sequence_number,
            robot_mode=nearest.robot_mode,
            valid=first.valid and second.valid and endpoints_reliable,
            quality=quality,
            receive_timestamp_s=max(
                query, first.receive_timestamp_s, second.receive_timestamp_s
            ),
            timestamp_source=timestamp_source,
            timestamp_reliable=endpoints_reliable,
            timestamp_uncertainty_s=uncertainty_s,
        )
