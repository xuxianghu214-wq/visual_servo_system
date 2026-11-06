"""把 20 Hz 位姿目标平滑采样为 125 Hz ``servoL`` 目标。"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from visual_servo_system.geometry.pose_transforms import transform_to_ur_pose, validate_transform


def _slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    q0 = Rotation.from_matrix(first).as_quat()
    q1 = Rotation.from_matrix(second).as_quat()
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        quaternion = q0 + fraction * (q1 - q0)
        quaternion /= np.linalg.norm(quaternion)
    else:
        angle = float(np.arccos(dot))
        quaternion = (
            np.sin((1.0 - fraction) * angle) / np.sin(angle) * q0
            + np.sin(fraction * angle) / np.sin(angle) * q1
        )
    return Rotation.from_quat(quaternion).as_matrix()


class ServoTargetInterpolator:
    """保留最近两个 MPC 目标，严格限制时间和序号倒退。"""

    def __init__(self, *, max_extrapolation_s: float = 0.05) -> None:
        if not np.isfinite(max_extrapolation_s) or max_extrapolation_s < 0.0:
            raise ValueError("max_extrapolation_s must be finite and non-negative")
        self._max_extrapolation_s = float(max_extrapolation_s)
        self._targets: deque[tuple[float, np.ndarray, int]] = deque(maxlen=2)
        self._lock = threading.RLock()

    def reset(self) -> None:
        with self._lock:
            self._targets.clear()

    def push_target(
        self, host_timestamp_s: float, T_base_flange: np.ndarray, sequence_number: int
    ) -> None:
        timestamp = float(host_timestamp_s)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("target timestamp must be finite and non-negative")
        if type(sequence_number) is not int or sequence_number < 0:
            raise ValueError("sequence_number must be a non-negative integer")
        transform = validate_transform(T_base_flange)
        with self._lock:
            if self._targets:
                previous_time, _, previous_sequence = self._targets[-1]
                if timestamp <= previous_time or sequence_number <= previous_sequence:
                    raise ValueError(
                        "target timestamps and sequence numbers must be strictly increasing"
                    )
            self._targets.append((timestamp, transform, sequence_number))

    def sample_transform(self, host_timestamp_s: float) -> np.ndarray:
        query = float(host_timestamp_s)
        if not np.isfinite(query) or query < 0.0:
            raise ValueError("sample timestamp must be finite and non-negative")
        with self._lock:
            targets = tuple(self._targets)
        if not targets:
            raise LookupError("no servo target is available")
        if query < targets[0][0]:
            raise LookupError("cannot sample before the oldest target")
        if query > targets[-1][0] + self._max_extrapolation_s + 1e-12:
            raise LookupError("servo target extrapolation exceeds one MPC period")
        if len(targets) == 1:
            return np.array(targets[0][1], copy=True)

        first_time, first, _ = targets[0]
        second_time, second, _ = targets[1]
        fraction = (query - first_time) / (second_time - first_time)
        position = first[:3, 3] + fraction * (second[:3, 3] - first[:3, 3])
        result = np.eye(4)
        result[:3, :3] = _slerp(first[:3, :3], second[:3, :3], fraction)
        result[:3, 3] = position
        return validate_transform(result)

    def sample_ur_pose(self, host_timestamp_s: float) -> np.ndarray:
        return transform_to_ur_pose(self.sample_transform(host_timestamp_s))
