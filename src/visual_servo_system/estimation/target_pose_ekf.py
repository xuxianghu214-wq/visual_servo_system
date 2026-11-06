"""目标六维位姿与六维速度的误差状态 EKF。

名义姿态始终保存为合法旋转矩阵；协方差里的三维姿态量是 SO(3) 切空间小量，
因此不会出现欧拉角在 ±180° 跳变的问题。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visual_servo_system.estimation.so3_error_state import (
    propagate_pose_twist_covariance,
    so3_right_jacobian,
)
from visual_servo_system.geometry.pose_transforms import so3_exp, so3_log, validate_transform
from visual_servo_system.models.data_types import TargetStateEstimate


class EkfTimestampGapError(ValueError):
    """相邻图像间隔超过 EKF 单步模型允许值。"""


@dataclass(frozen=True, slots=True)
class TargetPoseEkfConfig:
    position_measurement_std_m: float = 0.005
    orientation_measurement_std_rad: float = np.deg2rad(1.5)
    linear_acceleration_std_mps2: float = 0.20
    angular_acceleration_std_rad_s2: float = np.deg2rad(20.0)
    mahalanobis_gate_99_6d: float = 16.812
    max_dt_s: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.position_measurement_std_m,
            self.orientation_measurement_std_rad,
            self.linear_acceleration_std_mps2,
            self.angular_acceleration_std_rad_s2,
            self.mahalanobis_gate_99_6d,
            self.max_dt_s,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("all EKF configuration values must be finite and positive")


class TargetPoseEkf:
    """常速度目标模型的 12 维误差状态 EKF。"""

    def __init__(self, config: TargetPoseEkfConfig | None = None) -> None:
        self.config = config or TargetPoseEkfConfig()
        self._position = np.zeros(3)
        self._rotation = np.eye(3)
        self._linear_velocity = np.zeros(3)
        self._angular_velocity = np.zeros(3)
        self._covariance = np.eye(12)
        self._timestamp_s: float | None = None
        self._last_measurement_timestamp_s: float | None = None
        self.last_update_accepted = False
        self.last_rejection_reason = "not initialized"

    @property
    def initialized(self) -> bool:
        return self._timestamp_s is not None

    def reset(self) -> None:
        """丢弃上一目标/授权代次的全部名义状态和协方差。"""

        self._position.fill(0.0)
        self._rotation = np.eye(3)
        self._linear_velocity.fill(0.0)
        self._angular_velocity.fill(0.0)
        self._covariance = np.eye(12)
        self._timestamp_s = None
        self._last_measurement_timestamp_s = None
        self.last_update_accepted = False
        self.last_rejection_reason = "not initialized"

    def _initial_covariance(self) -> np.ndarray:
        cfg = self.config
        diagonal = np.concatenate(
            (
                np.full(3, cfg.position_measurement_std_m**2),
                np.full(3, cfg.orientation_measurement_std_rad**2),
                np.full(3, 0.05**2),
                np.full(3, np.deg2rad(10.0) ** 2),
            )
        )
        return np.diag(diagonal)

    @staticmethod
    def _timestamp(value: float) -> float:
        result = float(value)
        if not np.isfinite(result) or result < 0.0:
            raise ValueError("host timestamp must be finite and non-negative")
        return result

    def _predict_nominal(self, dt: float) -> None:
        self._position = self._position + self._linear_velocity * dt
        rotation_increment = self._angular_velocity * dt
        self._rotation = self._rotation @ so3_exp(rotation_increment)
        self._covariance = propagate_pose_twist_covariance(
            self._covariance,
            dt,
            self._angular_velocity,
            self.config.linear_acceleration_std_mps2,
            self.config.angular_acceleration_std_rad_s2,
        )
        self._covariance = self._make_psd(self._covariance)

    @staticmethod
    def _make_psd(matrix: np.ndarray) -> np.ndarray:
        symmetric = (matrix + matrix.T) * 0.5
        values, vectors = np.linalg.eigh(symmetric)
        return (vectors * np.maximum(values, 1e-15)) @ vectors.T

    def _advance_to(self, timestamp_s: float) -> None:
        if self._timestamp_s is None:
            raise RuntimeError("EKF is not initialized")
        dt = timestamp_s - self._timestamp_s
        if dt <= 0.0:
            raise ValueError("EKF timestamps must be strictly increasing")
        if dt > self.config.max_dt_s:
            raise EkfTimestampGapError(
                f"EKF timestamp gap {dt:.6f}s exceeds max_dt_s "
                f"{self.config.max_dt_s:.6f}s"
            )
        self._predict_nominal(dt)
        self._timestamp_s = timestamp_s

    def update(self, T_base_target: np.ndarray, host_timestamp_s: float) -> TargetStateEstimate:
        """融合一次位姿观测；异常值保留预测状态并增加测量年龄。"""

        transform = validate_transform(T_base_target)
        timestamp = self._timestamp(host_timestamp_s)
        if not self.initialized:
            self._position = transform[:3, 3].copy()
            self._rotation = transform[:3, :3].copy()
            self._linear_velocity.fill(0.0)
            self._angular_velocity.fill(0.0)
            self._covariance = self._initial_covariance()
            self._timestamp_s = timestamp
            self._last_measurement_timestamp_s = timestamp
            self.last_update_accepted = True
            self.last_rejection_reason = "accepted initial measurement"
            return self._estimate(valid=True, quality=1.0)

        self._advance_to(timestamp)
        residual = np.concatenate(
            (
                transform[:3, 3] - self._position,
                so3_log(self._rotation.T @ transform[:3, :3]),
            )
        )
        observation = np.zeros((6, 12))
        observation[:, :6] = np.eye(6)
        measurement_covariance = np.diag(
            np.concatenate(
                (
                    np.full(3, self.config.position_measurement_std_m**2),
                    np.full(3, self.config.orientation_measurement_std_rad**2),
                )
            )
        )
        innovation_covariance = (
            observation @ self._covariance @ observation.T + measurement_covariance
        )
        mahalanobis = float(residual @ np.linalg.solve(innovation_covariance, residual))
        if not np.isfinite(mahalanobis) or mahalanobis > self.config.mahalanobis_gate_99_6d:
            self.last_update_accepted = False
            self.last_rejection_reason = (
                f"mahalanobis gate rejected measurement ({mahalanobis:.3f} > "
                f"{self.config.mahalanobis_gate_99_6d:.3f})"
            )
            return self._estimate(valid=True, quality=0.5)

        gain = np.linalg.solve(
            innovation_covariance, observation @ self._covariance
        ).T
        correction = gain @ residual
        self._position += correction[0:3]
        self._rotation = self._rotation @ so3_exp(correction[3:6])
        self._linear_velocity += correction[6:9]
        self._angular_velocity += correction[9:12]
        identity = np.eye(12)
        joseph_left = identity - gain @ observation
        self._covariance = (
            joseph_left @ self._covariance @ joseph_left.T
            + gain @ measurement_covariance @ gain.T
        )
        # 名义姿态吸收 correction 后，剩余右误差需重新映射到新的切空间。
        reset_jacobian = np.eye(12)
        reset_jacobian[3:6, 3:6] = so3_right_jacobian(correction[3:6])
        self._covariance = reset_jacobian @ self._covariance @ reset_jacobian.T
        self._covariance = self._make_psd(self._covariance)
        self._last_measurement_timestamp_s = timestamp
        self.last_update_accepted = True
        self.last_rejection_reason = "accepted"
        quality = float(np.clip(1.0 - mahalanobis / self.config.mahalanobis_gate_99_6d, 0, 1))
        return self._estimate(valid=True, quality=quality)

    def predict_to(self, host_timestamp_s: float) -> TargetStateEstimate:
        """无新观测时，把滤波状态预测到指定单调时间。"""

        timestamp = self._timestamp(host_timestamp_s)
        self._advance_to(timestamp)
        self.last_update_accepted = False
        self.last_rejection_reason = "prediction without measurement"
        return self._estimate(valid=True, quality=0.7)

    def _estimate(self, *, valid: bool, quality: float) -> TargetStateEstimate:
        transform = np.eye(4)
        transform[:3, :3] = self._rotation
        transform[:3, 3] = self._position
        assert self._timestamp_s is not None
        assert self._last_measurement_timestamp_s is not None
        return TargetStateEstimate(
            host_timestamp_s=float(self._timestamp_s),
            T_base_target=transform,
            linear_velocity_mps=self._linear_velocity,
            angular_velocity_rad_s=self._angular_velocity,
            covariance=self._covariance,
            valid=valid,
            quality=float(quality),
            measurement_age_s=float(self._timestamp_s - self._last_measurement_timestamp_s),
        )
