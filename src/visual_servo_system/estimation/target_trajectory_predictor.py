"""把 EKF 状态变成 MPC 使用的一秒短时轨迹。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visual_servo_system.estimation.so3_error_state import (
    exponential_velocity_discretization,
    propagate_pose_twist_covariance,
)
from visual_servo_system.geometry.pose_transforms import so3_exp
from visual_servo_system.models.data_types import PredictedTrajectory, TargetStateEstimate


@dataclass(frozen=True, slots=True)
class TrajectoryPredictionConfig:
    horizon_s: float = 1.0
    step_s: float = 0.05
    velocity_decay_time_constant_s: float = 0.35
    dropout_timeout_s: float = 1.0
    max_3sigma_position_m: float = 0.030
    max_3sigma_rotation_rad: float = np.deg2rad(5.0)
    linear_acceleration_std_mps2: float = 0.01
    angular_acceleration_std_rad_s2: float = np.deg2rad(2.0)

    def __post_init__(self) -> None:
        values = (
            self.horizon_s,
            self.step_s,
            self.velocity_decay_time_constant_s,
            self.dropout_timeout_s,
            self.max_3sigma_position_m,
            self.max_3sigma_rotation_rad,
            self.linear_acceleration_std_mps2,
            self.angular_acceleration_std_rad_s2,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("all trajectory prediction settings must be finite and positive")
        count = self.horizon_s / self.step_s
        if not np.isclose(count, round(count), atol=1e-12, rtol=0.0):
            raise ValueError("horizon_s must be an integer multiple of step_s")


class TargetTrajectoryPredictor:
    """常速度预测；只有已经遮挡时才对速度作指数衰减。"""

    def __init__(self, config: TrajectoryPredictionConfig | None = None) -> None:
        self.config = config or TrajectoryPredictionConfig()

    def _propagate_covariance(
        self,
        covariance: np.ndarray,
        duration_s: float,
        angular_velocity_body_rad_s: np.ndarray,
        initial_measurement_age_s: float,
    ) -> np.ndarray:
        """分段传播到指定时刻，并让遮挡角速度按累计年龄衰减。"""

        if duration_s <= 0.0:
            return np.array(covariance, copy=True)
        result = np.array(covariance, copy=True)
        elapsed = 0.0
        tau = self.config.velocity_decay_time_constant_s
        while elapsed < duration_s - 1e-15:
            step = min(self.config.step_s, duration_s - elapsed)
            start_age = initial_measurement_age_s + elapsed
            angular_velocity = angular_velocity_body_rad_s * np.exp(-start_age / tau)
            result = propagate_pose_twist_covariance(
                result,
                step,
                angular_velocity,
                self.config.linear_acceleration_std_mps2,
                self.config.angular_acceleration_std_rad_s2,
                velocity_decay_time_constant_s=tau,
            )
            elapsed += step
        return result

    @staticmethod
    def _three_sigma_radius(covariance_block: np.ndarray) -> float:
        """相关三维高斯量的最坏方向 3σ，而不是只看三个对角元。"""

        symmetric = (covariance_block + covariance_block.T) * 0.5
        largest_variance = float(np.max(np.linalg.eigvalsh(symmetric)))
        return 3.0 * float(np.sqrt(max(largest_variance, 0.0)))

    def predict(
        self, estimate: TargetStateEstimate, host_timestamp_s: float
    ) -> PredictedTrajectory:
        timestamp = float(host_timestamp_s)
        if not np.isfinite(timestamp) or timestamp < estimate.host_timestamp_s:
            raise ValueError("prediction timestamp must be finite and not precede the estimate")
        propagation_age_s = timestamp - estimate.host_timestamp_s
        effective_measurement_age_s = estimate.measurement_age_s + propagation_age_s
        covariance_now = self._propagate_covariance(
            estimate.covariance,
            propagation_age_s,
            estimate.angular_velocity_rad_s,
            estimate.measurement_age_s,
        )
        tau = self.config.velocity_decay_time_constant_s
        if propagation_age_s > 0.0:
            velocity_at_estimate_factor = float(
                np.exp(-estimate.measurement_age_s / tau)
            )
            propagation_factor = float(np.exp(-propagation_age_s / tau))
            propagation_integral = (
                velocity_at_estimate_factor * tau * (1.0 - propagation_factor)
            )
        else:
            propagation_integral = 0.0
        position_now = (
            estimate.T_base_target[:3, 3]
            + estimate.linear_velocity_mps * propagation_integral
        )
        rotation_now = estimate.T_base_target[:3, :3] @ so3_exp(
            estimate.angular_velocity_rad_s * propagation_integral
        )
        total_decay_factor = float(np.exp(-effective_measurement_age_s / tau))
        linear_velocity_now = estimate.linear_velocity_mps * total_decay_factor
        angular_velocity_now = estimate.angular_velocity_rad_s * total_decay_factor
        count = int(round(self.config.horizon_s / self.config.step_s))
        times = np.arange(1, count + 1, dtype=np.float64) * self.config.step_s
        transforms = np.repeat(estimate.T_base_target[None, :, :], count, axis=0)
        twists = np.zeros((count, 6), dtype=np.float64)

        reason = "ok"
        valid = bool(estimate.valid)
        if not estimate.valid:
            reason = "target estimate is invalid"
        elif effective_measurement_age_s >= self.config.dropout_timeout_s:
            valid = False
            reason = "dropout timeout reached"

        position_sigma = self._three_sigma_radius(covariance_now[0:3, 0:3])
        rotation_sigma = self._three_sigma_radius(covariance_now[3:6, 3:6])
        if position_sigma > self.config.max_3sigma_position_m:
            valid = False
            reason = "position uncertainty exceeds 3-sigma limit"
        elif rotation_sigma > self.config.max_3sigma_rotation_rad:
            valid = False
            reason = "rotation uncertainty exceeds 3-sigma limit"

        occluded = effective_measurement_age_s > 0.0
        future_covariance = covariance_now
        predicted_position = position_now.copy()
        predicted_rotation = rotation_now.copy()
        predicted_linear_velocity = linear_velocity_now.copy()
        predicted_angular_velocity = angular_velocity_now.copy()
        for index, future_s in enumerate(times):
            decay_tau = tau if occluded else None
            speed_factor, integrated_time, _ = exponential_velocity_discretization(
                self.config.step_s, decay_tau
            )
            step_start_angular_velocity = predicted_angular_velocity.copy()
            predicted_position += predicted_linear_velocity * integrated_time
            predicted_rotation = predicted_rotation @ so3_exp(
                step_start_angular_velocity * integrated_time
            )
            predicted_linear_velocity *= speed_factor
            predicted_angular_velocity *= speed_factor
            transforms[index, :3, 3] = predicted_position
            transforms[index, :3, :3] = predicted_rotation
            twists[index, :3] = predicted_linear_velocity
            # EKF 的角速度在目标局部坐标；轨迹接口统一输出 base/spatial 角速度。
            twists[index, 3:] = (
                predicted_rotation @ predicted_angular_velocity
            )
            future_covariance = propagate_pose_twist_covariance(
                future_covariance,
                self.config.step_s,
                step_start_angular_velocity,
                self.config.linear_acceleration_std_mps2,
                self.config.angular_acceleration_std_rad_s2,
                velocity_decay_time_constant_s=decay_tau,
            )
            # 遮挡时只评估尚未到达 1 秒硬超时的那段未来；可见目标会在下一帧刷新。
            if occluded and future_s <= self.config.dropout_timeout_s - effective_measurement_age_s:
                future_position_sigma = self._three_sigma_radius(
                    future_covariance[0:3, 0:3]
                )
                future_rotation_sigma = self._three_sigma_radius(
                    future_covariance[3:6, 3:6]
                )
                if valid and future_position_sigma > self.config.max_3sigma_position_m:
                    valid = False
                    reason = "predicted position uncertainty exceeds 3-sigma limit"
                elif valid and future_rotation_sigma > self.config.max_3sigma_rotation_rad:
                    valid = False
                    reason = "predicted rotation uncertainty exceeds 3-sigma limit"

        age_quality = max(
            0.0, 1.0 - effective_measurement_age_s / self.config.dropout_timeout_s
        )
        quality = float(np.clip(estimate.quality * age_quality if valid else 0.0, 0.0, 1.0))
        return PredictedTrajectory(
            host_timestamp_s=timestamp,
            sample_times_s=times,
            transforms=transforms,
            twists=twists,
            valid=valid,
            quality=quality,
            reason=reason,
        )
