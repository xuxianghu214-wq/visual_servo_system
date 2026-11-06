"""EKF 与轨迹预测器共用的 SO(3) 右误差传播工具。"""

from __future__ import annotations

import numpy as np

from visual_servo_system.geometry.pose_transforms import so3_exp


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_right_jacobian(rotation_vector: np.ndarray) -> np.ndarray:
    """SO(3) 右雅可比；小角度时使用稳定级数。"""

    vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    matrix = skew(vector)
    if angle < 1e-7:
        return np.eye(3) - 0.5 * matrix + matrix @ matrix / 6.0
    return (
        np.eye(3)
        - (1.0 - np.cos(angle)) / angle**2 * matrix
        + (angle - np.sin(angle)) / angle**3 * (matrix @ matrix)
    )


def exponential_velocity_discretization(
    dt_s: float, velocity_decay_time_constant_s: float | None = None
) -> tuple[float, float, float]:
    """返回速度衰减模型的 ``(a, b, c)`` 离散系数。

    连续模型为 ``v_dot = -v/tau + u``。一个采样周期内：

    * ``v_next = a * v + b * u``；
    * ``p_next = p + b * v + c * u``。

    ``tau=None`` 表示 EKF 使用的常速度模型，此时 ``a=1``、``b=dt``、
    ``c=dt²/2``。使用 ``expm1`` 可避免很小采样周期下的相消误差。
    """

    dt = float(dt_s)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt_s must be finite and non-negative")
    if velocity_decay_time_constant_s is None:
        return 1.0, dt, 0.5 * dt**2
    tau = float(velocity_decay_time_constant_s)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("velocity decay time constant must be finite and positive")
    one_minus_decay = -float(np.expm1(-dt / tau))
    decay = 1.0 - one_minus_decay
    integrated_velocity_time = tau * one_minus_decay
    acceleration_position_gain = tau * (dt - integrated_velocity_time)
    return decay, integrated_velocity_time, acceleration_position_gain


def pose_twist_error_transition(
    dt_s: float,
    angular_velocity_body_rad_s: np.ndarray,
    *,
    velocity_decay_time_constant_s: float | None = None,
) -> np.ndarray:
    """返回 ``[p,theta,v,w]`` 右误差模型的 12×12 转移矩阵。

    遮挡衰减时，名义速度和速度误差必须使用相同的 ``a``，位姿积分及
    位姿—速度误差耦合必须使用相同的 ``b``，否则均值与协方差会描述两个
    不同的系统。
    """

    dt = float(dt_s)
    angular_velocity = np.asarray(angular_velocity_body_rad_s, dtype=np.float64)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt_s must be finite and non-negative")
    if angular_velocity.shape != (3,) or not np.all(np.isfinite(angular_velocity)):
        raise ValueError("angular velocity must be a finite three-vector")
    decay, integrated_time, _ = exponential_velocity_discretization(
        dt, velocity_decay_time_constant_s
    )
    increment = angular_velocity * integrated_time
    transition = np.eye(12)
    transition[0:3, 6:9] = np.eye(3) * integrated_time
    transition[6:9, 6:9] = np.eye(3) * decay
    transition[3:6, 3:6] = so3_exp(-increment)
    transition[3:6, 9:12] = so3_right_jacobian(increment) * integrated_time
    transition[9:12, 9:12] = np.eye(3) * decay
    return transition


def pose_twist_process_noise(
    dt_s: float,
    linear_acceleration_std_mps2: float,
    angular_acceleration_std_rad_s2: float,
    *,
    angular_velocity_body_rad_s: np.ndarray | None = None,
    velocity_decay_time_constant_s: float | None = None,
) -> np.ndarray:
    """将一个周期内的加速度扰动映射到位姿和速度误差。

    这里使用零阶保持加速度：线性输入矩阵为 ``[c I, b I]``；角向输入
    矩阵为 ``[Jr(phi)c, b I]``。用输入矩阵外积构造 Q，可明确保证 PSD，
    并且其 ``a/b/c`` 与均值及误差转移完全一致。
    """

    dt = float(dt_s)
    standards = (linear_acceleration_std_mps2, angular_acceleration_std_rad_s2)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt_s must be finite and non-negative")
    if not all(np.isfinite(value) and value >= 0.0 for value in standards):
        raise ValueError("acceleration noise values must be finite and non-negative")
    angular_velocity = (
        np.zeros(3)
        if angular_velocity_body_rad_s is None
        else np.asarray(angular_velocity_body_rad_s, dtype=np.float64)
    )
    if angular_velocity.shape != (3,) or not np.all(np.isfinite(angular_velocity)):
        raise ValueError("angular velocity must be a finite three-vector")
    _, integrated_time, acceleration_position_gain = (
        exponential_velocity_discretization(dt, velocity_decay_time_constant_s)
    )
    increment = angular_velocity * integrated_time

    input_matrix = np.zeros((12, 6))
    input_matrix[0:3, 0:3] = np.eye(3) * acceleration_position_gain
    input_matrix[6:9, 0:3] = np.eye(3) * integrated_time
    input_matrix[3:6, 3:6] = (
        so3_right_jacobian(increment) * acceleration_position_gain
    )
    input_matrix[9:12, 3:6] = np.eye(3) * integrated_time
    acceleration_covariance = np.diag(
        np.concatenate(
            (
                np.full(3, standards[0] ** 2),
                np.full(3, standards[1] ** 2),
            )
        )
    )
    noise = input_matrix @ acceleration_covariance @ input_matrix.T
    return (noise + noise.T) * 0.5


def propagate_pose_twist_covariance(
    covariance: np.ndarray,
    dt_s: float,
    angular_velocity_body_rad_s: np.ndarray,
    linear_acceleration_std_mps2: float,
    angular_acceleration_std_rad_s2: float,
    *,
    velocity_decay_time_constant_s: float | None = None,
) -> np.ndarray:
    """用共用 SO(3) 模型传播一次协方差并恢复数值对称。"""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (12, 12) or not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be a finite 12x12 matrix")
    transition = pose_twist_error_transition(
        dt_s,
        angular_velocity_body_rad_s,
        velocity_decay_time_constant_s=velocity_decay_time_constant_s,
    )
    process = pose_twist_process_noise(
        dt_s,
        linear_acceleration_std_mps2,
        angular_acceleration_std_rad_s2,
        angular_velocity_body_rad_s=angular_velocity_body_rad_s,
        velocity_decay_time_constant_s=velocity_decay_time_constant_s,
    )
    propagated = transition @ matrix @ transition.T + process
    return (propagated + propagated.T) * 0.5
