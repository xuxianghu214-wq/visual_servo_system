import numpy as np
import pytest

from visual_servo_system.estimation.so3_error_state import (
    pose_twist_error_transition,
    propagate_pose_twist_covariance,
    so3_right_jacobian,
)
from visual_servo_system.geometry.pose_transforms import so3_exp


def test_decay_transition_matches_exponential_position_velocity_model():
    """衰减时位置积分系数必须是 b，而速度误差也必须同步乘 a。"""

    dt_s = 0.20
    tau_s = 0.35
    decay = np.exp(-dt_s / tau_s)
    integrated_time = tau_s * (1.0 - decay)
    covariance = np.eye(12) * 1e-12
    covariance[0, 0] = 4e-4
    covariance[6, 6] = 9e-4
    covariance[0, 6] = covariance[6, 0] = 5e-4

    propagated = propagate_pose_twist_covariance(
        covariance,
        dt_s,
        np.zeros(3),
        0.0,
        0.0,
        velocity_decay_time_constant_s=tau_s,
    )

    expected_position_variance = (
        covariance[0, 0]
        + 2.0 * integrated_time * covariance[0, 6]
        + integrated_time**2 * covariance[6, 6]
    )
    expected_cross_covariance = decay * (
        covariance[0, 6] + integrated_time * covariance[6, 6]
    )
    assert propagated[0, 0] == pytest.approx(expected_position_variance)
    assert propagated[0, 6] == pytest.approx(expected_cross_covariance)
    assert propagated[6, 6] == pytest.approx(decay**2 * covariance[6, 6])
    assert np.min(np.linalg.eigvalsh(propagated)) >= -1e-12


def test_decay_transition_matches_so3_orientation_angular_velocity_model():
    """姿态—角速度相关项必须使用同一个衰减积分和 SO(3) 雅可比。"""

    dt_s = 0.20
    tau_s = 0.35
    angular_velocity = np.array([0.2, -0.1, np.deg2rad(30.0)])
    decay = np.exp(-dt_s / tau_s)
    integrated_time = tau_s * (1.0 - decay)
    rotation_increment = angular_velocity * integrated_time
    expected_transition = np.eye(12)
    expected_transition[0:3, 6:9] = np.eye(3) * integrated_time
    expected_transition[6:9, 6:9] = np.eye(3) * decay
    expected_transition[3:6, 3:6] = so3_exp(-rotation_increment)
    expected_transition[3:6, 9:12] = (
        so3_right_jacobian(rotation_increment) * integrated_time
    )
    expected_transition[9:12, 9:12] = np.eye(3) * decay

    transition = pose_twist_error_transition(
        dt_s,
        angular_velocity,
        velocity_decay_time_constant_s=tau_s,
    )
    assert transition == pytest.approx(expected_transition)

    generator = np.arange(144, dtype=float).reshape(12, 12) / 5000.0
    covariance = generator @ generator.T + np.eye(12) * 1e-8
    propagated = propagate_pose_twist_covariance(
        covariance,
        dt_s,
        angular_velocity,
        0.01,
        np.deg2rad(2.0),
        velocity_decay_time_constant_s=tau_s,
    )
    assert np.min(np.linalg.eigvalsh(propagated)) >= -1e-12

