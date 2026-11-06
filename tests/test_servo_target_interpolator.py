import numpy as np
import pytest

from visual_servo_system.control.servo_target_interpolator import ServoTargetInterpolator
from visual_servo_system.geometry.pose_transforms import so3_exp, ur_pose_to_transform


def transform(x, angle):
    result = np.eye(4)
    result[0, 3] = x
    result[:3, :3] = so3_exp([0.0, 0.0, angle])
    return result


def test_interpolator_uses_linear_translation_and_slerp():
    interpolator = ServoTargetInterpolator(max_extrapolation_s=0.05)
    interpolator.push_target(1.0, transform(0.0, 0.0), 1)
    interpolator.push_target(1.05, transform(0.05, np.pi / 2), 2)
    sampled = ur_pose_to_transform(interpolator.sample_ur_pose(1.025))
    assert sampled[0, 3] == pytest.approx(0.025)
    assert sampled[0, 0] == pytest.approx(np.sqrt(0.5), abs=1e-6)


def test_interpolator_rejects_bad_order_and_long_extrapolation():
    interpolator = ServoTargetInterpolator(max_extrapolation_s=0.05)
    interpolator.push_target(1.0, np.eye(4), 1)
    with pytest.raises(ValueError, match="strictly"):
        interpolator.push_target(1.0, np.eye(4), 2)
    with pytest.raises(LookupError):
        interpolator.sample_ur_pose(1.051)

