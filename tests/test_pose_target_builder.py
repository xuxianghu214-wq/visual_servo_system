import numpy as np
import pytest

from visual_servo_system.control.pose_target_builder import (
    build_desired_flange_pose,
    build_desired_flange_trajectory,
)
from visual_servo_system.geometry.pose_transforms import compose_transforms
from visual_servo_system.models.data_types import PredictedTrajectory


def translated(x, y, z):
    result = np.eye(4)
    result[:3, 3] = [x, y, z]
    return result


def test_desired_flange_pose_closes_the_coordinate_chain():
    target = translated(0.4, -0.1, 0.8)
    flange_camera = translated(0.02, 0.0, 0.1)
    camera_target_desired = translated(0.0, 0.0, 0.5)
    flange = build_desired_flange_pose(target, flange_camera, camera_target_desired)
    rebuilt = compose_transforms(flange, flange_camera, camera_target_desired)
    assert np.allclose(rebuilt, target)


def test_flange_trajectory_recomputes_twist_from_transformed_poses():
    markers = np.repeat(np.eye(4)[None, :, :], 3, axis=0)
    markers[:, 0, 3] = [0.0, 0.01, 0.02]
    source = PredictedTrajectory(
        1.0,
        np.array([0.05, 0.10, 0.15]),
        markers,
        np.zeros((3, 6)),
        True,
        1.0,
    )
    result = build_desired_flange_trajectory(source, np.eye(4), np.eye(4))
    assert result.twists[0, 0] == pytest.approx(0.2)
    assert result.twists[-1, 0] == pytest.approx(0.2)

