import numpy as np
import pytest

from visual_servo_system.geometry.pose_transforms import (
    compose_transforms,
    invert_transform,
    is_rotation_matrix,
    so3_exp,
    so3_log,
    transform_to_ur_pose,
    ur_pose_to_transform,
    validate_transform,
)


def test_transform_composition_and_inverse_follow_parent_child_convention() -> None:
    T_base_flange = ur_pose_to_transform([0.4, -0.1, 0.5, 0.1, -0.2, 0.3])
    T_flange_camera = ur_pose_to_transform([0.03, 0.0, 0.08, 0.0, 0.0, 0.2])

    T_base_camera = compose_transforms(T_base_flange, T_flange_camera)

    np.testing.assert_allclose(
        compose_transforms(invert_transform(T_base_flange), T_base_camera),
        T_flange_camera,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        compose_transforms(T_base_camera, invert_transform(T_base_camera)),
        np.eye(4),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "pose",
    [
        np.zeros(6),
        np.array([0.4, -0.2, 0.8, 0.2, -0.3, 0.1]),
        np.array([-0.1, 0.2, 0.3, np.pi - 1e-6, 0.0, 0.0]),
    ],
)
def test_ur_axis_angle_pose_round_trip_preserves_transform(pose: np.ndarray) -> None:
    transform = ur_pose_to_transform(pose)
    round_trip = ur_pose_to_transform(transform_to_ur_pose(transform))

    np.testing.assert_allclose(round_trip, transform, atol=1e-9)


@pytest.mark.parametrize(
    "rotation_vector",
    [np.zeros(3), np.array([1e-10, -2e-10, 3e-10]), np.array([0.3, -0.4, 0.2])],
)
def test_so3_log_exp_round_trip(rotation_vector: np.ndarray) -> None:
    rotation = so3_exp(rotation_vector)

    assert is_rotation_matrix(rotation)
    np.testing.assert_allclose(so3_exp(so3_log(rotation)), rotation, atol=1e-10)


def test_validate_transform_rejects_bad_bottom_row_and_non_orthonormal_rotation() -> None:
    bad_bottom = np.eye(4)
    bad_bottom[3, 0] = 0.1
    with pytest.raises(ValueError, match="bottom row"):
        validate_transform(bad_bottom)

    bad_rotation = np.eye(4)
    bad_rotation[0, 0] = 1.2
    with pytest.raises(ValueError, match="orthonormal"):
        validate_transform(bad_rotation)

