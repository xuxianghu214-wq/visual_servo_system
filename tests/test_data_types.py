from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from visual_servo_system.models.data_types import (
    FramePacket,
    MarkerPoseObservation,
    MpcResult,
    PredictedTrajectory,
    RobotStateSample,
    SafetyDecision,
    TargetStateEstimate,
)


def test_frame_packet_copies_image_and_is_immutable() -> None:
    source = np.zeros((4, 6), dtype=np.uint8)
    packet = FramePacket(
        host_timestamp_s=1.25,
        sequence_number=7,
        image=source,
        valid=True,
        quality=0.9,
    )
    source[0, 0] = 255

    assert packet.image[0, 0] == 0
    assert not packet.image.flags.writeable
    with pytest.raises(FrozenInstanceError):
        packet.valid = False  # type: ignore[misc]


def test_frame_packet_exposes_capture_receive_time_and_reliability() -> None:
    packet = FramePacket(
        1.0,
        2,
        np.zeros((2, 2), np.uint8),
        True,
        1.0,
        receive_timestamp_s=1.01,
        timestamp_source="camera_device",
        timestamp_reliable=True,
        timestamp_uncertainty_s=0.001,
    )
    assert packet.host_timestamp_s == pytest.approx(1.0)
    assert packet.receive_timestamp_s == pytest.approx(1.01)
    assert packet.timestamp_source == "camera_device"
    assert packet.timestamp_reliable is True
    with pytest.raises(ValueError, match="receive_timestamp_s"):
        FramePacket(
            2.0,
            2,
            np.zeros((2, 2), np.uint8),
            True,
            1.0,
            receive_timestamp_s=1.0,
        )


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: MarkerPoseObservation(
                host_timestamp_s=0.0,
                marker_id=0,
                T_camera_marker=np.eye(3),
                reprojection_error_px=0.2,
                decision_margin=None,
                pixel_area_px2=100.0,
                valid=True,
                quality=1.0,
            ),
            "T_camera_marker",
        ),
        (
            lambda: RobotStateSample(
                host_timestamp_s=0.0,
                T_base_flange=np.eye(4),
                tcp_speed=np.zeros(5),
            ),
            "tcp_speed",
        ),
        (
            lambda: TargetStateEstimate(
                host_timestamp_s=0.0,
                T_base_target=np.eye(4),
                linear_velocity_mps=np.zeros(3),
                angular_velocity_rad_s=np.zeros(3),
                covariance=np.eye(11),
                valid=True,
                quality=1.0,
            ),
            "covariance",
        ),
        (
            lambda: PredictedTrajectory(
                host_timestamp_s=0.0,
                sample_times_s=np.array([0.0, 0.05]),
                transforms=np.zeros((2, 3, 3)),
                twists=np.zeros((2, 6)),
                valid=True,
                quality=1.0,
            ),
            "transforms",
        ),
        (
            lambda: MpcResult(
                host_timestamp_s=0.0,
                commanded_twist=np.zeros(5),
                solve_time_s=0.001,
                solver_status="solved",
                valid=True,
                quality=1.0,
            ),
            "commanded_twist",
        ),
        (
            lambda: SafetyDecision(
                host_timestamp_s=0.0,
                allow_motion=True,
                safe_twist=np.zeros(7),
                reason="ok",
                valid=True,
                quality=1.0,
            ),
            "safe_twist",
        ),
    ],
)
def test_public_data_types_reject_wrong_array_shapes(factory, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_public_data_types_reject_invalid_timestamp_and_quality() -> None:
    with pytest.raises(ValueError, match="host_timestamp_s"):
        FramePacket(-1.0, 0, np.zeros((2, 2), dtype=np.uint8), True, 1.0)
    with pytest.raises(ValueError, match="quality"):
        FramePacket(0.0, 0, np.zeros((2, 2), dtype=np.uint8), True, 1.1)


def test_arrays_are_read_only_in_all_seven_public_data_types() -> None:
    transforms = np.repeat(np.eye(4, dtype=float)[None, :, :], 2, axis=0)
    trajectory = PredictedTrajectory(
        host_timestamp_s=1.0,
        sample_times_s=np.array([0.0, 0.05]),
        transforms=transforms,
        twists=np.zeros((2, 6)),
        valid=True,
        quality=1.0,
    )
    instances = [
        FramePacket(1.0, 1, np.zeros((2, 2), dtype=np.uint8), True, 1.0),
        MarkerPoseObservation(1.0, 0, np.eye(4), 0.1, None, 100.0, True, 1.0),
        RobotStateSample(1.0, np.eye(4)),
        TargetStateEstimate(1.0, np.eye(4), np.zeros(3), np.zeros(3), np.eye(12), True, 1.0),
        trajectory,
        MpcResult(1.0, np.zeros(6), 0.001, "solved", True, 1.0, trajectory),
        SafetyDecision(1.0, True, np.zeros(6), "ok", True, 1.0),
    ]

    assert len(instances) == 7
    for instance in instances:
        arrays = [
            getattr(instance, item.name)
            for item in fields(instance)
            if isinstance(getattr(instance, item.name), np.ndarray)
        ]
        assert arrays, f"{type(instance).__name__} must expose at least one array"
        for array in arrays:
            assert not array.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                array.reshape(-1)[0] = 123.0
            with pytest.raises(ValueError):
                array.setflags(write=True)


def _valid_public_type_factories() -> list:
    transforms = np.repeat(np.eye(4, dtype=float)[None, :, :], 2, axis=0)
    return [
        lambda valid: FramePacket(1.0, 1, np.zeros((2, 2), dtype=np.uint8), valid, 1.0),
        lambda valid: MarkerPoseObservation(
            1.0, 0, np.eye(4), 0.1, None, 100.0, valid, 1.0
        ),
        lambda valid: RobotStateSample(1.0, np.eye(4), valid=valid),
        lambda valid: TargetStateEstimate(
            1.0, np.eye(4), np.zeros(3), np.zeros(3), np.eye(12), valid, 1.0
        ),
        lambda valid: PredictedTrajectory(
            1.0, np.array([0.0, 0.05]), transforms, np.zeros((2, 6)), valid, 1.0
        ),
        lambda valid: MpcResult(1.0, np.zeros(6), 0.001, "solved", valid, 1.0),
        lambda valid: SafetyDecision(1.0, False, np.zeros(6), "stopped", valid, 1.0),
    ]


@pytest.mark.parametrize("factory", _valid_public_type_factories())
@pytest.mark.parametrize("invalid_bool", [0, 1, "true", np.bool_(True)])
def test_all_public_data_types_require_real_bool_valid_fields(factory, invalid_bool) -> None:
    with pytest.raises(TypeError, match="valid"):
        factory(invalid_bool)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FramePacket(1.0, True, np.zeros((2, 2), dtype=np.uint8), True, 1.0),
        lambda: FramePacket(1.0, 1.5, np.zeros((2, 2), dtype=np.uint8), True, 1.0),
        lambda: MarkerPoseObservation(1.0, True, np.eye(4), 0.1, None, 100.0, True, 1.0),
        lambda: RobotStateSample(1.0, np.eye(4), sequence_number=False),
    ],
)
def test_identifiers_and_sequences_require_integer_non_bool_values(factory) -> None:
    with pytest.raises(TypeError, match="sequence_number|marker_id"):
        factory()


@pytest.mark.parametrize("invalid_timestamp", [True, 1, "1.0", np.nan, np.inf])
def test_host_timestamp_requires_finite_float(invalid_timestamp) -> None:
    expected_error = TypeError if isinstance(invalid_timestamp, (bool, int, str)) else ValueError
    with pytest.raises(expected_error, match="host_timestamp_s"):
        FramePacket(invalid_timestamp, 0, np.zeros((2, 2), dtype=np.uint8), True, 1.0)


@pytest.mark.parametrize("invalid_quality", [True, 1, "1.0", np.nan, np.inf])
def test_quality_requires_finite_float(invalid_quality) -> None:
    expected_error = TypeError if isinstance(invalid_quality, (bool, int, str)) else ValueError
    with pytest.raises(expected_error, match="quality"):
        FramePacket(1.0, 0, np.zeros((2, 2), dtype=np.uint8), True, invalid_quality)


def test_target_covariance_must_be_positive_semidefinite() -> None:
    covariance = np.eye(12)
    covariance[0, 0] = -0.01

    with pytest.raises(ValueError, match="positive semidefinite"):
        TargetStateEstimate(
            1.0, np.eye(4), np.zeros(3), np.zeros(3), covariance, True, 1.0
        )


def test_data_type_transform_validation_matches_geometry_strict_tolerance() -> None:
    almost_rotation = np.eye(4)
    almost_rotation[0, 0] += 5e-7

    with pytest.raises(ValueError, match="T_camera_marker.*orthonormal"):
        MarkerPoseObservation(1.0, 0, almost_rotation, 0.1, None, 100.0, True, 1.0)


def test_marker_observation_separates_optional_margin_from_pixel_area() -> None:
    observation = MarkerPoseObservation(
        1.0, 0, np.eye(4), 0.2, None, 1234.5, True, 0.9
    )
    assert observation.decision_margin is None
    assert observation.pixel_area_px2 == pytest.approx(1234.5)
    with pytest.raises(ValueError, match="pixel_area_px2"):
        MarkerPoseObservation(1.0, 0, np.eye(4), 0.2, None, -1.0, True, 0.9)


@pytest.mark.parametrize("allow_motion", [0, 1, "yes", np.bool_(True)])
def test_safety_decision_requires_real_bool_allow_motion(allow_motion) -> None:
    with pytest.raises(TypeError, match="allow_motion"):
        SafetyDecision(1.0, allow_motion, np.zeros(6), "test", True, 1.0)


@pytest.mark.parametrize(
    ("allow_motion", "valid"), [(False, True), (True, False), (False, False)]
)
def test_stopped_or_invalid_safety_decision_requires_zero_twist(
    allow_motion: bool, valid: bool
) -> None:
    with pytest.raises(ValueError, match="safe_twist must be exactly zero"):
        SafetyDecision(1.0, allow_motion, np.ones(6), "stop", valid, 1.0)

