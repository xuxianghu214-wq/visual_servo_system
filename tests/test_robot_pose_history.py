import threading

import numpy as np
import pytest

from visual_servo_system.geometry.pose_transforms import ur_pose_to_transform
from visual_servo_system.geometry.robot_pose_history import RobotPoseHistory
from visual_servo_system.models.data_types import RobotStateSample


def _sample(
    timestamp: float,
    x: float,
    rz: float,
    *,
    reliable: bool = True,
    uncertainty_s: float = 0.001,
    source: str = "ur_controller",
) -> RobotStateSample:
    return RobotStateSample(
        host_timestamp_s=timestamp,
        T_base_flange=ur_pose_to_transform([x, 0.0, 0.0, 0.0, 0.0, rz]),
        tcp_speed=np.array([x, 0.0, 0.0, 0.0, 0.0, rz]),
        joint_positions_rad=np.arange(6, dtype=float) + x,
        valid=True,
        quality=0.8,
        receive_timestamp_s=timestamp + 0.001,
        timestamp_source=source,
        timestamp_reliable=reliable,
        timestamp_uncertainty_s=uncertainty_s,
    )


def test_history_interpolates_position_rotation_and_numeric_state() -> None:
    history = RobotPoseHistory(capacity=4, max_extrapolation_s=0.05)
    history.add(_sample(1.0, 0.0, 0.0))
    history.add(_sample(1.1, 0.2, np.pi / 2.0))

    middle = history.interpolate(1.05)

    np.testing.assert_allclose(middle.T_base_flange[:3, 3], [0.1, 0.0, 0.0], atol=1e-12)
    expected_rotation = ur_pose_to_transform([0, 0, 0, 0, 0, np.pi / 4])[:3, :3]
    np.testing.assert_allclose(middle.T_base_flange[:3, :3], expected_rotation, atol=1e-10)
    np.testing.assert_allclose(middle.tcp_speed, [0.1, 0, 0, 0, 0, np.pi / 4])
    np.testing.assert_allclose(middle.joint_positions_rad, np.arange(6) + 0.1)
    assert middle.host_timestamp_s == pytest.approx(1.05)
    assert middle.receive_timestamp_s == pytest.approx(1.101)
    assert middle.timestamp_source == "interpolated_ur_controller"
    assert middle.timestamp_reliable
    assert middle.timestamp_uncertainty_s > 0.001


def test_history_is_bounded_and_rejects_out_of_window_queries() -> None:
    history = RobotPoseHistory(capacity=2, max_extrapolation_s=0.02)
    for timestamp in (1.0, 1.1, 1.2):
        history.add(_sample(timestamp, timestamp, 0.0))

    assert len(history) == 2
    with pytest.raises(LookupError, match="outside"):
        history.interpolate(1.05)
    with pytest.raises(LookupError, match="outside"):
        history.interpolate(1.23)


def test_history_extrapolates_inside_configured_window() -> None:
    history = RobotPoseHistory(capacity=3, max_extrapolation_s=0.02)
    history.add(_sample(1.0, 0.0, 0.0))
    history.add(_sample(1.1, 0.1, 0.1))

    extrapolated = history.interpolate(1.11)

    assert extrapolated.T_base_flange[0, 3] == pytest.approx(0.11)
    np.testing.assert_allclose(
        extrapolated.T_base_flange[:3, :3],
        ur_pose_to_transform([0, 0, 0, 0, 0, 0.11])[:3, :3],
        atol=1e-10,
    )


def test_history_rejects_non_increasing_samples() -> None:
    history = RobotPoseHistory(capacity=2, max_extrapolation_s=0.0)
    history.add(_sample(1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="strictly increasing"):
        history.add(_sample(1.0, 0.1, 0.0))


def test_snapshot_is_stable_after_concurrent_add() -> None:
    history = RobotPoseHistory(capacity=2, max_extrapolation_s=0.0)
    history.add(_sample(1.0, 0.0, 0.0))
    history.add(_sample(1.1, 0.1, 0.1))
    captured = history.snapshot()

    writer = threading.Thread(target=lambda: history.add(_sample(1.2, 0.2, 0.2)))
    writer.start()
    writer.join(timeout=1.0)

    assert not writer.is_alive()
    assert [sample.host_timestamp_s for sample in captured] == [1.0, 1.1]
    assert [sample.host_timestamp_s for sample in history.snapshot()] == [1.1, 1.2]


def test_add_cannot_modify_history_during_interpolation(monkeypatch) -> None:
    import visual_servo_system.geometry.robot_pose_history as history_module

    history = RobotPoseHistory(capacity=3, max_extrapolation_s=0.0)
    history.add(_sample(1.0, 0.0, 0.0))
    history.add(_sample(1.1, 0.1, 0.1))
    interpolation_entered = threading.Event()
    release_interpolation = threading.Event()
    writer_attempting = threading.Event()
    writer_finished = threading.Event()
    original_slerp = history_module._slerp_quaternion_xyzw

    def controlled_slerp(first, second, fraction):
        interpolation_entered.set()
        assert release_interpolation.wait(timeout=1.0)
        return original_slerp(first, second, fraction)

    monkeypatch.setattr(history_module, "_slerp_quaternion_xyzw", controlled_slerp)

    reader = threading.Thread(target=lambda: history.interpolate(1.05))

    def write_sample() -> None:
        writer_attempting.set()
        history.add(_sample(1.2, 0.2, 0.2))
        writer_finished.set()

    reader.start()
    assert interpolation_entered.wait(timeout=1.0)
    writer = threading.Thread(target=write_sample)
    writer.start()
    assert writer_attempting.wait(timeout=1.0)
    try:
        assert not writer_finished.wait(timeout=0.05)
    finally:
        release_interpolation.set()
        reader.join(timeout=1.0)
        writer.join(timeout=1.0)

    assert not reader.is_alive()
    assert not writer.is_alive()
    assert writer_finished.is_set()


def test_interpolation_with_an_unreliable_endpoint_is_fail_closed() -> None:
    history = RobotPoseHistory(capacity=3, max_extrapolation_s=0.0)
    history.add(_sample(1.0, 0.0, 0.0, reliable=True))
    history.add(_sample(1.1, 0.1, 0.0, reliable=False))

    result = history.interpolate(1.05)

    assert not result.valid
    assert not result.timestamp_reliable
    assert result.timestamp_source == "interpolated_mixed"


def test_clear_removes_every_pose_from_the_current_authorization_epoch() -> None:
    history = RobotPoseHistory(capacity=3, max_extrapolation_s=0.0)
    history.add(_sample(1.0, 0.0, 0.0))

    history.clear()

    assert len(history) == 0
    with pytest.raises(LookupError, match="empty"):
        history.interpolate(1.0)

