import numpy as np
import pytest

from visual_servo_system.estimation.target_pose_ekf import TargetPoseEkf, TargetPoseEkfConfig
from visual_servo_system.geometry.pose_transforms import so3_exp, so3_log


def pose(position=(0.0, 0.0, 0.5), rotation=(0.0, 0.0, 0.0)):
    result = np.eye(4)
    result[:3, :3] = so3_exp(np.asarray(rotation, dtype=float))
    result[:3, 3] = position
    return result


def test_ekf_estimates_constant_velocity_and_keeps_psd():
    ekf = TargetPoseEkf()
    ekf.update(pose(), 1.0)
    for index in range(1, 21):
        t = 1.0 + index * 0.05
        estimate = ekf.update(pose((0.02 * (t - 1.0), 0.0, 0.5)), t)
    assert estimate.linear_velocity_mps[0] == pytest.approx(0.02, abs=0.005)
    assert np.min(np.linalg.eigvalsh(estimate.covariance)) >= -1e-10


def test_ekf_rejects_mahalanobis_outlier_without_resetting_state():
    ekf = TargetPoseEkf()
    ekf.update(pose(), 1.0)
    accepted = ekf.update(pose((0.001, 0.0, 0.5)), 1.05)
    rejected = ekf.update(pose((2.0, 0.0, 0.5)), 1.10)
    assert not ekf.last_update_accepted
    assert "mahalanobis" in ekf.last_rejection_reason
    assert rejected.T_base_target[0, 3] < 0.02
    assert rejected.measurement_age_s == pytest.approx(0.05)
    assert accepted.valid


def test_ekf_so3_innovation_is_continuous_across_pi():
    ekf = TargetPoseEkf()
    first = np.deg2rad(179.0)
    second = np.deg2rad(-179.0)
    ekf.update(pose(rotation=(0.0, 0.0, first)), 1.0)
    estimate = ekf.update(pose(rotation=(0.0, 0.0, second)), 1.05)
    error = so3_log(estimate.T_base_target[:3, :3].T @ pose(rotation=(0, 0, second))[:3, :3])
    assert np.linalg.norm(error) < np.deg2rad(3.0)


def test_ekf_rejects_bad_time_and_nonfinite_measurement():
    ekf = TargetPoseEkf()
    ekf.update(pose(), 1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        ekf.update(pose(), 1.0)
    bad = pose()
    bad[0, 3] = np.nan
    with pytest.raises(ValueError):
        ekf.update(bad, 1.1)


def test_ekf_rotates_anisotropic_right_error_covariance_during_90deg_motion():
    ekf = TargetPoseEkf(
        TargetPoseEkfConfig(
            angular_acceleration_std_rad_s2=1e-9,
            linear_acceleration_std_mps2=1e-9,
            max_dt_s=0.25,
        )
    )
    ekf.update(pose(), 1.0)
    ekf._covariance[:] = 0.0
    ekf._covariance[3:6, 3:6] = np.diag([1e-4, 4e-4, 9e-4])
    ekf._covariance[9:12, 9:12] = np.eye(3) * 1e-15
    ekf._angular_velocity[:] = [0.0, 0.0, 2.0 * np.pi]
    predicted = ekf.predict_to(1.25)
    assert predicted.covariance[3, 3] == pytest.approx(4e-4, rel=1e-4)
    assert predicted.covariance[4, 4] == pytest.approx(1e-4, rel=1e-4)


def test_ekf_large_accepted_orientation_innovation_keeps_covariance_psd():
    ekf = TargetPoseEkf(TargetPoseEkfConfig(mahalanobis_gate_99_6d=1e9))
    ekf.update(pose(), 1.0)
    result = ekf.update(pose(rotation=(2.0, -0.8, 0.4)), 1.05)
    assert ekf.last_update_accepted
    assert np.min(np.linalg.eigvalsh(result.covariance)) >= -1e-10


def test_reset_discards_the_previous_target_epoch():
    ekf = TargetPoseEkf()
    first = np.eye(4)
    first[0, 3] = 0.4
    ekf.update(first, 1.0)

    ekf.reset()

    assert not ekf.initialized
    second = np.eye(4)
    second[0, 3] = -0.2
    estimate = ekf.update(second, 2.0)
    assert estimate.T_base_target[0, 3] == pytest.approx(-0.2)
    assert ekf.last_rejection_reason == "accepted initial measurement"

