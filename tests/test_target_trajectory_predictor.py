import numpy as np
import pytest

from visual_servo_system.estimation.target_trajectory_predictor import (
    TargetTrajectoryPredictor,
    TrajectoryPredictionConfig,
)
from visual_servo_system.models.data_types import TargetStateEstimate


def estimate(*, age=0.0, velocity=(0.1, 0.0, 0.0), covariance=None):
    return TargetStateEstimate(
        host_timestamp_s=1.0,
        T_base_target=np.eye(4),
        linear_velocity_mps=np.asarray(velocity, dtype=float),
        angular_velocity_rad_s=np.zeros(3),
        covariance=np.eye(12) * 1e-8 if covariance is None else covariance,
        valid=True,
        quality=1.0,
        measurement_age_s=float(age),
    )


def test_prediction_has_twenty_steps_and_constant_velocity_when_visible():
    predictor = TargetTrajectoryPredictor()
    result = predictor.predict(estimate(), 1.0)
    assert result.valid
    assert result.transforms.shape == (20, 4, 4)
    assert result.sample_times_s[-1] == pytest.approx(1.0)
    assert result.transforms[-1, 0, 3] == pytest.approx(0.1)


def test_occlusion_velocity_decays_smoothly():
    predictor = TargetTrajectoryPredictor(
        TrajectoryPredictionConfig(velocity_decay_time_constant_s=0.25)
    )
    result = predictor.predict(estimate(age=0.2), 1.2)
    assert result.valid
    assert 0.0 < result.twists[-1, 0] < result.twists[0, 0] < 0.1
    assert np.all(np.diff(result.transforms[:, 0, 3]) > 0)


def test_occlusion_decay_uses_total_measurement_age_not_each_call_age():
    predictor = TargetTrajectoryPredictor(
        TrajectoryPredictionConfig(velocity_decay_time_constant_s=0.25)
    )
    early = predictor.predict(estimate(age=0.2), 1.0)
    late = predictor.predict(estimate(age=0.8), 1.0)
    assert late.twists[0, 0] < early.twists[0, 0] * 0.2


def test_prediction_stops_at_one_second_or_excess_uncertainty():
    predictor = TargetTrajectoryPredictor()
    timeout = predictor.predict(estimate(age=1.0), 2.0)
    assert not timeout.valid
    assert "dropout" in timeout.reason
    covariance = np.eye(12) * 1e-8
    covariance[0, 0] = (0.031 / 3.0) ** 2
    uncertain = predictor.predict(estimate(covariance=covariance), 1.0)
    assert not uncertain.valid
    assert "position uncertainty" in uncertain.reason


def test_prediction_propagates_state_and_covariance_to_requested_now():
    predictor = TargetTrajectoryPredictor()
    delayed = predictor.predict(estimate(velocity=(0.1, 0.0, 0.0)), 1.1)
    # 请求时间晚于估计时间时，基准位姿也必须前推，不能只增加“年龄”。
    assert delayed.transforms[0, 0, 3] > 0.0
    assert delayed.twists[0, 0] < 0.1


def test_correlated_covariance_uses_largest_eigenvalue_for_3sigma():
    covariance = np.eye(12) * 1e-8
    covariance[0:2, 0:2] = [[8e-5, 6e-5], [6e-5, 8e-5]]
    result = TargetTrajectoryPredictor().predict(estimate(covariance=covariance), 1.0)
    assert not result.valid
    assert "position uncertainty" in result.reason


def test_rotating_delayed_covariance_uses_same_so3_transition_as_ekf():
    covariance = np.eye(12) * 1e-10
    covariance[3:6, 3:6] = np.diag([1e-5, 2e-5, 3e-5])
    covariance[9:12, 9:12] = np.diag([0.012, 0.0002, 0.0002])
    # 相关项仍构成 PSD；它能捕获遮挡衰减下姿态与角速度的不确定性耦合。
    covariance[3, 9] = covariance[9, 3] = 2.0e-4
    rotating = TargetStateEstimate(
        host_timestamp_s=1.0,
        T_base_target=np.eye(4),
        linear_velocity_mps=np.zeros(3),
        angular_velocity_rad_s=np.array([0.0, 0.0, np.deg2rad(30.0)]),
        covariance=covariance,
        valid=True,
        quality=1.0,
        measurement_age_s=0.0,
    )
    result = TargetTrajectoryPredictor().predict(rotating, 1.8)
    assert not result.valid
    assert "rotation uncertainty" in result.reason

