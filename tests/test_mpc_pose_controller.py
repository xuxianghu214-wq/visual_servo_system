from types import SimpleNamespace

import numpy as np
import pytest

from visual_servo_system.config.config_loader import load_system_config
from visual_servo_system.control.mpc_pose_controller import (
    MpcPoseController,
    MpcPoseControllerConfig,
)
from visual_servo_system.geometry.pose_transforms import so3_exp
from visual_servo_system.models.data_types import PredictedTrajectory


def test_mpc_runtime_limits_are_loaded_from_yaml():
    config = load_system_config("config_files")
    runtime = MpcPoseControllerConfig.from_settings(config.mpc, config.safety)
    assert runtime.max_linear_acceleration_mps2 == 0.25
    assert runtime.max_angular_acceleration_rad_s2 == np.deg2rad(90.0)
    assert runtime.commissioning_angular_speed_rad_s == np.deg2rad(15.0)
    assert runtime.experiment_angular_speed_rad_s == np.deg2rad(30.0)


def desired_x(x=0.1, *, timestamp_s=1.0):
    transforms = np.repeat(np.eye(4)[None, :, :], 20, axis=0)
    transforms[:, 0, 3] = x
    return PredictedTrajectory(
        timestamp_s,
        np.arange(1, 21, dtype=float) * 0.05,
        transforms,
        np.zeros((20, 6)),
        True,
        1.0,
    )


def test_mpc_moves_toward_positive_target_and_honours_commissioning_limits():
    controller = MpcPoseController()
    result = controller.solve(np.eye(4), np.zeros(6), desired_x(), 1.0, "commissioning")
    assert result.valid, result.solver_status
    assert 0.0 < result.commanded_twist[0] <= 0.05
    assert np.max(np.abs(result.commanded_twist[:3])) <= 0.05 + 1e-9
    assert np.max(np.abs(result.commanded_twist[3:])) <= np.deg2rad(15) + 1e-9


def test_mpc_acceleration_bound_and_warm_updates():
    controller = MpcPoseController()
    first = controller.solve(np.eye(4), np.zeros(6), desired_x(), 1.0, "experiment")
    second = controller.solve(np.eye(4), first.commanded_twist, desired_x(0.11), 1.05, "experiment")
    assert first.valid and second.valid
    assert abs(first.commanded_twist[0]) <= 0.25 * 0.05 + 1e-6
    assert controller.solve_count == 2


def test_mpc_invalid_input_and_solver_timeout_fail_closed():
    invalid = desired_x()
    invalid = PredictedTrajectory(
        invalid.host_timestamp_s,
        invalid.sample_times_s,
        invalid.transforms,
        invalid.twists,
        False,
        0.0,
        "uncertain",
    )
    controller = MpcPoseController()
    result = controller.solve(np.eye(4), np.zeros(6), invalid, 1.0, "commissioning")
    assert not result.valid
    assert np.all(result.commanded_twist == 0)
    too_fast = MpcPoseController(MpcPoseControllerConfig(solver_time_limit_s=0.0))
    timed_out = too_fast.solve(np.eye(4), np.zeros(6), desired_x(), 1.0, "commissioning")
    assert not timed_out.valid
    assert "time" in timed_out.solver_status


class _FailingSolver:
    def setup(self, **_kwargs):
        pass

    def update(self, **_kwargs):
        raise RuntimeError("synthetic solver failure")

    def solve(self, **_kwargs):
        raise AssertionError("solve must not follow a failed update")


class _StatusSolver:
    status = "primal infeasible"
    solution = np.zeros(120)

    def setup(self, **_kwargs):
        pass

    def update(self, **_kwargs):
        pass

    def solve(self, **_kwargs):
        return SimpleNamespace(info=SimpleNamespace(status=self.status), x=self.solution.copy())


def test_mpc_reset_rebuilds_solver_and_discards_warm_start_epoch():
    created = []

    def factory():
        solver = _StatusSolver()
        created.append(solver)
        return solver

    controller = MpcPoseController(solver_factory=factory)
    controller.solve(np.eye(4), np.zeros(6), desired_x(), 1.0, "commissioning")
    assert controller.solve_count == 1

    controller.reset()

    assert len(created) == 2
    assert controller.solve_count == 0


def test_mpc_solver_exception_and_infeasible_status_fail_closed():
    failed = MpcPoseController(solver_factory=_FailingSolver).solve(
        np.eye(4), np.zeros(6), desired_x(), 1.0, "commissioning"
    )
    assert not failed.valid
    assert "exception" in failed.solver_status
    infeasible = MpcPoseController(solver_factory=_StatusSolver).solve(
        np.eye(4), np.zeros(6), desired_x(), 1.0, "commissioning"
    )
    assert not infeasible.valid
    assert "infeasible" in infeasible.solver_status


def test_mpc_uses_future_reference_not_only_first_sample():
    transforms = np.repeat(np.eye(4)[None, :, :], 20, axis=0)
    transforms[:, 0, 3] = np.linspace(0.0, 0.1, 20)
    twists = np.zeros((20, 6))
    twists[:, 0] = 0.1
    moving = PredictedTrajectory(
        1.0, np.arange(1, 21) * 0.05, transforms, twists, True, 1.0
    )
    result = MpcPoseController().solve(
        np.eye(4), np.zeros(6), moving, 1.0, "experiment"
    )
    assert result.valid
    assert result.commanded_twist[0] > 0.0


def test_mpc_rotation_command_is_expressed_in_base_spatial_coordinates():
    current = np.eye(4)
    current[:3, :3] = so3_exp([np.pi / 2, 0.0, 0.0])
    desired = np.repeat(current[None, :, :], 20, axis=0)
    desired[:, :3, :3] = so3_exp([0.0, 0.0, 0.2]) @ current[:3, :3]
    trajectory = PredictedTrajectory(
        1.0, np.arange(1, 21) * 0.05, desired, np.zeros((20, 6)), True, 1.0
    )
    result = MpcPoseController().solve(
        current, np.zeros(6), trajectory, 1.0, "experiment"
    )
    assert result.valid
    assert result.commanded_twist[5] > 0.0
    assert abs(result.commanded_twist[4]) < abs(result.commanded_twist[5]) * 0.1


def test_mpc_vector_norm_limits_hold_for_three_axis_motion():
    trajectory = desired_x()
    transforms = np.array(trajectory.transforms, copy=True)
    transforms[:, :3, 3] = [0.2, 0.2, 0.2]
    diagonal = PredictedTrajectory(
        1.0, trajectory.sample_times_s, transforms, trajectory.twists, True, 1.0
    )
    result = MpcPoseController().solve(
        np.eye(4), np.zeros(6), diagonal, 1.0, "commissioning"
    )
    assert result.valid
    assert np.linalg.norm(result.commanded_twist[:3]) <= 0.05 + 1e-9
    assert np.linalg.norm(result.commanded_twist[:3]) <= 0.25 * 0.05 + 1e-6


def test_mpc_cold_start_diagonal_target_is_valid_with_explicit_solver_tolerances():
    trajectory = desired_x()
    transforms = np.array(trajectory.transforms, copy=True)
    transforms[:, :3, 3] = [0.04, 0.04, 0.04]
    diagonal = PredictedTrajectory(
        1.0, trajectory.sample_times_s, transforms, trajectory.twists, True, 1.0
    )
    controller = MpcPoseController()
    result = controller.solve(np.eye(4), np.zeros(6), diagonal, 1.0, "commissioning")
    assert result.valid, result.solver_status
    assert np.linalg.norm(result.commanded_twist[:3]) <= 0.0125 + 5e-6
    assert controller.solver_absolute_tolerance <= 1e-6


def test_mpc_rejects_inaccurate_nonfinite_or_later_constraint_violation():
    _StatusSolver.status = "solved inaccurate"
    _StatusSolver.solution = np.zeros(120)
    inaccurate = MpcPoseController(solver_factory=_StatusSolver).solve(
        np.eye(4), np.zeros(6), desired_x(), 1.0, "commissioning"
    )
    assert not inaccurate.valid

    _StatusSolver.status = "solved"
    _StatusSolver.solution = np.zeros(120)
    _StatusSolver.solution[-1] = np.nan
    nonfinite = MpcPoseController(solver_factory=_StatusSolver).solve(
        np.eye(4), np.zeros(6), desired_x(), 1.0, "commissioning"
    )
    assert not nonfinite.valid

    _StatusSolver.solution = np.zeros(120)
    _StatusSolver.solution[6:12] = 100.0
    violated = MpcPoseController(solver_factory=_StatusSolver).solve(
        np.eye(4), np.zeros(6), desired_x(), 1.0, "commissioning"
    )
    assert not violated.valid
    assert np.all(violated.commanded_twist == 0.0)


class _MissingInfoResult:
    x = np.zeros(120)


class _ExplodingXResult:
    info = SimpleNamespace(status="solved")

    @property
    def x(self):
        raise RuntimeError("malicious x property")


class _WeirdResultSolver(_StatusSolver):
    result = _MissingInfoResult()

    def solve(self, **_kwargs):
        return self.result


def test_mpc_malformed_solver_result_is_always_fail_closed():
    for malformed in (_MissingInfoResult(), _ExplodingXResult(), SimpleNamespace(info=None, x=[])):
        _WeirdResultSolver.result = malformed
        result = MpcPoseController(solver_factory=_WeirdResultSolver).solve(
            np.eye(4), np.zeros(6), desired_x(), 1.0, "commissioning"
        )
        assert not result.valid
        assert np.all(result.commanded_twist == 0.0)
        assert "result" in result.solver_status


def test_mpc_rejects_future_and_stale_reference_anchors_at_explicit_boundary():
    controller = MpcPoseController(MpcPoseControllerConfig(max_reference_age_s=0.10))

    boundary = controller.solve(
        np.eye(4), np.zeros(6), desired_x(timestamp_s=0.90), 1.0, "commissioning"
    )
    assert boundary.valid, boundary.solver_status

    stale = controller.solve(
        np.eye(4), np.zeros(6), desired_x(timestamp_s=0.899), 1.0, "commissioning"
    )
    assert not stale.valid
    assert "stale" in stale.solver_status
    assert np.all(stale.commanded_twist == 0.0)

    future = controller.solve(
        np.eye(4), np.zeros(6), desired_x(timestamp_s=1.001), 1.0, "commissioning"
    )
    assert not future.valid
    assert "future" in future.solver_status
    assert np.all(future.commanded_twist == 0.0)


def test_mpc_preserves_absolute_time_when_accepting_recent_reference():
    sample_times = np.arange(1, 21, dtype=float) * 0.05
    old_transforms = np.repeat(np.eye(4)[None, :, :], 20, axis=0)
    old_transforms[:, 0, 3] = sample_times * 0.005
    old = PredictedTrajectory(
        0.95,
        sample_times,
        old_transforms,
        np.zeros((20, 6)),
        True,
        1.0,
    )
    aligned_transforms = np.array(old_transforms, copy=True)
    aligned_transforms[:, 0, 3] = np.minimum(sample_times + 0.05, 1.0) * 0.005
    aligned = PredictedTrajectory(
        1.0,
        sample_times,
        aligned_transforms,
        np.zeros((20, 6)),
        True,
        1.0,
    )

    old_result = MpcPoseController().solve(
        np.eye(4), np.zeros(6), old, 1.0, "experiment"
    )
    aligned_result = MpcPoseController().solve(
        np.eye(4), np.zeros(6), aligned, 1.0, "experiment"
    )
    assert old_result.valid and aligned_result.valid
    assert old_result.commanded_twist == pytest.approx(
        aligned_result.commanded_twist, abs=1e-8
    )

