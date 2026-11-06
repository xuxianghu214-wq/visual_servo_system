import threading

import numpy as np
import pytest

from visual_servo_system.models.data_types import (
    MpcResult,
    PredictedTrajectory,
    TargetStateEstimate,
)
from visual_servo_system.safety.visual_servo_safety_monitor import (
    SafetyInputs,
    VisualServoSafetyMonitor,
    VisualServoState,
)


def estimate(covariance=None, age=0.0, *, timestamp_s=2.0):
    return TargetStateEstimate(
        timestamp_s,
        np.eye(4),
        np.zeros(3),
        np.zeros(3),
        np.eye(12) * 1e-8 if covariance is None else covariance,
        True,
        1.0,
        float(age),
    )


def mpc(twist=None, valid=True, *, timestamp_s=2.0, trajectory=None):
    return MpcResult(
        timestamp_s,
        np.zeros(6) if twist is None else np.asarray(twist, dtype=float),
        0.005,
        "solved" if valid else "failed",
        bool(valid),
        1.0 if valid else 0.0,
        trajectory,
    )


def observe_stable(monitor, count, *, first_sequence=1, first_timestamp_s=1.0):
    for offset in range(count):
        monitor.observe_target(
            True,
            0.01,
            np.deg2rad(1.0),
            frame_sequence=first_sequence + offset,
            observation_host_timestamp_s=first_timestamp_s + offset * 0.05,
        )


def ready_monitor():
    monitor = VisualServoSafetyMonitor()
    monitor.connect_preview()
    assert monitor.system_check(True, True, True, True)
    observe_stable(monitor, 10)
    monitor.enable(np.eye(4))
    monitor.start_following()
    return monitor


def safe_inputs(**changes):
    values = dict(
        calibration_valid=True,
        resolution_matches=True,
        tcp_zero_verified=True,
        robot_state_valid=True,
        latency_s=0.02,
        target_estimate=estimate(),
        mpc_result=mpc([0.01, 0, 0, 0, 0, 0]),
        commanded_flange_pose=np.eye(4),
        previous_flange_pose=np.eye(4),
    )
    values.update(changes)
    return SafetyInputs(**values)


def test_state_machine_requires_checks_and_ten_stable_frames():
    monitor = VisualServoSafetyMonitor()
    with pytest.raises(RuntimeError):
        monitor.enable(np.eye(4))
    monitor.connect_preview()
    assert monitor.system_check(True, True, True, True)
    observe_stable(monitor, 9)
    with pytest.raises(RuntimeError, match="stable"):
        monitor.enable(np.eye(4))
    monitor.observe_target(
        True,
        0.01,
        0.01,
        frame_sequence=10,
        observation_host_timestamp_s=1.45,
    )
    monitor.enable(np.eye(4))
    assert monitor.state is VisualServoState.ENABLED


def test_safe_motion_and_workspace_fault_are_fail_closed_and_latched():
    monitor = ready_monitor()
    decision = monitor.evaluate(2.0, safe_inputs())
    assert decision.allow_motion
    far = np.eye(4)
    far[0, 3] = 0.101
    stopped = monitor.evaluate(2.05, safe_inputs(commanded_flange_pose=far))
    assert not stopped.allow_motion
    assert np.all(stopped.safe_twist == 0)
    assert monitor.state is VisualServoState.FAULT
    with pytest.raises(RuntimeError):
        monitor.start_following()
    monitor.acknowledge()
    assert monitor.state is VisualServoState.PREVIEW


def test_occlusion_uncertainty_and_timeout_stop_motion():
    monitor = ready_monitor()
    monitor.enter_occlusion_decay()
    covariance = np.eye(12) * 1e-8
    covariance[3, 3] = (np.deg2rad(6.0) / 3.0) ** 2
    decision = monitor.evaluate(2.0, safe_inputs(target_estimate=estimate(covariance, 0.2)))
    assert not decision.allow_motion
    assert monitor.state is VisualServoState.FAULT


def test_user_stop_is_latched_until_acknowledged():
    monitor = ready_monitor()
    monitor.stop("user stop")
    decision = monitor.evaluate(2.0, safe_inputs())
    assert not decision.allow_motion
    assert monitor.state is VisualServoState.STOPPED
    monitor.acknowledge()
    assert monitor.state is VisualServoState.PREVIEW


def test_safety_uses_vector_speed_norm_and_covariance_eigenvalue():
    monitor = ready_monitor()
    diagonal_twist = mpc([0.04, 0.04, 0.04, 0.0, 0.0, 0.0])
    decision = monitor.evaluate(2.0, safe_inputs(mpc_result=diagonal_twist))
    assert not decision.allow_motion
    assert monitor.state is VisualServoState.FAULT

    monitor.acknowledge()
    assert monitor.system_check(True, True, True, True)
    observe_stable(monitor, 10, first_sequence=20, first_timestamp_s=3.0)
    monitor.enable(np.eye(4))
    monitor.start_following()
    covariance = np.eye(12) * 1e-8
    covariance[3:5, 3:5] = (
        np.array([[0.8, 0.6], [0.6, 0.8]]) * (np.deg2rad(5.0) / 3.0) ** 2
    )
    uncertain = monitor.evaluate(
        2.1, safe_inputs(target_estimate=estimate(covariance=covariance))
    )
    assert not uncertain.allow_motion
    assert "rotation uncertainty" in uncertain.reason


def test_safety_rejects_future_stale_and_mismatched_product_timestamps():
    stale_monitor = ready_monitor()
    stale = stale_monitor.evaluate(
        2.0,
        safe_inputs(
            latency_s=0.0,
            target_estimate=estimate(timestamp_s=1.899),
        ),
    )
    assert not stale.allow_motion
    assert "estimate timestamp is stale" in stale.reason

    future_monitor = ready_monitor()
    future = future_monitor.evaluate(
        2.0,
        safe_inputs(mpc_result=mpc(timestamp_s=2.001)),
    )
    assert not future.allow_motion
    assert "MPC result timestamp is in the future" in future.reason

    transforms = np.repeat(np.eye(4)[None, :, :], 20, axis=0)
    trajectory = PredictedTrajectory(
        1.899,
        np.arange(1, 21) * 0.05,
        transforms,
        np.zeros((20, 6)),
        True,
        1.0,
    )
    trajectory_monitor = ready_monitor()
    mismatch = trajectory_monitor.evaluate(
        2.0,
        safe_inputs(mpc_result=mpc(trajectory=trajectory)),
    )
    assert not mismatch.allow_motion
    assert "trajectory timestamp is stale" in mismatch.reason


def test_repeated_or_out_of_order_frame_never_satisfies_stability_gate():
    monitor = VisualServoSafetyMonitor()
    monitor.connect_preview()
    assert monitor.system_check(True, True, True, True)
    for _ in range(10):
        monitor.observe_target(
            True,
            0.01,
            0.01,
            frame_sequence=7,
            observation_host_timestamp_s=1.0,
        )
    assert monitor.stable_frames == 0
    with pytest.raises(RuntimeError, match="stable"):
        monitor.enable(np.eye(4))

    observe_stable(monitor, 10, first_sequence=8, first_timestamp_s=1.05)
    monitor.enable(np.eye(4))
    assert monitor.state is VisualServoState.ENABLED


def test_stability_gate_resets_after_large_observation_gap():
    monitor = VisualServoSafetyMonitor()
    monitor.connect_preview()
    assert monitor.system_check(True, True, True, True)
    observe_stable(monitor, 5)
    monitor.observe_target(
        True,
        0.01,
        0.01,
        frame_sequence=6,
        observation_host_timestamp_s=2.0,
    )
    assert monitor.stable_frames == 1


def test_new_hardware_epoch_explicitly_discards_dry_run_stability():
    monitor = VisualServoSafetyMonitor()
    monitor.connect_preview()
    assert monitor.system_check(True, True, True, True)
    observe_stable(monitor, 10)
    assert monitor.stable_frames == 10

    monitor.reset_target_stability()

    assert monitor.state is VisualServoState.VALIDATED
    assert monitor.stable_frames == 0


def test_guarded_execution_does_not_block_stop_and_rejects_stale_decision():
    monitor = ready_monitor()
    decision = monitor.evaluate(2.0, safe_inputs())
    callback_entered = threading.Event()
    callback_release = threading.Event()
    stop_finished = threading.Event()
    sent = []

    def blocking_sender(twist):
        sent.append(np.array(twist, copy=True))
        callback_entered.set()
        assert callback_release.wait(timeout=1.0)

    execute_thread = threading.Thread(
        target=monitor.execute_guarded,
        args=(decision, blocking_sender),
        kwargs={"now_s": 2.0},
    )
    execute_thread.start()
    assert callback_entered.wait(timeout=1.0)

    def request_stop():
        monitor.stop("concurrent user stop")
        stop_finished.set()

    stop_thread = threading.Thread(target=request_stop)
    stop_thread.start()
    # callback 可能卡在厂商 SDK；安全锁绝不能因此阻塞人工停止和代次失效。
    assert stop_finished.wait(timeout=0.05)
    assert monitor.state is VisualServoState.STOPPED
    assert monitor.inflight_guard_count == 1
    callback_release.set()
    execute_thread.join(timeout=1.0)
    stop_thread.join(timeout=1.0)
    assert stop_finished.is_set()
    assert monitor.inflight_guard_count == 0
    assert len(sent) == 1
    assert monitor.state is VisualServoState.STOPPED

    assert not monitor.execute_guarded(
        decision, lambda twist: sent.append(twist), now_s=2.0
    )
    assert len(sent) == 1


def test_guarded_execution_checks_decision_age_future_and_latency_boundary():
    monitor = ready_monitor()
    decision = monitor.evaluate(2.0, safe_inputs())
    sent = []

    assert monitor.execute_guarded(
        decision, lambda twist: sent.append(np.array(twist)), now_s=2.10
    )
    assert len(sent) == 1

    assert not monitor.execute_guarded(
        decision, lambda twist: sent.append(np.array(twist)), now_s=2.125
    )
    assert "stale" in monitor.last_guard_rejection_reason
    assert len(sent) == 1

    assert not monitor.execute_guarded(
        decision, lambda twist: sent.append(np.array(twist)), now_s=1.999
    )
    assert "future" in monitor.last_guard_rejection_reason
    assert len(sent) == 1

    monitor.stop("operator stop")
    assert not monitor.execute_guarded(
        decision, lambda twist: sent.append(np.array(twist)), now_s=2.05
    )
    assert "generation" in monitor.last_guard_rejection_reason
    assert len(sent) == 1


def test_guarded_execution_uses_injected_monotonic_clock_by_default():
    monitor = VisualServoSafetyMonitor(clock=lambda: 2.05)
    monitor.connect_preview()
    assert monitor.system_check(True, True, True, True)
    observe_stable(monitor, 10)
    monitor.enable(np.eye(4))
    monitor.start_following()
    decision = monitor.evaluate(2.0, safe_inputs())
    sent = []
    assert monitor.execute_guarded(decision, lambda twist: sent.append(twist))
    assert len(sent) == 1
