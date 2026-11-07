from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest

from visual_servo_system.estimation.target_pose_ekf import EkfTimestampGapError
from visual_servo_system.geometry.robot_pose_history import RobotPoseHistory
from visual_servo_system.models.data_types import (
    FramePacket,
    MarkerPoseObservation,
    MpcResult,
    RobotStateSample,
)
from visual_servo_system.robot.deferred_ur_rtde_servo_sender import (
    DeferredUrRtdeServoSender,
)
from visual_servo_system.robot.ur_rtde_servo_sender import (
    FakeRobotBackend,
    UrRtdeServoSender,
)
from visual_servo_system.runtime.latest_value_buffer import LatestValueBuffer
from visual_servo_system.runtime.visual_servo_orchestrator import (
    RuntimeMode,
    RuntimeShutdownError,
    RuntimeStopUnconfirmedError,
    VisualServoOrchestrator,
)
from visual_servo_system.safety.visual_servo_safety_monitor import VisualServoState


class _Buffer:
    def __init__(self, value=None):
        self.value = value

    def get_latest(self):
        return None if self.value is None else (1, self.value)


class _CameraWorker:
    def __init__(self, frame):
        self.latest_frame = _Buffer(frame)
        self.started = 0
        self.stopped = 0
        self.last_error = None

    def start(self):
        self.started += 1

    def stop(self, timeout_s=2.0):
        self.stopped += 1


class _Reader:
    def __init__(self, sample):
        from visual_servo_system.geometry.robot_pose_history import RobotPoseHistory

        self.latest_state = _Buffer(sample)
        self.pose_history = RobotPoseHistory(8, 0.1)
        self.pose_history.add(sample)
        self.started = 0
        self.stopped = 0
        self.connected = False
        self.tcp_zero_confirmed = False
        self.communication_healthy = True
        self.watchdog_fault_event = type("E", (), {"is_set": lambda self: False})()
        self.last_error = None

    def start(self):
        self.started += 1
        self.connected = True

    def stop(self, timeout_s=2.0):
        self.stopped += 1
        self.connected = False

    def confirm_zero_tcp(self, value):
        assert np.all(np.asarray(value) == 0)
        self.tcp_zero_confirmed = True

    def revoke_tcp_confirmation(self):
        self.tcp_zero_confirmed = False


class _Sender:
    def __init__(self):
        self.connected = 0
        self.authorized = 0
        self.sent: list[np.ndarray] = []
        self.stopped = 0
        self.disconnected = 0

    def connect(self):
        if self.connected == 0:
            self.connected = 1

    def authorize_state_reader(self, reader):
        self.authorized += 1
        reader.confirm_zero_tcp(np.zeros(6))

    def send_pose(self, pose):
        self.sent.append(np.array(pose, copy=True))

    def stop(self):
        self.stopped += 1

    def disconnect(self):
        self.disconnected += 1


class _Detector:
    def __init__(self, observation):
        self.observation = observation
        self.calls = 0

    def detect(self, _image, _timestamp):
        self.calls += 1
        return self.observation

    def draw_detection(self, image, _observation):
        return np.array(image, copy=True)

    def reset_tracking(self):
        pass


class _TimestampDetector(_Detector):
    """用当前图像时刳返回固定相机-标签位姿。"""

    def __init__(self, transform):
        super().__init__(None)
        self.transform = np.array(transform, copy=True)
        self.reset_calls = 0

    def detect(self, _image, timestamp):
        self.calls += 1
        return MarkerPoseObservation(
            float(timestamp),
            0,
            self.transform,
            0.1,
            None,
            1200.0,
            True,
            1.0,
        )

    def reset_tracking(self):
        self.reset_calls += 1


class _EpochReader:
    """不连网的授权代次读取器，用于验证严格时序。"""

    def __init__(self, initial_sample):
        self.latest_state = LatestValueBuffer()
        self.latest_state.publish(initial_sample)
        self.pose_history = RobotPoseHistory(128, 0.1)
        self.started = 0
        self.stopped = 0
        self.connected = False
        self.tcp_zero_confirmed = False
        self.communication_healthy = True
        self.watchdog_fault_event = type("E", (), {"is_set": lambda self: False})()
        self.last_error = None
        self.epoch_started = threading.Event()
        self.epoch_not_before_s = None

    def start(self):
        self.started += 1
        self.connected = True

    def stop(self, timeout_s=2.0):
        self.stopped += 1
        self.connected = False

    def confirm_zero_tcp(self, value):
        assert np.all(np.asarray(value) == 0)
        self.tcp_zero_confirmed = True
        self.pose_history.clear()

    def begin_authorized_pose_epoch(self, not_before_host_s):
        assert self.tcp_zero_confirmed
        self.epoch_not_before_s = float(not_before_host_s)
        self.pose_history.clear()
        self.epoch_started.set()

    def revoke_tcp_confirmation(self):
        self.tcp_zero_confirmed = False
        self.pose_history.clear()

    def publish(self, sample):
        if (
            self.tcp_zero_confirmed
            and self.epoch_not_before_s is not None
            and sample.valid
            and sample.timestamp_reliable
            and sample.host_timestamp_s >= self.epoch_not_before_s
            and sample.receive_timestamp_s >= self.epoch_not_before_s
        ):
            self.pose_history.add(sample)
        self.latest_state.publish(sample)


@dataclass
class _Calibration:
    T_flange_camera: np.ndarray


def _identity_at(x=0.0):
    value = np.eye(4)
    value[0, 3] = x
    return value


def _objects():
    frame = FramePacket(
        1.0,
        7,
        np.zeros((6, 8), np.uint8),
        True,
        1.0,
        receive_timestamp_s=1.001,
        timestamp_source="camera_device",
        timestamp_reliable=True,
        timestamp_uncertainty_s=0.001,
    )
    observation = MarkerPoseObservation(1.0, 0, _identity_at(0.5), 0.2, None, 1000.0, True, 1.0)
    sample = RobotStateSample(
        1.0,
        np.eye(4),
        sequence_number=1,
        robot_mode="running",
        valid=True,
        quality=1.0,
        receive_timestamp_s=1.001,
        timestamp_source="ur_controller",
        timestamp_reliable=True,
        timestamp_uncertainty_s=0.001,
    )
    return frame, observation, sample


def _runtime(performance_profiler=None):
    frame, observation, sample = _objects()
    sender = _Sender()
    runtime = VisualServoOrchestrator(
        camera_worker=_CameraWorker(frame),
        state_reader=_Reader(sample),
        real_sender=sender,
        detector=_Detector(observation),
        project_root=None,
        clock=lambda: 1.0,
        start_background_threads=False,
        performance_profiler=performance_profiler,
    )
    return runtime, sender


def test_vision_pipeline_profiles_lock_queue_detection_ekf_and_total() -> None:
    class Recorder:
        def __init__(self):
            self.records = []

        def record(self, **values):
            self.records.append(values)

    recorder = Recorder()
    runtime, _ = _runtime(recorder)
    runtime.start_preview()
    runtime.connect_robot_read_only()
    runtime.install_calibration_for_testing(np.eye(4), (8, 6))
    assert runtime.system_check()

    assert runtime.process_vision_step().valid

    stages = {record["stage"] for record in recorder.records}
    assert {
        "pipeline_lock_wait",
        "frame_queue_age",
        "marker_detect",
        "annotation",
        "robot_pose_interpolation",
        "ekf_update",
        "vision_step_total",
    } <= stages
    assert next(
        record for record in recorder.records if record["stage"] == "pipeline_lock_wait"
    )["kind"] == "lock"


def test_default_is_safe_and_order_is_enforced():
    runtime, sender = _runtime()
    assert runtime.mode is RuntimeMode.PREVIEW
    assert runtime.state is VisualServoState.DISCONNECTED
    assert sender.connected == 0
    with pytest.raises(RuntimeError):
        runtime.enable()
    runtime.start_preview()
    runtime.connect_robot_read_only()
    assert sender.connected == 0


def test_invalid_calibration_keeps_preview_only(tmp_path):
    runtime, _ = _runtime()
    runtime.project_root = tmp_path
    (tmp_path / "calibration_files").mkdir()
    runtime.start_preview()
    runtime.connect_robot_read_only()
    assert not runtime.system_check()
    assert runtime.state is VisualServoState.PREVIEW
    # 标定无效只禁止位姿/控制，不应阻断安全的相机原始画面预览。
    assert runtime.ui_snapshot()["image"].shape == (6, 8)


def test_dry_run_never_connects_real_sender_and_stop_is_latched():
    runtime, sender = _runtime()
    runtime.detector = _Detector(_desired_marker_observation())
    runtime.start_preview()
    runtime.connect_robot_read_only()
    runtime.install_calibration_for_testing(np.eye(4), (8, 6))
    assert runtime.system_check()
    assert sender.connected == 0
    assert sender.authorized == 0
    runtime.start_dry_run()
    assert runtime.mode is RuntimeMode.DRY_RUN
    assert runtime.ui_snapshot()["dry_run_completed"]
    assert len(runtime._dry_sender.sent_poses) >= 1
    # 系统检查和完整干运行都不能接触真实控制发送器。
    assert sender.connected == 0
    assert sender.authorized == 0
    assert sender.sent == []
    runtime.emergency_stop("test stop")
    assert runtime.state is VisualServoState.STOPPED
    assert sender.stopped >= 1
    with pytest.raises(RuntimeError):
        runtime.start_following()
    runtime.acknowledge_stop()
    assert runtime.state is VisualServoState.PREVIEW


def test_process_frame_uses_image_time_pose_and_correct_transform():
    runtime, _ = _runtime()
    runtime.start_preview()
    runtime.connect_robot_read_only()
    runtime.install_calibration_for_testing(_identity_at(0.1), (8, 6))
    assert runtime.system_check()
    observation = runtime.process_vision_step()
    assert observation.valid
    estimate = runtime.latest_estimate.get_latest()[1]
    # base->flange 0, flange->camera +0.1, camera->marker +0.5
    assert estimate.T_base_target[0, 3] == pytest.approx(0.6)


def test_validated_pipeline_recovers_from_ekf_timestamp_gap_without_faulting():
    """未使能时的慢帧不得杀死视觉线程；应丢弃旧 EKF 代次并从当前观测重新开始。"""

    runtime, _ = _runtime()
    runtime.detector = _TimestampDetector(_desired_marker_observation().T_camera_marker)
    runtime.start_preview()
    runtime.connect_robot_read_only()
    runtime.install_calibration_for_testing(np.eye(4), (8, 6))
    assert runtime.system_check()

    first = FramePacket(
        1.0,
        7,
        np.zeros((6, 8), np.uint8),
        True,
        1.0,
        receive_timestamp_s=1.001,
        timestamp_source="camera_device",
        timestamp_reliable=True,
        timestamp_uncertainty_s=0.001,
    )
    delayed = FramePacket(
        1.268386,
        15,
        np.zeros((6, 8), np.uint8),
        True,
        1.0,
        receive_timestamp_s=1.269,
        timestamp_source="camera_device",
        timestamp_reliable=True,
        timestamp_uncertainty_s=0.001,
    )

    assert runtime.process_vision_step(first).valid
    assert runtime.process_vision_step(delayed).valid
    assert runtime.state is VisualServoState.VALIDATED
    estimate = runtime.latest_estimate.get_latest()[1]
    assert estimate.host_timestamp_s == pytest.approx(delayed.host_timestamp_s)
    assert estimate.measurement_age_s == pytest.approx(0.0)
    assert runtime.safety.stable_frames == 1


def test_enabled_pipeline_keeps_timestamp_gap_as_a_fail_safe_error():
    """已经使能后不得静默重置 EKF；间隔超限必须交给上层停车。"""

    runtime, _ = _runtime()
    runtime.detector = _TimestampDetector(_desired_marker_observation().T_camera_marker)
    runtime.start_preview()
    runtime.connect_robot_read_only()
    runtime.install_calibration_for_testing(np.eye(4), (8, 6))
    assert runtime.system_check()
    runtime.start_dry_run()
    runtime.force_stable_target_for_testing()
    runtime.enable()
    assert runtime.state is VisualServoState.ENABLED

    delayed = FramePacket(
        1.268386,
        15,
        np.zeros((6, 8), np.uint8),
        True,
        1.0,
        receive_timestamp_s=1.269,
        timestamp_source="camera_device",
        timestamp_reliable=True,
        timestamp_uncertainty_s=0.001,
    )

    with pytest.raises(EkfTimestampGapError, match="max_dt_s"):
        runtime.process_vision_step(delayed)


def test_enable_hardware_authorizes_reader_only_after_stable_unique_frames():
    runtime, reader, sender, _ = _hardware_epoch_runtime(
        flange_x=0.0, marker_x=0.0
    )
    errors = _run_hardware_enable_with_frames(runtime, reader, flange_x=0.0)
    assert errors == []
    assert sender.connected == 1
    assert sender.authorized >= 1
    assert runtime.state is VisualServoState.ENABLED


def _hardware_epoch_runtime(*, flange_x: float, marker_x: float):
    frame, _, initial = _objects()
    initial = RobotStateSample(
        initial.host_timestamp_s,
        _identity_at(flange_x),
        sequence_number=1,
        robot_mode="pre_authorization",
        valid=True,
        quality=1.0,
        receive_timestamp_s=initial.receive_timestamp_s,
        timestamp_source="ur_controller",
        timestamp_reliable=True,
        timestamp_uncertainty_s=0.001,
    )
    marker = _desired_marker_observation().T_camera_marker.copy()
    clock_value = [1.0]
    reader = _EpochReader(initial)
    sender = _Sender()
    detector = _TimestampDetector(marker)
    runtime = VisualServoOrchestrator(
        camera_worker=_CameraWorker(frame),
        state_reader=reader,
        real_sender=sender,
        detector=detector,
        project_root=None,
        clock=lambda: clock_value[0],
        start_background_threads=False,
    )
    runtime.start_preview()
    runtime.connect_robot_read_only()
    runtime.install_calibration_for_testing(np.eye(4), (8, 6))
    assert runtime.system_check()
    runtime.start_dry_run()
    runtime.select_execution_mode("hardware", confirmed=True)
    detector.transform[0, 3] = marker_x
    clock_value[0] = 2.0
    return runtime, reader, sender, detector


def _run_hardware_enable_with_frames(
    runtime,
    reader,
    *,
    flange_x: float,
    start_frame_sequence: int = 8,
    start_frame_time_s: float = 2.02,
):
    errors = []
    reader.epoch_started.clear()

    def enable():
        try:
            runtime.enable()
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=enable)
    thread.start()
    assert reader.epoch_started.wait(1.0)
    robot_timestamps = np.arange(
        start_frame_time_s - 0.019,
        start_frame_time_s + 0.131,
        0.005,
    )
    for index, timestamp in enumerate(robot_timestamps, start=2):
        reader.publish(
            RobotStateSample(
                timestamp,
                _identity_at(flange_x),
                sequence_number=index,
                robot_mode="running",
                valid=True,
                quality=1.0,
                receive_timestamp_s=timestamp + 0.001,
                timestamp_source="ur_controller",
                timestamp_reliable=True,
                timestamp_uncertainty_s=0.001,
            )
        )
    for index in range(10):
        timestamp = start_frame_time_s + index * 0.01
        frame = FramePacket(
            timestamp,
            start_frame_sequence + index,
            np.zeros((6, 8), np.uint8),
            True,
            1.0,
            receive_timestamp_s=timestamp + 0.001,
            timestamp_source="camera_device",
            timestamp_reliable=True,
            timestamp_uncertainty_s=0.001,
        )
        runtime.camera_worker.latest_frame.value = frame
        runtime.process_vision_step(frame)
    thread.join(1.0)
    assert not thread.is_alive()
    return errors


def test_hardware_enable_resets_dry_epoch_and_relearns_from_ten_real_frames():
    runtime, reader, sender, detector = _hardware_epoch_runtime(
        flange_x=0.0, marker_x=0.0
    )
    runtime.force_stable_target_for_testing()
    assert runtime.safety.stable_frames >= 10
    assert runtime.ekf.initialized
    assert runtime.latest_prediction.get_latest() is not None
    assert runtime.latest_mpc.get_latest() is not None
    assert len(runtime._dry_pose_history) > 0

    errors = _run_hardware_enable_with_frames(runtime, reader, flange_x=0.0)

    assert errors == []
    assert runtime.state is VisualServoState.ENABLED
    assert runtime.safety.stable_frames == 10
    assert runtime._hardware_epoch_stable
    assert runtime._hardware_pre_authorization_sequence == 1
    assert runtime._hardware_pre_authorization_time_s == pytest.approx(1.001)
    assert len(runtime._dry_pose_history) == 0
    assert runtime._dry_robot_sample is None
    assert detector.reset_calls >= 1
    assert sender.authorized == 1
    runtime.start_following(start_threads=False)


def test_real_flange_nine_centimetres_from_goal_cannot_reuse_dry_stability():
    runtime, reader, sender, _ = _hardware_epoch_runtime(
        flange_x=0.09, marker_x=-0.09
    )
    runtime.force_stable_target_for_testing()
    runtime._hardware_sync_timeout_s = 0.1

    errors = _run_hardware_enable_with_frames(runtime, reader, flange_x=0.09)

    assert len(errors) == 1
    assert "稳定" in str(errors[0]) or "80" in str(errors[0])
    assert runtime.state is VisualServoState.FAULT
    assert sender.stopped >= 1
    assert not runtime._hardware_epoch_stable


@pytest.mark.parametrize(
    ("second_reliable", "endpoint_uncertainty_s"),
    ((False, 0.001), (True, 0.019)),
)
def test_hardware_sync_uses_interpolation_endpoint_reliability_not_latest_cache(
    second_reliable,
    endpoint_uncertainty_s,
):
    runtime, reader, _, _ = _hardware_epoch_runtime(flange_x=0.0, marker_x=0.0)
    runtime._reset_for_hardware_epoch()
    reader.confirm_zero_tcp(np.zeros(6))
    reader.begin_authorized_pose_epoch(2.0)
    runtime._hardware_authorization_time_s = 2.0
    runtime._hardware_sync_active = True

    def sample(timestamp, reliable, uncertainty):
        return RobotStateSample(
            timestamp,
            np.eye(4),
            valid=True,
            receive_timestamp_s=timestamp + 0.001,
            timestamp_source="ur_controller",
            timestamp_reliable=reliable,
            timestamp_uncertainty_s=uncertainty,
        )

    reader.pose_history.add(sample(2.01, True, endpoint_uncertainty_s))
    reader.pose_history.add(sample(2.03, second_reliable, endpoint_uncertainty_s))
    # 当前最新缓存故意保持可靠；门禁必须仍以图像时刻的两个
    # 实际插值端点为准，不得用“最新值”替代。
    reader.latest_state.publish(sample(2.04, True, 0.001))
    frame = FramePacket(
        2.02,
        8,
        np.zeros((6, 8), np.uint8),
        True,
        1.0,
        receive_timestamp_s=2.021,
        timestamp_source="camera_device",
        timestamp_reliable=True,
        timestamp_uncertainty_s=0.001,
    )
    runtime.camera_worker.latest_frame.value = frame

    assert runtime.process_vision_step(frame).valid
    assert runtime.latest_estimate.get_latest() is None
    assert runtime.safety.stable_frames == 0


def test_blocking_real_connect_cannot_let_dry_vision_repopulate_cleared_ekf():
    runtime, reader, _, _ = _hardware_epoch_runtime(flange_x=0.0, marker_x=0.0)

    class BlockingSender(_Sender):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def connect(self):
            self.entered.set()
            assert self.release.wait(1.0)
            super().connect()

    sender = BlockingSender()
    runtime.real_sender = sender
    enable_errors = []
    enable_thread = threading.Thread(
        target=lambda: _capture_error(runtime.enable, enable_errors)
    )
    enable_thread.start()
    assert sender.entered.wait(1.0)

    frame = FramePacket(
        2.01,
        8,
        np.zeros((6, 8), np.uint8),
        True,
        1.0,
        receive_timestamp_s=2.011,
        timestamp_source="camera_device",
        timestamp_reliable=True,
        timestamp_uncertainty_s=0.001,
    )
    vision_errors = []
    vision_thread = threading.Thread(
        target=lambda: _capture_error(
            lambda: runtime.process_vision_step(frame), vision_errors
        )
    )
    vision_thread.start()
    time.sleep(0.03)
    assert vision_thread.is_alive()
    assert runtime.latest_estimate.get_latest() is None
    assert runtime.latest_prediction.get_latest() is None
    assert runtime.latest_mpc.get_latest() is None
    assert runtime.latest_safety.get_latest() is None
    assert runtime._last_commanded_pose is None
    assert runtime.safety.stable_frames == 0
    assert runtime.mpc.solve_count == 0
    with pytest.raises(LookupError, match="no servo target"):
        runtime.interpolator.sample_transform(2.0)

    sender.release.set()
    assert reader.epoch_started.wait(1.0)
    for sequence, timestamp in enumerate((2.001, 2.02), start=2):
        reader.publish(
            RobotStateSample(
                timestamp,
                np.eye(4),
                sequence_number=sequence,
                valid=True,
                receive_timestamp_s=timestamp + 0.001,
                timestamp_source="ur_controller",
                timestamp_reliable=True,
                timestamp_uncertainty_s=0.001,
            )
        )
    vision_thread.join(1.0)
    runtime.emergency_stop("test release blocked connect")
    enable_thread.join(1.0)
    assert not vision_thread.is_alive()
    assert not enable_thread.is_alive()
    assert vision_errors == []


def test_stop_during_blocking_real_connect_prevents_late_tcp_authorization():
    runtime, reader, _, _ = _hardware_epoch_runtime(flange_x=0.0, marker_x=0.0)

    class BlockingSender(_Sender):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def connect(self):
            self.entered.set()
            assert self.release.wait(1.0)
            super().connect()

    sender = BlockingSender()
    runtime.real_sender = sender
    errors = []
    thread = threading.Thread(target=lambda: _capture_error(runtime.enable, errors))
    thread.start()
    assert sender.entered.wait(1.0)

    runtime.emergency_stop("operator stop while connect blocks")
    sender.release.set()
    thread.join(1.0)

    assert not thread.is_alive()
    assert errors
    assert sender.authorized == 0
    assert sender.stopped >= 1
    assert not reader.epoch_started.is_set()


def _install_blocking_production_sender(runtime, reader):
    """构造生产相同的 Deferred -> UrRtdeServoSender 两层连接链。"""

    factory_entered = threading.Event()
    factory_release = threading.Event()
    controls = []
    inners = []

    def blocking_control_factory(_host):
        factory_entered.set()
        factory_release.wait()
        control = FakeRobotBackend()
        controls.append(control)
        return control

    def inner_factory(host, **kwargs):
        inner = UrRtdeServoSender(
            host,
            control_factory=blocking_control_factory,
            **kwargs,
        )
        inners.append(inner)
        return inner

    deferred = DeferredUrRtdeServoSender(
        "192.168.0.10",
        reader,
        sender_factory=inner_factory,
    )
    runtime.real_sender = deferred
    return deferred, factory_entered, factory_release, controls, inners


def test_production_deferred_chain_stop_invalidates_pending_inner_immediately():
    runtime, reader, _, _ = _hardware_epoch_runtime(flange_x=0.0, marker_x=0.0)
    deferred, entered, release, controls, inners = _install_blocking_production_sender(
        runtime, reader
    )
    errors = []
    enable_thread = threading.Thread(
        target=lambda: _capture_error(runtime.enable, errors),
        daemon=True,
    )
    enable_thread.start()
    assert entered.wait(1.0)
    inner = inners[0]
    connecting_generation = inner._lifecycle_generation

    runtime.emergency_stop("operator stop while production factory blocks")

    assert inner._connection_state == "STOPPED"
    assert inner._lifecycle_generation > connecting_generation
    assert deferred.connection_state == "STOPPED"
    assert not reader.tcp_zero_confirmed
    release.set()
    enable_thread.join(1.0)
    assert not enable_thread.is_alive()
    assert errors
    assert not reader.epoch_started.is_set()
    assert controls[0].call_log == []
    assert controls[0].servo_commands == []
    assert controls[0].servo_stop_count == 1
    assert controls[0].stop_script_count == 1
    assert controls[0].disconnect_count == 1


def test_shutdown_reports_a_production_deferred_connect_that_never_returns():
    runtime, reader, _, _ = _hardware_epoch_runtime(flange_x=0.0, marker_x=0.0)
    deferred, entered, release, _, _ = _install_blocking_production_sender(
        runtime, reader
    )
    enable_errors = []
    enable_thread = threading.Thread(
        target=lambda: _capture_error(runtime.enable, enable_errors),
        daemon=True,
    )
    enable_thread.start()
    assert entered.wait(1.0)

    shutdown_errors = []
    shutdown_finished = threading.Event()

    def shutdown():
        _capture_error(runtime.shutdown, shutdown_errors)
        shutdown_finished.set()

    shutdown_thread = threading.Thread(target=shutdown, daemon=True)
    shutdown_thread.start()
    assert shutdown_finished.wait(0.5), "shutdown 被阻塞连接拖死，未报告 pending"
    assert len(shutdown_errors) == 1
    assert isinstance(shutdown_errors[0], RuntimeShutdownError)
    assert "连接仍在进行" in str(shutdown_errors[0])
    assert deferred.pending_connection_count == 1
    assert not runtime._shutdown_complete

    release.set()
    enable_thread.join(1.0)
    shutdown_thread.join(1.0)
    assert not enable_thread.is_alive()
    assert not shutdown_thread.is_alive()
    runtime.shutdown()
    assert runtime._shutdown_complete


def _capture_error(action, errors):
    try:
        action()
    except Exception as error:
        errors.append(error)


def test_hardware_selected_before_enable_keeps_using_simulated_pose_history():
    """选中实机不等于 TCP 已授信。

    后台视觉线程在真正使能前仍必须走独立仿真位姿路径，不能读取未授信的
    实机位姿历史。
    """

    runtime, _ = _runtime()
    runtime.detector = _Detector(_desired_marker_observation())
    runtime.start_preview()
    runtime.connect_robot_read_only()
    runtime.install_calibration_for_testing(np.eye(4), (8, 6))
    assert runtime.system_check()
    runtime.start_dry_run()
    runtime.select_execution_mode("hardware", confirmed=True)
    from visual_servo_system.geometry.robot_pose_history import RobotPoseHistory

    runtime.state_reader.pose_history = RobotPoseHistory(8, 0.1)

    observation = _desired_marker_observation()
    observation = MarkerPoseObservation(
        1.1,
        observation.marker_id,
        observation.T_camera_marker,
        observation.reprojection_error_px,
        observation.decision_margin,
        observation.pixel_area_px2,
        observation.valid,
        observation.quality,
    )
    runtime.detector = _Detector(observation)
    frame = FramePacket(
        1.1,
        8,
        np.zeros((6, 8), np.uint8),
        True,
        1.0,
        receive_timestamp_s=1.101,
        timestamp_source="camera_device",
        timestamp_reliable=True,
        timestamp_uncertainty_s=0.001,
    )

    assert runtime.process_vision_step(frame).valid
    assert runtime.state is VisualServoState.VALIDATED
    assert not runtime.state_reader.tcp_zero_confirmed


def test_hardware_enable_rejects_unreliable_camera_or_robot_time():
    runtime, _ = _runtime()
    frame, _, sample = _objects()
    runtime.camera_worker.latest_frame.value = FramePacket(
        frame.host_timestamp_s,
        frame.sequence_number,
        frame.image,
        True,
        1.0,
    )
    runtime.state_reader.latest_state.value = RobotStateSample(
        sample.host_timestamp_s,
        sample.T_base_flange,
        valid=True,
    )
    runtime.start_preview()
    runtime.connect_robot_read_only()
    runtime.install_calibration_for_testing(np.eye(4), (8, 6))
    assert runtime.system_check()
    runtime.detector = _Detector(_desired_marker_observation())
    runtime.start_dry_run()
    with pytest.raises(RuntimeError, match="时间"):
        runtime.select_execution_mode("hardware", confirmed=True)


def test_wait_for_robot_sample_rejects_every_pre_authorization_cache_entry():
    runtime, _ = _runtime()

    stale = RobotStateSample(
        1.9,
        np.eye(4),
        valid=True,
        receive_timestamp_s=1.95,
        timestamp_source="ur_controller",
        timestamp_reliable=True,
        timestamp_uncertainty_s=0.001,
    )
    fresh = RobotStateSample(
        2.1,
        _identity_at(0.02),
        valid=True,
        receive_timestamp_s=2.11,
        timestamp_source="ur_controller",
        timestamp_reliable=True,
        timestamp_uncertainty_s=0.001,
    )

    class SequencedBuffer:
        def __init__(self):
            self.items = [(5, stale), (6, stale), (7, fresh)]

        def get_latest(self):
            return self.items[0]

        def wait_for_newer(self, sequence, _timeout_s):
            return next((item for item in self.items if item[0] > sequence), None)

    runtime.state_reader.latest_state = SequencedBuffer()
    result = runtime._wait_for_valid_robot_sample(
        baseline_sequence=5,
        not_before_time_s=2.0,
        timeout_s=0.1,
    )
    assert result is fresh


def test_wait_for_robot_sample_times_out_when_link_only_has_old_cache():
    runtime, _ = _runtime()
    old = runtime.state_reader.latest_state.value

    class LostLinkBuffer:
        def get_latest(self):
            return 9, old

        def wait_for_newer(self, _sequence, _timeout_s):
            return None

    runtime.state_reader.latest_state = LostLinkBuffer()
    with pytest.raises(RuntimeError, match="授信后"):
        runtime._wait_for_valid_robot_sample(
            baseline_sequence=9,
            not_before_time_s=2.0,
            timeout_s=0.01,
        )


def test_shutdown_is_idempotent_and_always_stops_both_links():
    runtime, sender = _runtime()
    runtime.start_preview()
    runtime.connect_robot_read_only()
    runtime.shutdown()
    runtime.shutdown()
    assert sender.stopped >= 1
    assert sender.disconnected >= 1
    assert runtime.camera_worker.stopped >= 1
    assert runtime.state in {VisualServoState.STOPPED, VisualServoState.FAULT}


def test_shutdown_failure_is_reported_and_remaining_resources_can_retry():
    runtime, sender = _runtime()
    runtime.start_preview()
    runtime.connect_robot_read_only()
    original_stop = runtime.camera_worker.stop
    calls = 0

    def fail_once(timeout_s=2.0):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("camera still blocked")
        original_stop(timeout_s)

    runtime.camera_worker.stop = fail_once
    with pytest.raises(RuntimeShutdownError, match="camera still blocked"):
        runtime.shutdown()
    assert not runtime._shutdown_complete
    assert sender.disconnected >= 1
    runtime.shutdown()
    assert runtime._shutdown_complete
    assert calls == 2


def test_emergency_stop_propagates_unconfirmed_sender_and_never_claims_stopped():
    runtime, _ = _runtime()

    class FailingStopSender(_Sender):
        def stop(self):
            self.stopped += 1
            raise RuntimeError("waitPeriod 仍阻塞")

    sender = FailingStopSender()
    runtime.real_sender = sender

    with pytest.raises(RuntimeStopUnconfirmedError, match="waitPeriod"):
        runtime.emergency_stop("operator stop")

    message = runtime.ui_snapshot()["message"]
    assert "停车未确认" in message
    assert "物理急停" in message
    assert not message.startswith("已停止")
    assert runtime._dry_sender.stopped


def _desired_marker_observation():
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[2, 3] = 0.5
    return MarkerPoseObservation(1.0, 0, transform, 0.1, None, 1200.0, True, 1.0)


def _armed_following_runtime():
    frame, _, sample = _objects()
    sender = _Sender()
    runtime = VisualServoOrchestrator(
        camera_worker=_CameraWorker(frame),
        state_reader=_Reader(sample),
        real_sender=sender,
        detector=_Detector(_desired_marker_observation()),
        project_root=None,
        clock=lambda: 1.0,
        start_background_threads=False,
    )
    runtime.start_preview()
    runtime.connect_robot_read_only()
    runtime.install_calibration_for_testing(np.eye(4), (8, 6))
    assert runtime.system_check()
    runtime.start_dry_run()
    runtime.select_speed_profile("commissioning")
    runtime.force_stable_target_for_testing()
    runtime.enable()
    # 干运行已执行一次视觉、MPC、安全门、插值和发送全链。
    runtime.start_following(start_threads=False)
    return runtime, sender


def test_full_synchronous_chain_reaches_guarded_servo_interpolator():
    runtime, real_sender = _armed_following_runtime()
    sent_before = len(runtime._dry_sender.sent_poses)
    decision = runtime.process_control_step(1.0)
    assert decision.allow_motion
    assert runtime.latest_estimate.get_latest() is not None
    assert runtime.latest_prediction.get_latest() is not None
    assert runtime.latest_mpc.get_latest()[1].valid
    snapshot = runtime.ui_snapshot()
    assert snapshot["ekf_status"] == "有效"
    assert snapshot["mpc_status"] == "solved"
    assert snapshot["robot_status"] == "running / 有效"
    assert snapshot["safety_status"] == "FOLLOWING / 允许运动"
    assert runtime.process_servo_step(1.0)
    assert len(runtime._dry_sender.sent_poses) == sent_before + 1
    assert real_sender.sent == []


def test_mpc_failure_latches_fault_and_stops_all_senders():
    runtime, sender = _armed_following_runtime()

    class FailedMpc:
        config = type("Config", (), {"dt_s": 0.05})()

        def solve(self, *_args, **_kwargs):
            return MpcResult(1.0, np.zeros(6), 0.001, "infeasible", False, 0.0)

    runtime.mpc = FailedMpc()
    decision = runtime.process_control_step(1.0)
    assert not decision.allow_motion
    assert runtime.state is VisualServoState.FAULT
    assert sender.stopped >= 1


def test_expired_guard_decision_faults_and_stops_instead_of_silent_drop():
    runtime, sender = _armed_following_runtime()
    decision = runtime.process_control_step(1.0)
    assert decision.allow_motion
    assert not runtime.process_servo_step(1.2)
    assert runtime.state is VisualServoState.FAULT
    assert sender.stopped >= 1
    assert runtime._dry_sender.stopped


def test_following_without_any_new_safety_decision_faults_after_deadline():
    runtime, sender = _armed_following_runtime()
    # 丢弃干运行遗留决定，并模拟控制线程尚未产生本次跟随的新决定。
    latest = runtime.latest_safety.get_latest()
    runtime._minimum_safety_sequence = 0 if latest is None else latest[0]
    assert not runtime.process_servo_step(1.2)
    assert runtime.state is VisualServoState.FAULT
    assert sender.stopped >= 1


def test_stop_acknowledge_recheck_and_hardware_enable_reconnects_sender():
    runtime, reader, sender, _ = _hardware_epoch_runtime(
        flange_x=0.0, marker_x=0.0
    )
    assert _run_hardware_enable_with_frames(runtime, reader, flange_x=0.0) == []
    first_authorizations = sender.authorized
    runtime.emergency_stop("cycle one")
    assert not runtime._hardware_epoch_stable
    runtime.acknowledge_stop()
    assert runtime.system_check()
    # 重新检查仍不允许触碰控制侧；只有第二次实机使能才再次授权。
    assert sender.authorized == first_authorizations
    runtime.start_dry_run()
    runtime.select_execution_mode("hardware", confirmed=True)
    assert _run_hardware_enable_with_frames(
        runtime,
        reader,
        flange_x=0.0,
        start_frame_sequence=18,
        start_frame_time_s=2.30,
    ) == []
    assert sender.authorized == first_authorizations + 1


def test_invalid_observation_resets_stability_between_valid_frames():
    runtime, _ = _runtime()
    runtime.start_preview()
    runtime.connect_robot_read_only()
    runtime.install_calibration_for_testing(np.eye(4), (8, 6))
    assert runtime.system_check()
    runtime.force_stable_target_for_testing()
    assert runtime.safety.stable_frames == runtime.safety.config.stable_frames_required
    invalid = MarkerPoseObservation(1.1, 0, np.eye(4), 0.0, None, 0.0, False, 0.0)
    runtime.detector = _Detector(invalid)
    frame = FramePacket(
        1.1,
        8,
        np.zeros((6, 8), np.uint8),
        True,
        1.0,
        receive_timestamp_s=1.101,
        timestamp_source="camera_device",
        timestamp_reliable=True,
        timestamp_uncertainty_s=0.001,
    )
    runtime.process_vision_step(frame)
    assert runtime.safety.stable_frames == 0
