"""视觉伺服系统的生产调用链和生命周期管理。

本文件只负责“把已测试的单功能模块串起来”。硬件适配、位姿估计、MPC、
安全门和日志仍由各自的源文件实现。默认构造不连接任何硬件，也不发送任何运动。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from visual_servo_system.config.config_loader import (
    CalibrationError,
    SystemConfig,
    load_camera_intrinsics,
    load_hand_eye_calibration,
    load_system_config,
)
from visual_servo_system.control.mpc_pose_controller import (
    MpcPoseController,
    MpcPoseControllerConfig,
)
from visual_servo_system.control.pose_target_builder import (
    build_desired_flange_pose,
    build_desired_flange_trajectory,
)
from visual_servo_system.control.servo_target_interpolator import ServoTargetInterpolator
from visual_servo_system.estimation.target_pose_ekf import EkfTimestampGapError, TargetPoseEkf
from visual_servo_system.estimation.target_trajectory_predictor import TargetTrajectoryPredictor
from visual_servo_system.geometry.pose_transforms import (
    compose_transforms,
    so3_exp,
    so3_log,
    ur_pose_to_transform,
    validate_transform,
)
from visual_servo_system.geometry.robot_pose_history import RobotPoseHistory
from visual_servo_system.models.data_types import (
    FramePacket,
    MarkerPoseObservation,
    MpcResult,
    PredictedTrajectory,
    RobotStateSample,
    SafetyDecision,
    TargetStateEstimate,
)
from visual_servo_system.robot.ur_rtde_servo_sender import FakeRobotBackend
from visual_servo_system.runtime.latest_value_buffer import LatestValueBuffer
from visual_servo_system.safety.visual_servo_safety_monitor import (
    SafetyInputs,
    SafetyMonitorConfig,
    VisualServoSafetyMonitor,
    VisualServoState,
)
from visual_servo_system.vision.marker_pose_detector import (
    MarkerPoseDetector,
    MarkerPoseDetectorConfig,
)

CAMERA_MODEL = "Point Grey Research Blackfly S BFS-U3-13Y3C"
CAMERA_SERIAL = "16276663"
MAX_HARDWARE_TIMESTAMP_UNCERTAINTY_S = 0.020


class RuntimeMode(Enum):
    PREVIEW = "预览"
    DRY_RUN = "仿真/干运行"
    HARDWARE = "实机"


class RuntimeShutdownError(RuntimeError):
    """一个或多个资源未能关闭；再次调用 ``shutdown`` 会继续重试。"""

    def __init__(self, failures: list[str]) -> None:
        self.failures = tuple(failures)
        super().__init__("; ".join(failures))


class RuntimeStopUnconfirmedError(RuntimeError):
    """一个或多个控制后端未能确认停车，必须使用物理急停。"""

    def __init__(self, failures: list[str]) -> None:
        self.failures = tuple(failures)
        super().__init__("；".join(failures))


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    recognition_rate_hz: float = 0.0
    latency_s: float = 0.0
    last_message: str = "等待连接"


class _DryRunSender:
    """只记录命令的内存后端，永远不能创建真实 RTDE 连接。"""

    def __init__(self) -> None:
        self.backend = FakeRobotBackend()
        self.sent_poses: list[np.ndarray] = []
        self.connected = False
        self.stopped = True

    def connect(self) -> None:
        self.connected = True
        self.stopped = False

    def authorize_state_reader(self, reader: Any) -> None:
        if not self.connected:
            raise RuntimeError("dry-run sender is not connected")
        reader.confirm_zero_tcp(np.zeros(6, dtype=np.float64))

    def send_pose(self, pose_ur: np.ndarray) -> None:
        if not self.connected or self.stopped:
            raise RuntimeError("dry-run sender is stopped")
        self.sent_poses.append(np.array(pose_ur, dtype=np.float64, copy=True))

    def stop(self) -> None:
        self.stopped = True

    def disconnect(self) -> None:
        self.stopped = True
        self.connected = False


class VisualServoOrchestrator:
    """多线程运行器；同步 ``step`` 方法也可用于回放和测试。"""

    def __init__(
        self,
        *,
        camera_worker: Any,
        state_reader: Any,
        real_sender: Any,
        detector: Any | None = None,
        project_root: str | Path | None,
        clock: Callable[[], float] = time.monotonic,
        logger: Any | None = None,
        video_recorder: Any | None = None,
        ekf: Any | None = None,
        predictor: Any | None = None,
        mpc: Any | None = None,
        safety: VisualServoSafetyMonitor | None = None,
        interpolator: Any | None = None,
        start_background_threads: bool = True,
    ) -> None:
        self.camera_worker = camera_worker
        self.state_reader = state_reader
        self.real_sender = real_sender
        self.detector = detector
        self.project_root = None if project_root is None else Path(project_root)
        self._clock = clock
        self.logger = logger
        self.video_recorder = video_recorder
        # 同步回放/单元测试可关闭后台线程；生产工厂保留默认值 True。
        self._start_background_threads = bool(start_background_threads)
        self.mode = RuntimeMode.PREVIEW
        self.speed_profile = "commissioning"
        self.safety = safety or VisualServoSafetyMonitor(clock=clock)
        self.ekf = ekf or TargetPoseEkf()
        self.predictor = predictor or TargetTrajectoryPredictor()
        self.mpc = mpc or MpcPoseController(clock=time.perf_counter)
        self.interpolator = interpolator or ServoTargetInterpolator(max_extrapolation_s=0.05)
        self.latest_observation: LatestValueBuffer[MarkerPoseObservation] = LatestValueBuffer()
        self.latest_estimate: LatestValueBuffer[TargetStateEstimate] = LatestValueBuffer()
        self.latest_prediction: LatestValueBuffer[PredictedTrajectory] = LatestValueBuffer()
        self.latest_mpc: LatestValueBuffer[MpcResult] = LatestValueBuffer()
        self.latest_safety: LatestValueBuffer[SafetyDecision] = LatestValueBuffer()
        self.latest_annotated_frame: LatestValueBuffer[FramePacket] = LatestValueBuffer()
        self._T_flange_camera: np.ndarray | None = None
        self._T_camera_marker_desired = ur_pose_to_transform(
            np.array([0.0, 0.0, 0.5, np.pi, 0.0, 0.0])
        )
        self._calibration_resolution: tuple[int, int] | None = None
        self._system_config: SystemConfig | None = None
        self._robot_read_only_connected = False
        self._dry_run_completed = False
        self._hardware_confirmed = False
        self._active_sender: Any | None = None
        self._dry_sender = _DryRunSender()
        self._dry_pose_history = RobotPoseHistory(64, 0.25)
        self._dry_robot_sample: RobotStateSample | None = None
        self._last_commanded_pose: np.ndarray | None = None
        self._control_sequence = 0
        self._last_processed_frame_sequence = -1
        self._last_frame_arrival_s: float | None = None
        self._recognition_times: list[float] = []
        self._metrics = RuntimeMetrics()
        self._stop_event = threading.Event()
        self._stop_event.set()
        self._shutdown_event = threading.Event()
        self._vision_thread: threading.Thread | None = None
        self._control_threads: list[threading.Thread] = []
        self._lifecycle_lock = threading.RLock()
        # 关闭流程不能等待可能卡在厂商 connect 的 enable 生命周期锁；使用独立锁
        # 才能先停车，并把仍未返回的 pending 控制连接准确报告给界面。
        self._shutdown_lock = threading.Lock()
        self._stop_generation_lock = threading.Lock()
        self._stop_generation = 0
        self._shutdown_complete = False
        self._shutdown_in_progress = False
        self._shutdown_failures: tuple[str, ...] = ()
        self._following_started_s: float | None = None
        self._minimum_safety_sequence = 0
        self._pipeline_lock = threading.RLock()
        self._hardware_sync_condition = threading.Condition()
        self._hardware_sync_active = False
        self._hardware_epoch_stable = False
        self._hardware_authorization_time_s: float | None = None
        self._hardware_pre_authorization_sequence = 0
        self._hardware_pre_authorization_time_s: float | None = None
        self._hardware_frame_baseline_sequence = -1
        self._hardware_sync_timeout_s = 2.0

    @property
    def state(self) -> VisualServoState:
        return self.safety.state

    def _set_message(self, message: str, *, latency_s: float | None = None) -> None:
        self._metrics = RuntimeMetrics(
            self._metrics.recognition_rate_hz,
            self._metrics.latency_s if latency_s is None else latency_s,
            message,
        )

    def start_preview(self) -> None:
        with self._lifecycle_lock:
            if self.state is not VisualServoState.DISCONNECTED:
                raise RuntimeError("只能从未连接状态开始预览")
            self.camera_worker.start()
            self.safety.connect_preview()
            self.mode = RuntimeMode.PREVIEW
            self._set_message("相机预览已启动；机器人不会运动")

    def connect_robot_read_only(self) -> None:
        with self._lifecycle_lock:
            if self.state is not VisualServoState.PREVIEW:
                raise RuntimeError("必须先连接相机预览")
            self.state_reader.start()
            self._robot_read_only_connected = True
            self._set_message("UR5 只读状态已连接")

    def install_calibration_for_testing(
        self, T_flange_camera: np.ndarray, image_size: tuple[int, int]
    ) -> None:
        """仅供假设备/回放测试注入已验证标定，实机入口不调用。"""

        self._T_flange_camera = validate_transform(T_flange_camera)
        self._calibration_resolution = tuple(image_size)

    def _load_calibration(self) -> None:
        if self._T_flange_camera is not None and self._calibration_resolution is not None:
            return
        if self.project_root is None:
            raise CalibrationError("未指定工程目录，无法加载标定")
        config = load_system_config(self.project_root / "config_files")
        intrinsics = load_camera_intrinsics(
            self.project_root / "calibration_files" / "camera_intrinsics.yaml",
            expected_resolution=(
                config.camera.image_width_px,
                config.camera.image_height_px,
            ),
        )
        hand_eye = load_hand_eye_calibration(
            self.project_root / "calibration_files" / "hand_eye_eye_in_hand.yaml"
        )
        self._system_config = config
        self._T_flange_camera = hand_eye.T_flange_camera
        self._calibration_resolution = intrinsics.image_size_px
        self._T_camera_marker_desired = config.target.T_camera_target_desired
        self.safety = VisualServoSafetyMonitor(
            SafetyMonitorConfig.from_settings(config.safety), clock=self._clock
        )
        self.safety.connect_preview()
        self.mpc = MpcPoseController(
            MpcPoseControllerConfig.from_settings(config.mpc, config.safety)
        )
        if self.detector is None:
            marker = config.marker
            self.detector = MarkerPoseDetector(
                MarkerPoseDetectorConfig(
                    marker.detector,
                    marker.family,
                    marker.aruco_dictionary,
                    marker.marker_id,
                    marker.marker_size_m,
                    detection_scale=marker.detection_scale,
                ),
                intrinsics.camera_matrix,
                intrinsics.distortion_coefficients,
            )

    def _latest_frame_resolution_matches(self) -> bool:
        latest = self.camera_worker.latest_frame.get_latest()
        if latest is None or self._calibration_resolution is None:
            return False
        frame = latest[1]
        height, width = frame.image.shape[:2]
        return (width, height) == self._calibration_resolution

    def _reset_estimation_after_preview_gap(self) -> None:
        """未进入运动时丢弃过时估计，保留视觉线程继续重获。

        已使能/跟随状态不会调用该函数；这些状态的间隔超限仍由线程
        异常路径立即停车。
        """

        if self.state is not VisualServoState.VALIDATED:
            raise RuntimeError("EKF gap recovery is only allowed before motion")
        self.ekf.reset()
        self.latest_estimate.clear()
        self.latest_prediction.clear()
        self.latest_mpc.clear()
        self.latest_safety.clear()
        self.interpolator.reset()
        self._last_commanded_pose = None
        self.safety.reset_target_stability()
        self._set_message("EKF 图像间隔超限，已在未运动状态重新初始化")

    def _wait_for_valid_robot_sample(
        self,
        *,
        baseline_sequence: int,
        not_before_time_s: float,
        timeout_s: float = 0.5,
    ) -> RobotStateSample:
        """TCP 授信后只接受新代次、新时间且可靠的法兰样本。"""

        if type(baseline_sequence) is not int or baseline_sequence < 0:
            raise ValueError("baseline_sequence must be a non-negative integer")
        not_before = float(not_before_time_s)
        timeout = float(timeout_s)
        if not np.isfinite(not_before) or not_before < 0.0:
            raise ValueError("not_before_time_s must be finite and non-negative")
        if not np.isfinite(timeout) or timeout < 0.0:
            raise ValueError("timeout_s must be finite and non-negative")
        deadline = time.monotonic() + timeout
        cursor = baseline_sequence
        while True:
            latest = self.state_reader.latest_state.get_latest()
            if latest is not None and latest[0] > cursor:
                candidate = latest
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                candidate = self.state_reader.latest_state.wait_for_newer(cursor, remaining)
                if candidate is None:
                    break
            sequence, sample = candidate
            cursor = max(cursor, sequence)
            if (
                sequence > baseline_sequence
                and sample.valid
                and sample.timestamp_reliable
                and sample.host_timestamp_s >= not_before
                and sample.receive_timestamp_s >= not_before
            ):
                return sample
        raise RuntimeError("TCP 授信后未在时限内收到新的可靠法兰位姿")

    def system_check(self) -> bool:
        with self._lifecycle_lock:
            if self.state is not VisualServoState.PREVIEW:
                raise RuntimeError("系统检查只能在预览状态执行")
            if not self._robot_read_only_connected:
                raise RuntimeError("必须先连接 UR5 只读状态")
            try:
                self._load_calibration()
            except Exception as error:
                self._set_message(f"标定校验失败，仅允许预览：{error}")
                return False
            resolution_matches = self._latest_frame_resolution_matches()
            robot_link_ok = bool(
                getattr(self.state_reader, "connected", True)
                and getattr(self.state_reader, "communication_healthy", True)
            )
            # 系统检查只读配置、标定、图像和 RTDEReceive 健康状态。真实控制接口和
            # TCP=0 授权严格延迟到“实机使能”，保证预览/干运行完全隔离实机发送侧。
            passed = self.safety.system_check(
                calibration_valid=True,
                resolution_matches=resolution_matches,
                tcp_zero_verified=True,
                robot_state_valid=robot_link_ok,
            )
            if passed and self._start_background_threads:
                self._start_vision_thread_locked()
            self._set_message("系统检查通过" if passed else "系统检查未通过")
            return passed

    def start_dry_run(self) -> None:
        with self._lifecycle_lock:
            if self.state is not VisualServoState.VALIDATED:
                raise RuntimeError("干运行必须在系统检查通过后执行")
            self.mode = RuntimeMode.DRY_RUN
            self._dry_run_completed = False
            self._hardware_confirmed = False
            self._run_dry_validation_once()
            if self.state is not VisualServoState.VALIDATED:
                raise RuntimeError("干运行期间收到停车/故障请求")
            self._dry_run_completed = True
            self._set_message("干运行完成：完整控制链只写入独立假机器人")

    def _dry_sample_at(self, timestamp_s: float) -> RobotStateSample:
        """生成完全独立于 RTDEReceive/TCP 授权的仿真法兰样本。"""

        sample = RobotStateSample(
            float(timestamp_s),
            np.eye(4, dtype=np.float64),
            sequence_number=(
                0 if self._dry_robot_sample is None else self._dry_robot_sample.sequence_number + 1
            ),
            robot_mode="dry_run_simulation",
            valid=True,
            quality=1.0,
            receive_timestamp_s=float(timestamp_s),
            timestamp_source="simulation",
            timestamp_reliable=True,
            timestamp_uncertainty_s=0.0,
        )
        if (
            self._dry_robot_sample is None
            or timestamp_s > self._dry_robot_sample.host_timestamp_s
        ):
            self._dry_pose_history.add(sample)
            self._dry_robot_sample = sample
        return self._dry_robot_sample

    def _run_dry_validation_once(self) -> None:
        """用独立安全门和假发送器实际执行一次完整控制/servo 调用链。"""

        latest_frame = self.camera_worker.latest_frame.get_latest()
        if latest_frame is None:
            raise RuntimeError("干运行需要至少一帧相机图像")
        frame = latest_frame[1]
        if frame.sequence_number > self._last_processed_frame_sequence:
            observation = self.process_vision_step(frame)
            if not observation.valid:
                raise RuntimeError("干运行需要有效的固定 ID 标签观测")
        estimate_item = self.latest_estimate.get_latest()
        if estimate_item is None or not estimate_item[1].valid:
            raise RuntimeError("干运行未产生有效 EKF 状态")

        now = max(float(self._clock()), estimate_item[1].host_timestamp_s)
        sample = self._dry_sample_at(now)
        dry_monitor = VisualServoSafetyMonitor(self.safety.config, clock=self._clock)
        dry_monitor.connect_preview()
        if not dry_monitor.system_check(
            calibration_valid=True,
            resolution_matches=True,
            tcp_zero_verified=True,
            robot_state_valid=True,
        ):
            raise RuntimeError("干运行安全检查失败")
        assert self._T_flange_camera is not None
        desired = build_desired_flange_pose(
            estimate_item[1].T_base_target,
            self._T_flange_camera,
            self._T_camera_marker_desired,
        )
        position_error = float(np.linalg.norm(desired[:3, 3] - sample.T_base_flange[:3, 3]))
        rotation_error = float(
            np.linalg.norm(so3_log(sample.T_base_flange[:3, :3].T @ desired[:3, :3]))
        )
        required = dry_monitor.config.stable_frames_required
        first_timestamp = max(0.0, now - (required - 1) * 0.001)
        for index in range(required):
            dry_monitor.observe_target(
                True,
                position_error,
                rotation_error,
                frame_sequence=index,
                observation_host_timestamp_s=first_timestamp + index * 0.001,
            )
        dry_monitor.enable(sample.T_base_flange, speed_mode=self.speed_profile)
        dry_monitor.start_following()

        sent_before = len(self._dry_sender.sent_poses)
        try:
            self._dry_sender.connect()
            self.interpolator.reset()
            self._control_sequence = 0
            self.interpolator.push_target(now, sample.T_base_flange, 0)
            self._last_commanded_pose = np.array(sample.T_base_flange, copy=True)
            decision = self.process_control_step(now, monitor=dry_monitor)
            if not decision.allow_motion or not self.process_servo_step(
                now, monitor=dry_monitor, sender=self._dry_sender
            ):
                raise RuntimeError("干运行安全门拒绝完整链")
            if len(self._dry_sender.sent_poses) <= sent_before:
                raise RuntimeError("干运行未到达假 servo 发送器")
        finally:
            self._dry_sender.disconnect()
            self.interpolator.reset()

    def select_execution_mode(self, mode: str, *, confirmed: bool = False) -> None:
        with self._lifecycle_lock:
            if not self._dry_run_completed:
                raise RuntimeError("必须先完成干运行")
            if mode == "dry_run":
                self.mode = RuntimeMode.DRY_RUN
                self._hardware_confirmed = False
                self._hardware_sync_active = False
                self._hardware_epoch_stable = False
            elif mode == "hardware":
                if confirmed is not True:
                    raise RuntimeError("切换实机必须人工明确确认")
                frame_item = self.camera_worker.latest_frame.get_latest()
                robot_item = self.state_reader.latest_state.get_latest()
                if frame_item is None or not frame_item[1].timestamp_reliable:
                    raise RuntimeError("相机采样时间不可靠，只允许预览/干运行")
                if robot_item is None or not robot_item[1].timestamp_reliable:
                    raise RuntimeError("机器人采样时间不可靠，只允许预览/干运行")
                self._dry_sender.disconnect()
                if self._active_sender is self._dry_sender:
                    self._active_sender = None
                self.mode = RuntimeMode.HARDWARE
                self._hardware_confirmed = True
                self._hardware_sync_active = False
                self._hardware_epoch_stable = False
            else:
                raise ValueError("mode must be 'dry_run' or 'hardware'")

    def select_speed_profile(self, profile: str) -> None:
        if not self._dry_run_completed:
            raise RuntimeError("必须先完成干运行再选择限速档")
        if profile not in {"commissioning", "experiment"}:
            raise ValueError("profile must be commissioning or experiment")
        self.speed_profile = profile
        self._set_message("联调档 50 mm/s" if profile == "commissioning" else "试验档 120 mm/s")

    def force_stable_target_for_testing(self) -> None:
        """仅供单元测试建立十个唯一观测。"""

        latest = self.latest_observation.get_latest()
        base_sequence = max(0, self._last_processed_frame_sequence + 1)
        base_timestamp = (
            1.0 if latest is None else latest[1].host_timestamp_s + 0.01
        )
        for index in range(self.safety.config.stable_frames_required):
            self.safety.observe_target(
                True,
                0.0,
                0.0,
                frame_sequence=base_sequence + index,
                observation_host_timestamp_s=base_timestamp + index * 0.01,
            )

    def _latest_robot_sample(self) -> RobotStateSample:
        if self.mode is RuntimeMode.DRY_RUN:
            if self._dry_robot_sample is None:
                return self._dry_sample_at(float(self._clock()))
            return self._dry_robot_sample
        latest = self.state_reader.latest_state.get_latest()
        if latest is None:
            raise RuntimeError("尚未收到 UR5 状态")
        return latest[1]

    def _reset_for_hardware_epoch(self) -> None:
        """原子丢弃干运行的估计、控制、稳定度和位姿历史。"""

        with self._pipeline_lock:
            self._dry_sender.disconnect()
            self._dry_pose_history.clear()
            self._dry_robot_sample = None
            self.ekf.reset()
            reset_mpc = getattr(self.mpc, "reset", None)
            if callable(reset_mpc):
                reset_mpc()
            if self.detector is not None:
                self.detector.reset_tracking()
            self.latest_observation.clear()
            self.latest_estimate.clear()
            self.latest_prediction.clear()
            self.latest_mpc.clear()
            self.latest_safety.clear()
            self.interpolator.reset()
            self._last_commanded_pose = None
            self._control_sequence = 0
            self._minimum_safety_sequence = 0
            self._hardware_epoch_stable = False
            self._hardware_sync_active = False
            self._hardware_authorization_time_s = None
            latest_frame = self.camera_worker.latest_frame.get_latest()
            self._hardware_frame_baseline_sequence = max(
                self._last_processed_frame_sequence,
                -1 if latest_frame is None else latest_frame[1].sequence_number,
            )
            self.safety.reset_target_stability()

    def _wait_for_hardware_stability(self, operation_generation: int) -> None:
        """等待授权后 10 个唯一、可靠且可时间插值的实机视觉帧。"""

        deadline = time.monotonic() + self._hardware_sync_timeout_s
        required = self.safety.config.stable_frames_required
        with self._hardware_sync_condition:
            while self.safety.stable_frames < required:
                with self._stop_generation_lock:
                    if operation_generation != self._stop_generation:
                        raise RuntimeError("实机同步期间收到停车请求")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        "实机同步未获得连续稳定目标：需要10帧且误差不超过80 mm/10 deg"
                    )
                self._hardware_sync_condition.wait(min(remaining, 0.05))

    def _authorize_new_hardware_epoch(
        self,
        sender: Any,
        robot_item: tuple[int, RobotStateSample],
        operation_generation: int,
    ) -> tuple[int, float]:
        """在视觉管线锁内清旧代次并完成 TCP 授权切换。"""

        with self._pipeline_lock:
            baseline_sequence, baseline_sample = robot_item
            self._hardware_pre_authorization_sequence = baseline_sequence
            self._hardware_pre_authorization_time_s = max(
                baseline_sample.host_timestamp_s,
                baseline_sample.receive_timestamp_s,
            )
            self._reset_for_hardware_epoch()
            sender.connect()
            with self._stop_generation_lock:
                if operation_generation != self._stop_generation:
                    raise RuntimeError("实机连接期间收到停车请求，禁止迟到授权")
            sender.authorize_state_reader(self.state_reader)
            with self._stop_generation_lock:
                if operation_generation != self._stop_generation:
                    raise RuntimeError("实机授权期间收到停车请求")
            authorization_time = float(self._clock())
            self._hardware_authorization_time_s = authorization_time
            begin_epoch = getattr(
                self.state_reader, "begin_authorized_pose_epoch", None
            )
            if callable(begin_epoch):
                begin_epoch(authorization_time)
            else:
                self.state_reader.pose_history.clear()
            self._hardware_sync_active = True
            return baseline_sequence, authorization_time

    def enable(self) -> None:
        with self._lifecycle_lock:
            if self.state is not VisualServoState.VALIDATED:
                raise RuntimeError("使能前必须重新通过系统检查")
            if not self._dry_run_completed:
                raise RuntimeError("使能前必须先完成干运行")
            if self.mode is RuntimeMode.HARDWARE and not self._hardware_confirmed:
                raise RuntimeError("实机模式未人工确认")
            sender = self.real_sender if self.mode is RuntimeMode.HARDWARE else self._dry_sender
            with self._stop_generation_lock:
                operation_generation = self._stop_generation
            try:
                if self.mode is RuntimeMode.HARDWARE:
                    frame_item = self.camera_worker.latest_frame.get_latest()
                    robot_item = self.state_reader.latest_state.get_latest()
                    if frame_item is None or not frame_item[1].timestamp_reliable:
                        raise RuntimeError("相机采样时间不可靠，禁止实机使能")
                    if robot_item is None or not robot_item[1].timestamp_reliable:
                        raise RuntimeError("机器人采样时间不可靠，禁止实机使能")
                    baseline_sequence, authorization_time = (
                        self._authorize_new_hardware_epoch(
                            sender, robot_item, operation_generation
                        )
                    )
                    sample = self._wait_for_valid_robot_sample(
                        baseline_sequence=baseline_sequence,
                        not_before_time_s=authorization_time,
                    )
                    if not getattr(self.state_reader, "tcp_zero_confirmed", False):
                        raise RuntimeError("活动 TCP 尚未被控制侧确认为零")
                    self._wait_for_hardware_stability(operation_generation)
                    sample = self._latest_robot_sample()
                    if (
                        not sample.valid
                        or not sample.timestamp_reliable
                        or sample.host_timestamp_s < authorization_time
                        or sample.receive_timestamp_s < authorization_time
                    ):
                        raise RuntimeError("实机同步完成时的法兰样本不属于当前授权代次")
                else:
                    sender.connect()
                    sample = self._latest_robot_sample()
                with self._stop_generation_lock:
                    if operation_generation != self._stop_generation:
                        raise RuntimeError("使能期间收到停车请求，已取消使能")
                self.safety.enable(sample.T_base_flange, speed_mode=self.speed_profile)
                if self.mode is RuntimeMode.HARDWARE:
                    self._hardware_epoch_stable = True
                    self._hardware_sync_active = False
                self._active_sender = sender
                self.interpolator.reset()
                self._control_sequence = 0
                now = float(self._clock())
                self.interpolator.push_target(now, sample.T_base_flange, self._control_sequence)
                self._last_commanded_pose = np.array(sample.T_base_flange, copy=True)
            except Exception as error:
                self._hardware_sync_active = False
                self._hardware_epoch_stable = False
                sender.stop()
                if self.mode is RuntimeMode.HARDWARE:
                    self.emergency_stop(f"实机使能失败：{error}", fault=True)
                raise
            self._set_message("已使能，等待开始跟随")

    def start_following(self, *, start_threads: bool = True) -> None:
        with self._lifecycle_lock:
            if self.mode is RuntimeMode.HARDWARE and not self._hardware_epoch_stable:
                raise RuntimeError("实机授权代次尚未重新完成目标稳定同步")
            self.safety.start_following()
            self._stop_event.clear()
            self._following_started_s = float(self._clock())
            previous = self.latest_safety.get_latest()
            self._minimum_safety_sequence = 0 if previous is None else previous[0]
            if start_threads:
                self._start_control_threads_locked()
            self._set_message("视觉伺服跟随中")

    def _start_control_threads_locked(self) -> None:
        if any(thread.is_alive() for thread in self._control_threads):
            return
        self._control_threads = [
            threading.Thread(target=self._control_loop, name="mpc-20hz", daemon=True),
            threading.Thread(target=self._servo_loop, name="servo-125hz", daemon=True),
        ]
        for thread in self._control_threads:
            thread.start()

    def _start_vision_thread_locked(self) -> None:
        """系统检查通过后就开始处理唯一最新帧，以便在使能前累计 10 帧。"""

        if self._vision_thread is not None and self._vision_thread.is_alive():
            return
        self._vision_thread = threading.Thread(
            target=self._vision_loop,
            name="vision-processing",
            daemon=True,
        )
        self._vision_thread.start()

    def process_vision_step(
        self, frame: FramePacket | None = None
    ) -> MarkerPoseObservation:
        # 实机授权代次清理与视觉状态更新不得交叉，否则旧 EKF
        # 可能在 clear 之后又被写回。
        with self._pipeline_lock:
            return self._process_vision_step_locked(frame)

    def _process_vision_step_locked(
        self, frame: FramePacket | None = None
    ) -> MarkerPoseObservation:
        if self.detector is None:
            raise RuntimeError("标定未通过，检测器尚未创建")
        if frame is None:
            latest = self.camera_worker.latest_frame.get_latest()
            if latest is None:
                raise RuntimeError("尚未收到相机图像")
            frame = latest[1]
        if frame.sequence_number <= self._last_processed_frame_sequence:
            raise RuntimeError("拒绝重复处理同一帧")
        self._last_processed_frame_sequence = frame.sequence_number
        started = float(self._clock())
        observation = self.detector.detect(frame.image, frame.host_timestamp_s)
        self.latest_observation.publish(observation)
        annotated = self.detector.draw_detection(frame.image, observation)
        self.latest_annotated_frame.publish(
            FramePacket(
                frame.host_timestamp_s,
                frame.sequence_number,
                annotated,
                frame.valid,
                frame.quality,
                receive_timestamp_s=frame.receive_timestamp_s,
                timestamp_source=frame.timestamp_source,
                timestamp_reliable=frame.timestamp_reliable,
                timestamp_uncertainty_s=frame.timestamp_uncertainty_s,
            )
        )
        if self.video_recorder is not None:
            self.video_recorder.write(annotated)
        now = float(self._clock())
        self._last_frame_arrival_s = now
        if observation.valid:
            self._recognition_times.append(now)
            self._recognition_times = [
                value for value in self._recognition_times if now - value <= 1.0
            ]
        rate = float(len(self._recognition_times))
        latency = max(0.0, now - frame.host_timestamp_s)
        message = "视觉观测有效" if observation.valid else "未识别到固定 ID"
        self._metrics = RuntimeMetrics(rate, latency, message)
        if self.logger is not None:
            translation = observation.T_camera_marker[:3, 3]
            self.logger.write(
                "vision",
                {
                    "host_timestamp_s": observation.host_timestamp_s,
                    "frame_sequence": frame.sequence_number,
                    "marker_id": observation.marker_id,
                    "position_x_m": translation[0] if observation.valid else "",
                    "position_y_m": translation[1] if observation.valid else "",
                    "position_z_m": translation[2] if observation.valid else "",
                    "reprojection_error_px": observation.reprojection_error_px,
                    "pixel_area_px2": observation.pixel_area_px2,
                    "valid": observation.valid,
                    "quality": observation.quality,
                    "receive_timestamp_s": frame.receive_timestamp_s,
                    "timestamp_source": frame.timestamp_source,
                    "timestamp_reliable": frame.timestamp_reliable,
                    "timestamp_uncertainty_s": frame.timestamp_uncertainty_s,
                },
            )
        if not observation.valid:
            if self.state in {
                VisualServoState.VALIDATED,
                VisualServoState.OCCLUSION_DECAY,
            }:
                self.safety.observe_target(
                    False,
                    0.0,
                    0.0,
                    frame_sequence=frame.sequence_number,
                    observation_host_timestamp_s=observation.host_timestamp_s,
                )
            if self.ekf.initialized:
                try:
                    estimate = self.ekf.predict_to(frame.host_timestamp_s)
                except EkfTimestampGapError:
                    if self.state is not VisualServoState.VALIDATED:
                        raise
                    self._reset_estimation_after_preview_gap()
                    with self._hardware_sync_condition:
                        self._hardware_sync_condition.notify_all()
                    return observation
                prediction = self.predictor.predict(estimate, frame.host_timestamp_s)
                self.latest_estimate.publish(estimate)
                self.latest_prediction.publish(prediction)
            with self._hardware_sync_condition:
                self._hardware_sync_condition.notify_all()
            return observation
        hardware_epoch_active = self.mode is RuntimeMode.HARDWARE and (
            self._hardware_sync_active
            or self.state
            in {
                VisualServoState.ENABLED,
                VisualServoState.FOLLOWING,
                VisualServoState.OCCLUSION_DECAY,
            }
        )
        if hardware_epoch_active:
            authorization_time = self._hardware_authorization_time_s
            timing_valid = bool(
                authorization_time is not None
                and frame.sequence_number > self._hardware_frame_baseline_sequence
                and frame.host_timestamp_s >= authorization_time
                and frame.receive_timestamp_s >= authorization_time
                and frame.timestamp_reliable
                and frame.timestamp_uncertainty_s
                <= MAX_HARDWARE_TIMESTAMP_UNCERTAINTY_S
            )
            if not timing_valid:
                if self.state is VisualServoState.VALIDATED:
                    self.safety.observe_target(
                        False,
                        0.0,
                        0.0,
                        frame_sequence=frame.sequence_number,
                        observation_host_timestamp_s=observation.host_timestamp_s,
                    )
                    with self._hardware_sync_condition:
                        self._hardware_sync_condition.notify_all()
                    return observation
                raise RuntimeError("实机图像不属于当前授权代次或采样时间不可靠")
        # “选中实机”只是操作员的模式选择，并不代表控制侧已校验
        # TCP 并授信 RTDE 样本。使能前的后台视觉只能使用独立仿真历史；
        # 进入已使能状态后才允许读取实机历史。这也避免把授信前的图像与
        # 无效机器人样本组合成伪造位姿。
        hardware_pose_authorized = (
            self.mode is RuntimeMode.HARDWARE
            and (
                self._hardware_sync_active
                or self.state
                in {
                    VisualServoState.ENABLED,
                    VisualServoState.FOLLOWING,
                    VisualServoState.OCCLUSION_DECAY,
                }
            )
            and bool(getattr(self.state_reader, "tcp_zero_confirmed", False))
        )
        if hardware_pose_authorized:
            pose_history = self.state_reader.pose_history
        else:
            self._dry_sample_at(observation.host_timestamp_s)
            pose_history = self._dry_pose_history
        try:
            robot_at_image = pose_history.interpolate(observation.host_timestamp_s)
        except (LookupError, ValueError) as error:
            if self._hardware_sync_active and self.state is VisualServoState.VALIDATED:
                self.safety.observe_target(
                    False,
                    0.0,
                    0.0,
                    frame_sequence=frame.sequence_number,
                    observation_host_timestamp_s=observation.host_timestamp_s,
                )
                with self._hardware_sync_condition:
                    self._hardware_sync_condition.notify_all()
                return observation
            raise RuntimeError(f"图像时刻机器人位姿不可插值：{error}") from error
        robot_time_valid = bool(
            robot_at_image.valid
            and (
                not hardware_epoch_active
                or (
                    robot_at_image.timestamp_reliable
                    and robot_at_image.timestamp_uncertainty_s
                    <= MAX_HARDWARE_TIMESTAMP_UNCERTAINTY_S
                )
            )
        )
        if not robot_time_valid:
            if self._hardware_sync_active and self.state is VisualServoState.VALIDATED:
                self.safety.observe_target(
                    False,
                    0.0,
                    0.0,
                    frame_sequence=frame.sequence_number,
                    observation_host_timestamp_s=observation.host_timestamp_s,
                )
                with self._hardware_sync_condition:
                    self._hardware_sync_condition.notify_all()
                return observation
            raise RuntimeError("图像时刻机器人位姿的插值端点不可靠或不确定度超限")
        assert self._T_flange_camera is not None
        T_base_marker = compose_transforms(
            robot_at_image.T_base_flange,
            self._T_flange_camera,
            observation.T_camera_marker,
        )
        try:
            estimate = self.ekf.update(T_base_marker, observation.host_timestamp_s)
        except EkfTimestampGapError:
            if self.state is not VisualServoState.VALIDATED:
                raise
            self._reset_estimation_after_preview_gap()
            estimate = self.ekf.update(T_base_marker, observation.host_timestamp_s)
        prediction = self.predictor.predict(estimate, observation.host_timestamp_s)
        self.latest_estimate.publish(estimate)
        self.latest_prediction.publish(prediction)
        if self.logger is not None:
            position = estimate.T_base_target[:3, 3]
            self.logger.write(
                "ekf",
                {
                    "host_timestamp_s": estimate.host_timestamp_s,
                    "position_x_m": position[0],
                    "position_y_m": position[1],
                    "position_z_m": position[2],
                    "linear_speed_mps": np.linalg.norm(estimate.linear_velocity_mps),
                    "angular_speed_rad_s": np.linalg.norm(estimate.angular_velocity_rad_s),
                    "measurement_age_s": estimate.measurement_age_s,
                    "valid": estimate.valid,
                    "quality": estimate.quality,
                    "reason": self.ekf.last_rejection_reason,
                },
            )
            self.logger.write(
                "prediction",
                {
                    "host_timestamp_s": prediction.host_timestamp_s,
                    "horizon_s": prediction.sample_times_s[-1],
                    "sample_count": prediction.sample_times_s.size,
                    "valid": prediction.valid,
                    "quality": prediction.quality,
                    "reason": prediction.reason,
                },
            )
        if self.state is VisualServoState.VALIDATED:
            desired_flange = build_desired_flange_pose(
                T_base_marker,
                self._T_flange_camera,
                self._T_camera_marker_desired,
            )
            position_error = float(
                np.linalg.norm(desired_flange[:3, 3] - robot_at_image.T_base_flange[:3, 3])
            )
            rotation_error = float(
                np.linalg.norm(
                    so3_log(
                        robot_at_image.T_base_flange[:3, :3].T
                        @ desired_flange[:3, :3]
                    )
                )
            )
            self.safety.observe_target(
                prediction.valid,
                position_error,
                rotation_error,
                frame_sequence=frame.sequence_number,
                observation_host_timestamp_s=observation.host_timestamp_s,
            )
            with self._hardware_sync_condition:
                self._hardware_sync_condition.notify_all()
        if self.logger is not None:
            self.logger.write(
                "timing",
                {
                    "host_timestamp_s": now,
                    "stage": "vision",
                    "duration_ms": max(0.0, float(self._clock()) - started) * 1000.0,
                    "latency_ms": latency * 1000.0,
                },
            )
        return observation

    @staticmethod
    def _integrate_spatial_twist(
        pose: np.ndarray, twist: np.ndarray, dt_s: float
    ) -> np.ndarray:
        result = np.array(pose, dtype=np.float64, copy=True)
        result[:3, 3] += np.asarray(twist[:3]) * dt_s
        result[:3, :3] = so3_exp(np.asarray(twist[3:]) * dt_s) @ result[:3, :3]
        return validate_transform(result)

    def process_control_step(
        self,
        host_timestamp_s: float | None = None,
        *,
        monitor: VisualServoSafetyMonitor | None = None,
    ) -> SafetyDecision:
        now = float(self._clock() if host_timestamp_s is None else host_timestamp_s)
        active_monitor = self.safety if monitor is None else monitor
        if active_monitor.state not in {
            VisualServoState.FOLLOWING,
            VisualServoState.OCCLUSION_DECAY,
        }:
            raise RuntimeError("只有跟随/遮挡衰减状态可运行 MPC")
        latest_estimate = self.latest_estimate.get_latest()
        if latest_estimate is None:
            raise RuntimeError("没有 EKF 状态")
        estimate = latest_estimate[1]
        # 控制线程不修改 EKF 内部时间，否则较早曝光但稍后到达的图像会被误判
        # 为乱序。预测器在不改写 EKF 的前提下把快照外推到 now。
        marker_trajectory = self.predictor.predict(estimate, now)
        assert self._T_flange_camera is not None
        flange_trajectory = build_desired_flange_trajectory(
            marker_trajectory,
            self._T_flange_camera,
            self._T_camera_marker_desired,
        )
        self.latest_prediction.publish(marker_trajectory)
        robot = self._latest_robot_sample()
        if self.mode is RuntimeMode.HARDWARE:
            frame_item = self.camera_worker.latest_frame.get_latest()
            authorization_time = self._hardware_authorization_time_s
            if (
                not self._hardware_epoch_stable
                or authorization_time is None
                or frame_item is None
                or frame_item[1].sequence_number
                <= self._hardware_frame_baseline_sequence
                or frame_item[1].host_timestamp_s < authorization_time
                or not frame_item[1].timestamp_reliable
                or frame_item[1].timestamp_uncertainty_s
                > MAX_HARDWARE_TIMESTAMP_UNCERTAINTY_S
            ):
                raise RuntimeError("相机采样时间失去可靠性")
            if (
                not robot.valid
                or robot.host_timestamp_s < authorization_time
                or robot.receive_timestamp_s < authorization_time
                or not robot.timestamp_reliable
                or robot.timestamp_uncertainty_s
                > MAX_HARDWARE_TIMESTAMP_UNCERTAINTY_S
            ):
                raise RuntimeError("机器人采样时间失去可靠性")
        result = self.mpc.solve(
            robot.T_base_flange,
            robot.tcp_speed,
            flange_trajectory,
            now,
            self.speed_profile,
        )
        self.latest_mpc.publish(result)
        if self.logger is not None:
            self.logger.write(
                "robot_state",
                {
                    "host_timestamp_s": robot.host_timestamp_s,
                    "sequence_number": robot.sequence_number,
                    "robot_mode": robot.robot_mode,
                    "x_m": robot.T_base_flange[0, 3],
                    "y_m": robot.T_base_flange[1, 3],
                    "z_m": robot.T_base_flange[2, 3],
                    "valid": robot.valid,
                    "quality": robot.quality,
                    "receive_timestamp_s": robot.receive_timestamp_s,
                    "timestamp_source": robot.timestamp_source,
                    "timestamp_reliable": robot.timestamp_reliable,
                    "timestamp_uncertainty_s": robot.timestamp_uncertainty_s,
                },
            )
            self.logger.write(
                "mpc",
                {
                    "host_timestamp_s": result.host_timestamp_s,
                    "solver_status": result.solver_status,
                    "solve_time_s": result.solve_time_s,
                    "linear_speed_mps": np.linalg.norm(result.commanded_twist[:3]),
                    "angular_speed_rad_s": np.linalg.norm(result.commanded_twist[3:]),
                    "valid": result.valid,
                    "quality": result.quality,
                },
            )
        command_pose = self._integrate_spatial_twist(
            robot.T_base_flange, result.commanded_twist, self.mpc.config.dt_s
        )
        previous = (
            robot.T_base_flange
            if self._last_commanded_pose is None
            else self._last_commanded_pose
        )
        latest_observation = self.latest_observation.get_latest()
        capture_timestamp = (
            estimate.host_timestamp_s
            if latest_observation is None
            else latest_observation[1].host_timestamp_s
        )
        inputs = SafetyInputs(
            calibration_valid=self._T_flange_camera is not None,
            resolution_matches=self._latest_frame_resolution_matches(),
            tcp_zero_verified=(
                True
                if self.mode is RuntimeMode.DRY_RUN
                else bool(self.state_reader.tcp_zero_confirmed)
            ),
            robot_state_valid=bool(robot.valid),
            latency_s=max(0.0, now - capture_timestamp),
            target_estimate=estimate,
            mpc_result=result,
            commanded_flange_pose=command_pose,
            previous_flange_pose=previous,
        )
        decision = active_monitor.evaluate(now, inputs)
        self.latest_safety.publish(decision)
        if self.logger is not None:
            self.logger.write(
                "safety_events",
                {
                    "host_timestamp_s": decision.host_timestamp_s,
                    "state": self.state.name,
                    "allow_motion": decision.allow_motion,
                    "generation": decision.decision_generation,
                    "valid": decision.valid,
                    "quality": decision.quality,
                    "reason": decision.reason,
                },
            )
        if not decision.allow_motion:
            if active_monitor is self.safety:
                self.emergency_stop(decision.reason, fault=True)
            return decision
        self._control_sequence += 1
        self.interpolator.push_target(
            now + self.mpc.config.dt_s,
            command_pose,
            self._control_sequence,
        )
        self._last_commanded_pose = command_pose
        return decision

    def process_servo_step(
        self,
        host_timestamp_s: float | None = None,
        *,
        monitor: VisualServoSafetyMonitor | None = None,
        sender: Any | None = None,
    ) -> bool:
        now = float(self._clock() if host_timestamp_s is None else host_timestamp_s)
        active_monitor = self.safety if monitor is None else monitor
        active_sender = self._active_sender if sender is None else sender
        latest = self.latest_safety.get_latest()
        if latest is None or latest[0] <= self._minimum_safety_sequence:
            if (
                self.state
                in {VisualServoState.FOLLOWING, VisualServoState.OCCLUSION_DECAY}
                and self._following_started_s is not None
                and now - self._following_started_s > active_monitor.config.max_latency_s
            ):
                self.emergency_stop("servoL 等待安全决定超时", fault=True)
            return False
        if active_sender is None:
            if self.state in {
                VisualServoState.FOLLOWING,
                VisualServoState.OCCLUSION_DECAY,
            }:
                self.emergency_stop("servoL 没有活动发送器", fault=True)
            return False
        decision = latest[1]

        def send(_safe_twist: np.ndarray) -> None:
            pose_ur = self.interpolator.sample_ur_pose(now)
            active_sender.send_pose(pose_ur)

        executed = active_monitor.execute_guarded(decision, send, now_s=now)
        if not executed:
            reason = active_monitor.last_guard_rejection_reason or "servoL 安全门拒绝发送"
            if active_monitor is self.safety:
                self.emergency_stop(reason, fault=True)
        return executed

    def _vision_loop(self) -> None:
        sequence = 0
        while not self._shutdown_event.is_set():
            try:
                newer = self.camera_worker.latest_frame.wait_for_newer(sequence, 0.1)
                if newer is None:
                    if self.camera_worker.last_error is not None:
                        raise RuntimeError(f"相机断开：{self.camera_worker.last_error}")
                    continue
                sequence, frame = newer
                self.process_vision_step(frame)
            except Exception as error:
                self.emergency_stop(f"视觉线程异常：{error}", fault=True)
                return

    def _control_loop(self) -> None:
        period = 0.05
        deadline = self._clock()
        while not self._stop_event.is_set():
            try:
                if getattr(self.state_reader, "watchdog_fault_event", None) is not None:
                    if self.state_reader.watchdog_fault_event.is_set():
                        raise RuntimeError(f"RTDE 看门狗：{self.state_reader.last_error}")
                self.process_control_step(float(self._clock()))
            except Exception as error:
                self.emergency_stop(f"控制线程异常：{error}", fault=True)
                return
            deadline += period
            self._stop_event.wait(max(0.0, deadline - self._clock()))

    def _servo_loop(self) -> None:
        period = 1.0 / 125.0
        deadline = self._clock()
        while not self._stop_event.is_set():
            try:
                self.process_servo_step(float(self._clock()))
            except Exception as error:
                self.emergency_stop(f"servoL 线程异常：{error}", fault=True)
                return
            deadline += period
            self._stop_event.wait(max(0.0, deadline - self._clock()))

    def emergency_stop(self, reason: str, *, fault: bool = False) -> None:
        if not isinstance(reason, str) or not reason.strip():
            reason = "unknown stop"
        with self._stop_generation_lock:
            self._stop_generation += 1
        self._stop_event.set()
        self._hardware_sync_active = False
        self._hardware_epoch_stable = False
        # 安全状态机自身有锁；这里故意不等待可能被厂商 connect 阻塞的生命周期锁。
        if self.state not in {VisualServoState.STOPPED, VisualServoState.FAULT}:
            (self.safety.fault if fault else self.safety.stop)(reason)
        # 不论当前是干运行还是实机，都逐个 best-effort 停止；同一对象只调用一次。
        failures: list[str] = []
        seen: set[int] = set()
        for sender in (self._active_sender, self.real_sender, self._dry_sender):
            if sender is None or id(sender) in seen:
                continue
            seen.add(id(sender))
            try:
                sender.stop()
            except Exception as error:
                failures.append(f"{type(sender).__name__}：{error}")
        if failures:
            message = (
                f"停车未确认：{reason}；" + "；".join(failures)
                + "。请立即使用物理急停并检查机器人状态"
            )
            self._set_message(message)
            raise RuntimeStopUnconfirmedError(failures)
        self._set_message(f"已停止：{reason}")

    def acknowledge_stop(self) -> None:
        with self._lifecycle_lock:
            self.safety.acknowledge()
            self._active_sender = None
            self._dry_run_completed = False
            self._hardware_confirmed = False
            self.mode = RuntimeMode.PREVIEW
            self.interpolator.reset()
            self._following_started_s = None
            if self.detector is not None:
                self.detector.reset_tracking()
            self._set_message("已人工确认；必须重新系统检查")

    def shutdown(self) -> None:
        if not self._shutdown_lock.acquire(blocking=False):
            raise RuntimeShutdownError(["关闭流程已在另一个线程中执行"])
        failures: list[str] = []
        try:
            if self._shutdown_complete:
                return
            if self._shutdown_in_progress:
                raise RuntimeShutdownError(["关闭流程已在另一个线程中执行"])
            self._shutdown_in_progress = True
            self._shutdown_event.set()
            try:
                try:
                    self.emergency_stop("窗口关闭/系统退出")
                except RuntimeStopUnconfirmedError as error:
                    failures.append(f"停车未确认，请使用物理急停：{error}")
                current = threading.current_thread()
                threads = tuple(self._control_threads) + (
                    () if self._vision_thread is None else (self._vision_thread,)
                )
                for thread in threads:
                    if thread is current:
                        continue
                    thread.join(1.0)
                    if thread.is_alive():
                        failures.append(f"线程 {thread.name} 未在时限内退出")
                closers = (
                    ("相机", lambda: self.camera_worker.stop(timeout_s=2.0)),
                    ("机器人状态", lambda: self.state_reader.stop(timeout_s=2.0)),
                    ("实机发送器", self.real_sender.disconnect),
                    ("干运行发送器", self._dry_sender.disconnect),
                    ("视频", getattr(self.video_recorder, "close", lambda: None)),
                    ("日志", getattr(self.logger, "close", lambda: None)),
                )
                for name, closer in closers:
                    try:
                        closer()
                    except Exception as error:
                        failures.append(f"{name}关闭失败：{error}")
            finally:
                self._shutdown_in_progress = False
                self._shutdown_failures = tuple(failures)
                self._shutdown_complete = not failures
            if failures:
                self._set_message("关闭失败，可重试：" + "; ".join(failures))
                raise RuntimeShutdownError(failures)
        finally:
            self._shutdown_lock.release()

    def ui_snapshot(self) -> dict[str, Any]:
        observation_item = self.latest_observation.get_latest()
        frame_item = self.latest_annotated_frame.get_latest()
        if frame_item is None:
            # 标定未通过时仍允许显示未经位姿计算的原始预览画面。
            frame_item = self.camera_worker.latest_frame.get_latest()
        observation = None if observation_item is None else observation_item[1]
        distance = None
        reprojection = None
        marker_id = None
        if observation is not None and observation.valid:
            distance = float(np.linalg.norm(observation.T_camera_marker[:3, 3]))
            reprojection = observation.reprojection_error_px
            marker_id = observation.marker_id
        estimate_item = self.latest_estimate.get_latest()
        if estimate_item is None:
            ekf_status = "等待观测"
        else:
            estimate = estimate_item[1]
            ekf_status = "有效" if estimate.valid else "无效"
        mpc_item = self.latest_mpc.get_latest()
        mpc_status = "等待求解" if mpc_item is None else mpc_item[1].solver_status
        robot_item = self.state_reader.latest_state.get_latest()
        if robot_item is None:
            robot_status = "未连接"
        else:
            robot = robot_item[1]
            robot_status = f"{robot.robot_mode} / {'有效' if robot.valid else '无效'}"
        safety_item = self.latest_safety.get_latest()
        if safety_item is None:
            safety_status = self.state.name
        else:
            motion = "允许运动" if safety_item[1].allow_motion else "禁止运动"
            safety_status = f"{self.state.name} / {motion}"
        return {
            "state": self.state.name,
            "mode": self.mode.value,
            "speed_profile": self.speed_profile,
            "message": self._metrics.last_message,
            "marker_id": marker_id,
            "distance_m": distance,
            "reprojection_error_px": reprojection,
            "recognition_rate_hz": self._metrics.recognition_rate_hz,
            "latency_ms": self._metrics.latency_s * 1000.0,
            "image": None if frame_item is None else frame_item[1].image,
            "camera_model": CAMERA_MODEL,
            "camera_serial": CAMERA_SERIAL,
            "dry_run_completed": self._dry_run_completed,
            "ekf_status": ekf_status,
            "mpc_status": mpc_status,
            "robot_status": robot_status,
            "safety_status": safety_status,
            "robot_read_only_connected": self._robot_read_only_connected,
        }
