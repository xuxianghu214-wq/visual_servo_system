"""视觉伺服安全状态机和最终运动许可。

本模块采用故障闭锁：任何运行期门禁失败都会进入 ``FAULT``，之后正常数据恢复也不会
自动重新运动；必须人工确认并从预览、系统检查、稳定目标、使能的顺序重新开始。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from visual_servo_system.geometry.pose_transforms import so3_log, validate_transform
from visual_servo_system.models.data_types import MpcResult, SafetyDecision, TargetStateEstimate

if TYPE_CHECKING:
    from visual_servo_system.config.config_loader import SafetyLimits


class VisualServoState(Enum):
    DISCONNECTED = auto()
    PREVIEW = auto()
    VALIDATED = auto()
    ENABLED = auto()
    FOLLOWING = auto()
    OCCLUSION_DECAY = auto()
    STOPPED = auto()
    FAULT = auto()


@dataclass(frozen=True, slots=True)
class SafetyMonitorConfig:
    stable_frames_required: int = 10
    initial_max_position_error_m: float = 0.080
    initial_max_rotation_error_rad: float = np.deg2rad(10.0)
    workspace_envelope_position_m: float = 0.100
    workspace_envelope_rotation_rad: float = np.deg2rad(15.0)
    max_single_cycle_position_m: float = 0.0012
    max_single_cycle_rotation_rad: float = np.deg2rad(0.30)
    max_stable_frame_interval_s: float = 0.200
    max_latency_s: float = 0.100
    max_mpc_solve_time_s: float = 0.020
    dropout_timeout_s: float = 1.0
    max_3sigma_position_m: float = 0.030
    max_3sigma_rotation_rad: float = np.deg2rad(5.0)
    commissioning_speed_mps: float = 0.050
    experiment_speed_mps: float = 0.120
    commissioning_angular_speed_rad_s: float = np.deg2rad(15.0)
    experiment_angular_speed_rad_s: float = np.deg2rad(30.0)

    @classmethod
    def from_settings(cls, safety: SafetyLimits) -> SafetyMonitorConfig:
        """从 ``SafetyLimits`` 创建运行配置，角度在边界处统一转为弧度。"""

        return cls(
            stable_frames_required=int(safety.stable_frames_required),
            initial_max_position_error_m=float(safety.initial_max_position_error_m),
            initial_max_rotation_error_rad=np.deg2rad(
                safety.initial_max_rotation_error_deg
            ),
            workspace_envelope_position_m=float(safety.workspace_envelope_position_m),
            workspace_envelope_rotation_rad=np.deg2rad(
                safety.workspace_envelope_rotation_deg
            ),
            max_latency_s=float(safety.max_latency_s),
            dropout_timeout_s=float(safety.dropout_timeout_s),
            max_3sigma_position_m=float(safety.max_3sigma_position_m),
            max_3sigma_rotation_rad=np.deg2rad(safety.max_3sigma_rotation_deg),
            commissioning_speed_mps=float(safety.commissioning_speed_mps),
            experiment_speed_mps=float(safety.experiment_speed_mps),
            commissioning_angular_speed_rad_s=np.deg2rad(
                safety.commissioning_angular_speed_deg_s
            ),
            experiment_angular_speed_rad_s=np.deg2rad(
                safety.experiment_angular_speed_deg_s
            ),
        )

    def __post_init__(self) -> None:
        if type(self.stable_frames_required) is not int or self.stable_frames_required <= 0:
            raise ValueError("stable_frames_required must be a positive integer")
        numeric = tuple(
            value
            for name, value in vars_from_slots(self).items()
            if name != "stable_frames_required"
        )
        if not all(np.isfinite(value) and value > 0.0 for value in numeric):
            raise ValueError("all safety limits must be finite and positive")


def vars_from_slots(instance: object) -> dict[str, object]:
    return {name: getattr(instance, name) for name in instance.__slots__}


@dataclass(frozen=True, slots=True)
class SafetyInputs:
    calibration_valid: bool
    resolution_matches: bool
    tcp_zero_verified: bool
    robot_state_valid: bool
    latency_s: float
    target_estimate: TargetStateEstimate
    mpc_result: MpcResult
    commanded_flange_pose: np.ndarray
    previous_flange_pose: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "calibration_valid",
            "resolution_matches",
            "tcp_zero_verified",
            "robot_state_valid",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if not np.isfinite(self.latency_s) or self.latency_s < 0.0:
            raise ValueError("latency_s must be finite and non-negative")
        if not isinstance(self.target_estimate, TargetStateEstimate):
            raise TypeError("target_estimate must be TargetStateEstimate")
        if not isinstance(self.mpc_result, MpcResult):
            raise TypeError("mpc_result must be MpcResult")
        object.__setattr__(
            self, "commanded_flange_pose", validate_transform(self.commanded_flange_pose)
        )
        object.__setattr__(
            self, "previous_flange_pose", validate_transform(self.previous_flange_pose)
        )


class VisualServoSafetyMonitor:
    def __init__(
        self,
        config: SafetyMonitorConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.config = config or SafetyMonitorConfig()
        self._clock = clock
        self._lock = threading.RLock()
        self._state = VisualServoState.DISCONNECTED
        self._generation = 0
        self._stable_frames = 0
        self._last_observation_sequence: int | None = None
        self._last_observation_timestamp_s: float | None = None
        self._last_position_error_m: float | None = None
        self._last_rotation_error_rad: float | None = None
        self._arming_pose: np.ndarray | None = None
        self._latched_reason = "not connected"
        self._last_guard_rejection_reason = "not executed"
        self._speed_mode = "commissioning"
        self._next_guard_token = 0
        self._inflight_guard_tokens: set[int] = set()

    @property
    def stable_frames(self) -> int:
        with self._lock:
            return self._stable_frames

    @property
    def state(self) -> VisualServoState:
        with self._lock:
            return self._state

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def last_guard_rejection_reason(self) -> str:
        with self._lock:
            return self._last_guard_rejection_reason

    @property
    def inflight_guard_count(self) -> int:
        """已通过安全门、但回调尚未返回的命令数。"""

        with self._lock:
            return len(self._inflight_guard_tokens)

    def _transition(self, state: VisualServoState, reason: str) -> None:
        """锁内切换状态并使所有旧安全决定立即失效。"""

        self._state = state
        self._latched_reason = reason
        self._generation += 1

    def _reset_stability_tracking(self) -> None:
        self._stable_frames = 0
        self._last_position_error_m = None
        self._last_rotation_error_rad = None
        self._last_observation_sequence = None
        self._last_observation_timestamp_s = None

    def connect_preview(self) -> None:
        with self._lock:
            if self._state is not VisualServoState.DISCONNECTED:
                raise RuntimeError("preview connection is only allowed from DISCONNECTED")
            self._reset_stability_tracking()
            self._transition(VisualServoState.PREVIEW, "preview")

    def system_check(
        self,
        calibration_valid: bool,
        resolution_matches: bool,
        tcp_zero_verified: bool,
        robot_state_valid: bool,
    ) -> bool:
        with self._lock:
            if self._state is not VisualServoState.PREVIEW:
                raise RuntimeError("system check is only allowed from PREVIEW")
            checks = (
                calibration_valid,
                resolution_matches,
                tcp_zero_verified,
                robot_state_valid,
            )
            if any(type(value) is not bool for value in checks):
                raise TypeError("system check flags must be bool")
            self._reset_stability_tracking()
            if not all(checks):
                self._generation += 1
                return False
            self._transition(VisualServoState.VALIDATED, "system validated")
            return True

    def observe_target(
        self,
        valid: bool,
        position_error_m: float,
        rotation_error_rad: float,
        *,
        frame_sequence: int,
        observation_host_timestamp_s: float,
    ) -> int:
        with self._lock:
            if self._state not in {
                VisualServoState.VALIDATED,
                VisualServoState.OCCLUSION_DECAY,
            }:
                raise RuntimeError("target stability is only counted after validation")
            if type(valid) is not bool:
                raise TypeError("valid must be bool")
            if type(frame_sequence) is not int or frame_sequence < 0:
                raise ValueError("frame_sequence must be a non-negative integer")
            observation_timestamp = float(observation_host_timestamp_s)
            if not np.isfinite(observation_timestamp) or observation_timestamp < 0.0:
                raise ValueError("observation timestamp must be finite and non-negative")
            position = float(position_error_m)
            rotation = float(rotation_error_rad)
            if (
                not np.isfinite(position)
                or not np.isfinite(rotation)
                or position < 0
                or rotation < 0
            ):
                raise ValueError("target errors must be finite and non-negative")

            if self._last_observation_sequence is not None:
                if (
                    frame_sequence <= self._last_observation_sequence
                    or observation_timestamp <= self._last_observation_timestamp_s
                ):
                    self._stable_frames = 0
                    return self._stable_frames
                if (
                    observation_timestamp - self._last_observation_timestamp_s
                    > self.config.max_stable_frame_interval_s
                ):
                    self._stable_frames = 0

            self._last_observation_sequence = frame_sequence
            self._last_observation_timestamp_s = observation_timestamp
            self._last_position_error_m = position
            self._last_rotation_error_rad = rotation
            if (
                valid
                and position <= self.config.initial_max_position_error_m
                and rotation <= self.config.initial_max_rotation_error_rad
            ):
                self._stable_frames += 1
            else:
                self._stable_frames = 0
            return self._stable_frames

    def reset_target_stability(self) -> None:
        """进入新实机授权代次时丢弃干运行/旧目标的稳定计数。"""

        with self._lock:
            if self._state is not VisualServoState.VALIDATED:
                raise RuntimeError("target stability can only reset from VALIDATED")
            self._reset_stability_tracking()

    def enable(
        self, T_base_flange_arming: np.ndarray, *, speed_mode: str = "commissioning"
    ) -> None:
        with self._lock:
            if self._state is not VisualServoState.VALIDATED:
                raise RuntimeError("enable is only allowed from VALIDATED")
            if self._stable_frames < self.config.stable_frames_required:
                raise RuntimeError("target is not stable for the required number of frames")
            if (
                self._last_position_error_m is None
                or self._last_position_error_m > self.config.initial_max_position_error_m
                or self._last_rotation_error_rad is None
                or self._last_rotation_error_rad
                > self.config.initial_max_rotation_error_rad
            ):
                raise RuntimeError("initial pose error exceeds the enable limit")
            if speed_mode not in {"commissioning", "experiment"}:
                raise ValueError("speed_mode must be 'commissioning' or 'experiment'")
            self._arming_pose = validate_transform(T_base_flange_arming)
            self._speed_mode = speed_mode
            self._transition(VisualServoState.ENABLED, "enabled")

    def start_following(self) -> None:
        with self._lock:
            if self._state is not VisualServoState.ENABLED:
                raise RuntimeError("following can only start from ENABLED")
            self._transition(VisualServoState.FOLLOWING, "following")

    def enter_occlusion_decay(self) -> None:
        with self._lock:
            if self._state is not VisualServoState.FOLLOWING:
                raise RuntimeError("occlusion decay can only start from FOLLOWING")
            self._transition(
                VisualServoState.OCCLUSION_DECAY, "short target occlusion"
            )

    def recover_from_occlusion(self) -> None:
        with self._lock:
            if self._state is not VisualServoState.OCCLUSION_DECAY:
                raise RuntimeError("occlusion recovery requires OCCLUSION_DECAY")
            self._transition(VisualServoState.FOLLOWING, "tracking recovered")

    def stop(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("stop reason must be non-empty")
        with self._lock:
            self._transition(VisualServoState.STOPPED, reason)

    def fault(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("fault reason must be non-empty")
        with self._lock:
            self._transition(VisualServoState.FAULT, reason)

    def acknowledge(self) -> None:
        with self._lock:
            if self._state not in {VisualServoState.STOPPED, VisualServoState.FAULT}:
                raise RuntimeError("only a stopped or faulted system can be acknowledged")
            self._reset_stability_tracking()
            self._arming_pose = None
            self._transition(
                VisualServoState.PREVIEW,
                "manual acknowledgement; system check required",
            )

    def _zero_decision(self, timestamp: float, reason: str) -> SafetyDecision:
        return SafetyDecision(
            timestamp,
            False,
            np.zeros(6),
            reason,
            True,
            1.0,
            decision_generation=self._generation,
        )

    def _trip(self, timestamp: float, reason: str) -> SafetyDecision:
        self.fault(reason)
        return self._zero_decision(timestamp, reason)

    @staticmethod
    def _three_sigma_radius(covariance_block: np.ndarray) -> float:
        symmetric = (covariance_block + covariance_block.T) * 0.5
        largest_variance = float(np.max(np.linalg.eigvalsh(symmetric)))
        return 3.0 * float(np.sqrt(max(largest_variance, 0.0)))

    def _timestamp_fault_reason(
        self, now_s: float, product_timestamp_s: float, product_name: str
    ) -> str | None:
        age_s = now_s - product_timestamp_s
        tolerance_s = 1e-9
        if age_s < -tolerance_s:
            return f"{product_name} timestamp is in the future"
        if age_s > self.config.max_latency_s + tolerance_s:
            return f"{product_name} timestamp is stale"
        return None

    def evaluate(self, host_timestamp_s: float, inputs: SafetyInputs) -> SafetyDecision:
        timestamp = float(host_timestamp_s)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("host_timestamp_s must be finite and non-negative")
        with self._lock:
            if self._state in {VisualServoState.STOPPED, VisualServoState.FAULT}:
                return self._zero_decision(timestamp, self._latched_reason)
            if self._state not in {
                VisualServoState.FOLLOWING,
                VisualServoState.OCCLUSION_DECAY,
            }:
                return self._zero_decision(
                    timestamp, f"state {self._state.name} forbids motion"
                )

            flag_checks = (
                (inputs.calibration_valid, "calibration invalid"),
                (inputs.resolution_matches, "camera resolution mismatch"),
                (inputs.tcp_zero_verified, "active TCP is not verified zero"),
                (inputs.robot_state_valid, "robot state invalid"),
            )
            for passed, reason in flag_checks:
                if not passed:
                    return self._trip(timestamp, reason)
            # latency_s 是独立的端到端测量，不能替代每个数据产品自身的时间戳检查。
            if inputs.latency_s > self.config.max_latency_s:
                return self._trip(timestamp, "capture-to-control latency exceeded")

            estimate = inputs.target_estimate
            result = inputs.mpc_result
            products = [
                ("target estimate", estimate.host_timestamp_s),
                ("MPC result", result.host_timestamp_s),
            ]
            trajectory = result.predicted_trajectory
            if trajectory is not None:
                products.append(("trajectory", trajectory.host_timestamp_s))
            for product_name, product_timestamp in products:
                reason = self._timestamp_fault_reason(
                    timestamp, product_timestamp, product_name
                )
                if reason is not None:
                    return self._trip(timestamp, reason)
            product_timestamps = [item[1] for item in products]
            if (
                max(product_timestamps) - min(product_timestamps)
                > self.config.max_latency_s + 1e-9
            ):
                return self._trip(timestamp, "control product timestamps are mismatched")

            if not estimate.valid:
                return self._trip(timestamp, "target estimate invalid")
            estimate_transport_age_s = max(
                0.0, timestamp - estimate.host_timestamp_s
            )
            measurement_age_s = (
                estimate.measurement_age_s + estimate_transport_age_s
            )
            if measurement_age_s >= self.config.dropout_timeout_s:
                return self._trip(timestamp, "target dropout timeout reached")
            position_sigma = self._three_sigma_radius(estimate.covariance[0:3, 0:3])
            rotation_sigma = self._three_sigma_radius(estimate.covariance[3:6, 3:6])
            if position_sigma > self.config.max_3sigma_position_m:
                return self._trip(timestamp, "target position uncertainty exceeded")
            if rotation_sigma > self.config.max_3sigma_rotation_rad:
                return self._trip(timestamp, "target rotation uncertainty exceeded")

            if not result.valid:
                return self._trip(timestamp, f"MPC invalid: {result.solver_status}")
            if trajectory is not None and not trajectory.valid:
                return self._trip(timestamp, f"trajectory invalid: {trajectory.reason}")
            if result.solve_time_s > self.config.max_mpc_solve_time_s:
                return self._trip(timestamp, "MPC solve time exceeded")
            linear_limit = (
                self.config.commissioning_speed_mps
                if self._speed_mode == "commissioning"
                else self.config.experiment_speed_mps
            )
            angular_limit = (
                self.config.commissioning_angular_speed_rad_s
                if self._speed_mode == "commissioning"
                else self.config.experiment_angular_speed_rad_s
            )
            if np.linalg.norm(result.commanded_twist[:3]) > linear_limit + 1e-12:
                return self._trip(timestamp, "linear speed limit exceeded")
            if np.linalg.norm(result.commanded_twist[3:]) > angular_limit + 1e-12:
                return self._trip(timestamp, "angular speed limit exceeded")

            assert self._arming_pose is not None
            command = inputs.commanded_flange_pose
            position_from_arm = np.linalg.norm(
                command[:3, 3] - self._arming_pose[:3, 3]
            )
            rotation_from_arm = np.linalg.norm(
                so3_log(self._arming_pose[:3, :3].T @ command[:3, :3])
            )
            if position_from_arm > self.config.workspace_envelope_position_m:
                return self._trip(timestamp, "software position envelope exceeded")
            if rotation_from_arm > self.config.workspace_envelope_rotation_rad:
                return self._trip(timestamp, "software rotation envelope exceeded")

            previous = inputs.previous_flange_pose
            step_position = np.linalg.norm(command[:3, 3] - previous[:3, 3])
            step_rotation = np.linalg.norm(
                so3_log(previous[:3, :3].T @ command[:3, :3])
            )
            if step_position > self.config.max_single_cycle_position_m:
                return self._trip(timestamp, "single-cycle position increment exceeded")
            if step_rotation > self.config.max_single_cycle_rotation_rad:
                return self._trip(timestamp, "single-cycle rotation increment exceeded")

            if (
                estimate.measurement_age_s > 0.0
                and self._state is VisualServoState.FOLLOWING
            ):
                self._transition(
                    VisualServoState.OCCLUSION_DECAY, "occlusion decay"
                )
            return SafetyDecision(
                host_timestamp_s=timestamp,
                allow_motion=True,
                safe_twist=result.commanded_twist,
                reason=(
                    "occlusion decay"
                    if self._state is VisualServoState.OCCLUSION_DECAY
                    else "ok"
                ),
                valid=True,
                quality=float(min(estimate.quality, result.quality)),
                decision_generation=self._generation,
            )

    def execute_guarded(
        self,
        decision: SafetyDecision,
        callback: Callable[[np.ndarray], object],
        *,
        now_s: float | None = None,
    ) -> bool:
        """验证决定并登记在途 token，然后在安全锁外调用厂商发送函数。

        用户停车必须能立即取得安全锁并递增代次，不能被可能永久阻塞的 SDK 回调
        拖住。已进入回调的命令无法由 Python 假装中断，其最终确认由机器人发送层负责。
        """

        if not isinstance(decision, SafetyDecision):
            raise TypeError("decision must be SafetyDecision")
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            if decision.decision_generation != self._generation:
                self._last_guard_rejection_reason = (
                    "decision generation is stale after a safety state change"
                )
                return False
            if not decision.allow_motion or not decision.valid:
                self._last_guard_rejection_reason = (
                    "decision does not allow valid motion"
                )
                return False
            if self._state not in {
                VisualServoState.FOLLOWING,
                VisualServoState.OCCLUSION_DECAY,
            }:
                self._last_guard_rejection_reason = (
                    f"state {self._state.name} forbids guarded execution"
                )
                return False

            execution_timestamp = float(self._clock() if now_s is None else now_s)
            if not np.isfinite(execution_timestamp) or execution_timestamp < 0.0:
                raise ValueError("guard execution timestamp must be finite and non-negative")
            decision_age_s = execution_timestamp - decision.host_timestamp_s
            tolerance_s = 1e-9
            if decision_age_s < -tolerance_s:
                self._last_guard_rejection_reason = "decision timestamp is in the future"
                return False
            if decision_age_s > self.config.max_latency_s + tolerance_s:
                self._last_guard_rejection_reason = "decision timestamp is stale"
                return False
            self._next_guard_token += 1
            token = self._next_guard_token
            self._inflight_guard_tokens.add(token)

        try:
            callback(decision.safe_twist)
        except Exception as error:
            with self._lock:
                self._transition(
                    VisualServoState.FAULT,
                    f"guarded command callback failed: {type(error).__name__}: {error}",
                )
            raise
        finally:
            with self._lock:
                self._inflight_guard_tokens.discard(token)

        with self._lock:
            self._last_guard_rejection_reason = "executed"
        return True
