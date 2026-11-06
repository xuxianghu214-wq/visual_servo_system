"""UR ``servoL`` 安全发送封装和无硬件仿真假后端。"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from visual_servo_system.robot.ur_rtde_state_reader import _exact_zero_tcp


class MotionCommandRejected(RuntimeError):
    """软件运动门禁拒绝一条命令。"""


class ControlCleanupError(RuntimeError):
    """一个或多个停车/释放步骤失败；成功步骤不会在重试时重复。"""

    def __init__(self, failures: list[str], succeeded: set[str] | None = None) -> None:
        self.failures = tuple(failures)
        self.succeeded = frozenset(succeeded or set())
        super().__init__("；".join(failures))


class RobotStopTimeout(ControlCleanupError):
    """软件无法在短时限内确认机器人停车。"""


class TcpAuthorizationTarget(Protocol):
    """可被控制发送器授予/撤销“活动 TCP 为零”语义的状态读取器。"""

    def confirm_zero_tcp(self, tcp_offset_ur: Any) -> None: ...

    def revoke_tcp_confirmation(self) -> None: ...


_REQUIRED_CONTROL_METHODS = (
    "isPoseWithinSafetyLimits",
    "getInverseKinematicsHasSolution",
    "servoL",
    "initPeriod",
    "waitPeriod",
    "getTCPOffset",
    "servoStop",
    "stopScript",
    "disconnect",
)


def _cleanup_control(
    control: Any,
    *,
    send_stop: bool,
    disconnect: bool,
    completed: set[str] | None = None,
) -> set[str]:
    """逐项 best-effort 清理，并把所有失败连同成功项返回给调用者。"""

    succeeded = set(completed or set())
    operations: list[str] = []
    if send_stop:
        operations.extend(("servoStop", "stopScript"))
    if disconnect:
        operations.append("disconnect")
    failures: list[str] = []
    for method_name in operations:
        if method_name in succeeded:
            continue
        method = getattr(control, method_name, None)
        if not callable(method):
            failures.append(f"{method_name} 不可用")
            continue
        try:
            method()
        except Exception as error:
            failures.append(f"{method_name} 失败：{error}")
        else:
            succeeded.add(method_name)
    if failures:
        raise ControlCleanupError(failures, succeeded)
    return succeeded


@dataclass(frozen=True, slots=True)
class ServoMotionEnvelope:
    """相对使能起始位姿的软件包络与单周期增量限制。"""

    start_pose_ur: np.ndarray
    max_translation_m: float
    max_rotation_rad: float
    max_step_translation_m: float
    max_step_rotation_rad: float

    def __post_init__(self) -> None:
        pose = np.asarray(self.start_pose_ur, dtype=np.float64)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise ValueError("start_pose_ur 必须为有限六维向量")
        values = (
            self.max_translation_m,
            self.max_rotation_rad,
            self.max_step_translation_m,
            self.max_step_rotation_rad,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("运动包络必须为有限正数")
        copy = np.frombuffer(np.ascontiguousarray(pose).tobytes(), dtype=np.float64)
        object.__setattr__(self, "start_pose_ur", copy)


def _rotation_distance(first_rotvec: np.ndarray, second_rotvec: np.ndarray) -> float:
    # SciPy 的 Cython 入口要求可写内存；公共配置数组使用 bytes 后备存储，需在边界复制。
    first = Rotation.from_rotvec(np.array(first_rotvec, dtype=np.float64, copy=True))
    second = Rotation.from_rotvec(np.array(second_rotvec, dtype=np.float64, copy=True))
    return float((first.inv() * second).magnitude())


class UrRtdeServoSender:
    """每次 ``servoL`` 前执行 TCP、逆解、UR 安全和软件包络检查。"""

    def __init__(
        self,
        host: str,
        *,
        control_factory: Callable[[str], Any] | None = None,
        envelope: ServoMotionEnvelope,
        tcp_offset_provider: Callable[[], Any] | None = None,
        frequency_hz: float = 125.0,
        lookahead_time_s: float = 0.1,
        gain: float = 300.0,
        velocity: float = 0.05,
        acceleration: float = 0.25,
        stop_timeout_s: float = 0.05,
    ) -> None:
        numeric_values = (
            frequency_hz,
            velocity,
            acceleration,
            lookahead_time_s,
            gain,
            stop_timeout_s,
        )
        if any(isinstance(value, (bool, np.bool_)) for value in numeric_values):
            raise ValueError("伺服数值参数不能使用布尔值")
        if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise ValueError("frequency_hz 必须为有限正数；CB3 本系统配置为 125 Hz")
        if not np.isfinite(velocity) or velocity <= 0.0:
            raise ValueError("velocity 必须为有限正数")
        if not np.isfinite(acceleration) or acceleration <= 0.0:
            raise ValueError("acceleration 必须为有限正数")
        if not np.isfinite(lookahead_time_s) or not 0.03 <= lookahead_time_s <= 0.2:
            raise ValueError("lookahead_time_s 必须位于 [0.03, 0.2]")
        if not np.isfinite(gain) or not 100.0 <= gain <= 2000.0:
            raise ValueError("gain 必须位于 [100, 2000]")
        if not np.isfinite(stop_timeout_s) or stop_timeout_s <= 0.0:
            raise ValueError("stop_timeout_s 必须为有限正数")
        self.host = host
        self._factory = control_factory or self._default_factory
        self.envelope = envelope
        self._external_tcp_provider = tcp_offset_provider
        self._tcp_provider = tcp_offset_provider
        self.dt_s = 1.0 / float(frequency_hz)
        self.lookahead_time_s = float(lookahead_time_s)
        self.gain = float(gain)
        self.velocity = float(velocity)
        self.acceleration = float(acceleration)
        self.stop_timeout_s = float(stop_timeout_s)
        self._control: Any | None = None
        self._last_pose = np.array(envelope.start_pose_ur, copy=True)
        self._stopped = True
        self._authorized_state_readers: list[TcpAuthorizationTarget] = []
        # 命令锁只串行化 send_pose，不被 stop/disconnect 获取。因此即使厂商
        # TCP provider 卡死，生命周期停车仍可立即递增代次并撤销授信。
        self._command_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._lifecycle_condition = threading.Condition(self._lifecycle_lock)
        self._lifecycle_generation = 0
        self._connection_state = "DISCONNECTED"
        self._stop_commands_sent = False
        self._cleanup_succeeded: set[str] = set()
        self._cleanup_failures: list[str] = []
        self._cleanup_thread: threading.Thread | None = None
        self._active_operations = 0
        self._disconnect_requested = False

    @property
    def connection_state(self) -> str:
        with self._lifecycle_lock:
            return self._connection_state

    @staticmethod
    def _default_factory(host: str) -> Any:
        import rtde_control

        return rtde_control.RTDEControlInterface(host)

    def connect(self) -> None:
        with self._lifecycle_lock:
            if self._connection_state == "CONNECTED":
                return
            if self._connection_state == "CONNECTING":
                raise MotionCommandRejected("控制接口正在连接，不允许并发创建第二个接口")
            if self._connection_state in {"STOPPING", "STOP_UNCONFIRMED"}:
                raise MotionCommandRejected("上一次停车尚未确认，禁止重新连接")
            old_control = self._control
            old_completed = set(self._cleanup_succeeded)
            self._control = None
            self._revoke_state_reader_authorizations_locked()
            self._lifecycle_generation += 1
            generation = self._lifecycle_generation
            self._connection_state = "CONNECTING"
            self._stopped = True
            self._stop_commands_sent = False
            self._cleanup_succeeded = set()
            self._cleanup_failures = []
            self._disconnect_requested = False
        if old_control is not None:
            try:
                _cleanup_control(
                    old_control,
                    send_stop=not {"servoStop", "stopScript"}.issubset(old_completed),
                    disconnect=True,
                    completed=old_completed,
                )
            except ControlCleanupError as error:
                with self._lifecycle_lock:
                    self._control = old_control
                    self._cleanup_succeeded = set(error.succeeded)
                    self._connection_state = "STOP_UNCONFIRMED"
                raise MotionCommandRejected(
                    f"旧控制接口清理失败，禁止重连：{error}"
                ) from error

        control: Any | None = None
        try:
            control = self._factory(self.host)
            # factory 可能长时间阻塞。返回后先复核代次，若 stop/disconnect
            # 已生效，迟到对象只允许被清理，不再执行 TCP 读取等业务调用。
            with self._lifecycle_lock:
                factory_result_is_current = bool(
                    generation == self._lifecycle_generation
                    and self._connection_state == "CONNECTING"
                )
            if not factory_result_is_current:
                raise MotionCommandRejected(
                    "连接创建返回前生命周期已停止，拒绝使用迟到的控制接口"
                )
            missing = [
                name
                for name in _REQUIRED_CONTROL_METHODS
                if not callable(getattr(control, name, None))
            ]
            if missing:
                raise MotionCommandRejected(
                    "控制接口缺少必需安全 API：" + ", ".join(missing)
                )
            provider = self._external_tcp_provider or control.getTCPOffset
            if not _exact_zero_tcp(provider()):
                raise MotionCommandRejected("UR 活动 TCP 非零，禁止使能 servoL")
        except Exception as error:
            if control is not None:
                try:
                    _cleanup_control(control, send_stop=True, disconnect=True)
                except ControlCleanupError as cleanup_error:
                    with self._lifecycle_lock:
                        self._control = control
                        self._cleanup_succeeded = set(cleanup_error.succeeded)
                        self._connection_state = "STOP_UNCONFIRMED"
                    error = MotionCommandRejected(
                        f"控制接口连接失败且清理未完成：{error}；{cleanup_error}"
                    )
            with self._lifecycle_lock:
                if (
                    generation == self._lifecycle_generation
                    and self._connection_state == "CONNECTING"
                ):
                    self._lifecycle_generation += 1
                    self._connection_state = "DISCONNECTED"
                    self._stopped = True
            if isinstance(error, MotionCommandRejected):
                raise
            raise MotionCommandRejected(f"控制接口连接失败：{error}") from error
        with self._lifecycle_lock:
            can_commit = bool(
                generation == self._lifecycle_generation
                and self._connection_state == "CONNECTING"
            )
            if can_commit:
                self._control = control
                self._tcp_provider = provider
                self._connection_state = "CONNECTED"
                self._stopped = False
                self._stop_commands_sent = False
                self._cleanup_succeeded = set()
                self._cleanup_failures = []
                self._disconnect_requested = False
                self._last_pose = np.array(self.envelope.start_pose_ur, copy=True)
        if not can_commit:
            _cleanup_control(control, send_stop=True, disconnect=True)
            raise MotionCommandRejected("连接完成前生命周期已被停止，已清理迟到的控制接口")

    def _lifecycle_is_current_locked(self, generation: int, control: Any) -> bool:
        return bool(
            generation == self._lifecycle_generation
            and self._control is control
            and self._connection_state == "CONNECTED"
            and not self._stopped
        )

    def _stop_if_current(self, generation: int) -> None:
        with self._lifecycle_lock:
            current = generation == self._lifecycle_generation and not self._stopped
        if current:
            try:
                self.stop()
            except ControlCleanupError:
                # 调用者正在传播原始命令错误；停车失败状态保存在对象中，后续
                # orchestrator.stop/disconnect 会再次聚合并向 UI 报告。
                pass

    def _assert_operation_current(
        self, generation: int, control: Any, stage: str
    ) -> None:
        with self._lifecycle_lock:
            if not self._lifecycle_is_current_locked(generation, control):
                raise MotionCommandRejected(f"{stage}时控制生命周期已停止")

    def _begin_operation(self) -> tuple[Any, Callable[[], Any], int, np.ndarray]:
        with self._lifecycle_lock:
            if self._control is None or self._stopped or self._tcp_provider is None:
                raise MotionCommandRejected("servoL 控制接口尚未连接或已经停止")
            control = self._control
            provider = self._tcp_provider
            generation = self._lifecycle_generation
            last_pose = np.array(self._last_pose, copy=True)
            self._active_operations += 1
            return control, provider, generation, last_pose

    def _finish_operation(self) -> None:
        with self._lifecycle_condition:
            self._active_operations = max(0, self._active_operations - 1)
            self._maybe_finish_stop_locked()
            self._lifecycle_condition.notify_all()

    def _validate_pose(
        self,
        pose_ur: np.ndarray,
        *,
        control: Any,
        provider: Callable[[], Any],
        last_pose: np.ndarray,
        generation: int,
    ) -> np.ndarray:
        pose = np.asarray(pose_ur, dtype=np.float64)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise MotionCommandRejected("servoL 目标必须为有限六维 UR 位姿")
        self._assert_operation_current(generation, control, "读取 TCP 前")
        if not _exact_zero_tcp(provider()):
            raise MotionCommandRejected("UR 活动 TCP 非零，禁止发送 servoL")
        self._assert_operation_current(generation, control, "读取 TCP 后")
        translation_from_start = float(
            np.linalg.norm(pose[:3] - self.envelope.start_pose_ur[:3])
        )
        rotation_from_start = _rotation_distance(
            self.envelope.start_pose_ur[3:], pose[3:]
        )
        if translation_from_start > self.envelope.max_translation_m:
            raise MotionCommandRejected("目标超出相对起始位姿的平移包络")
        if rotation_from_start > self.envelope.max_rotation_rad:
            raise MotionCommandRejected("目标超出相对起始位姿的旋转包络")
        if np.linalg.norm(pose[:3] - last_pose[:3]) > self.envelope.max_step_translation_m:
            raise MotionCommandRejected("目标平移单周期增量过大")
        if _rotation_distance(last_pose[3:], pose[3:]) > self.envelope.max_step_rotation_rad:
            raise MotionCommandRejected("目标旋转单周期增量过大")
        for method_name, message in (
            ("isPoseWithinSafetyLimits", "UR 安全限制拒绝该目标位姿"),
            ("getInverseKinematicsHasSolution", "该目标位姿不存在逆解"),
        ):
            self._assert_operation_current(generation, control, f"{method_name} 前")
            method = getattr(control, method_name, None)
            if method is not None and not bool(method(pose.tolist())):
                raise MotionCommandRejected(message)
            self._assert_operation_current(generation, control, f"{method_name} 后")
        return np.array(pose, copy=True)

    def send_pose(self, pose_ur: np.ndarray) -> None:
        """串行化执行一条从周期开始到 ``last_pose`` 提交的完整命令。"""

        # 已停车时必须在等待上一条阻塞命令的 command lock 之前立即拒绝。
        with self._lifecycle_lock:
            if self._control is None or self._stopped:
                raise MotionCommandRejected("servoL 控制接口尚未连接或已经停止")
        with self._command_lock:
            self._send_pose_transaction(pose_ur)

    def _send_pose_transaction(self, pose_ur: np.ndarray) -> None:
        """命令锁内事务；所有厂商调用都在生命周期锁外运行。"""

        generation: int | None = None
        operation_started = False
        try:
            control, provider, generation, last_pose = self._begin_operation()
            operation_started = True
            self._assert_operation_current(generation, control, "initPeriod 前")
            period_token = (
                control.initPeriod() if hasattr(control, "initPeriod") else time.monotonic()
            )
            self._assert_operation_current(generation, control, "initPeriod 后")
            pose = self._validate_pose(
                pose_ur,
                control=control,
                provider=provider,
                last_pose=last_pose,
                generation=generation,
            )
            self._assert_operation_current(generation, control, "servoL 前")
            ok = control.servoL(
                pose.tolist(),
                self.velocity,
                self.acceleration,
                self.dt_s,
                self.lookahead_time_s,
                self.gain,
            )
            if ok is False:
                raise RuntimeError("servoL 返回失败")
            self._assert_operation_current(generation, control, "servoL 后")
            if hasattr(control, "waitPeriod"):
                self._assert_operation_current(generation, control, "waitPeriod 前")
                control.waitPeriod(period_token)
                self._assert_operation_current(generation, control, "waitPeriod 后")
            with self._lifecycle_lock:
                if not self._lifecycle_is_current_locked(generation, control):
                    raise MotionCommandRejected("提交命令结果时控制生命周期已停止")
                self._last_pose = pose
        except Exception:
            if generation is not None:
                self._stop_if_current(generation)
            raise
        finally:
            if operation_started:
                self._finish_operation()

    def authorize_state_reader(self, reader: TcpAuthorizationTarget) -> None:
        """把真实控制接口的 TCP=0 结果与一个状态读取器的生命周期绑定。

        读取器先被登记到撤销链，再读取 TCP；因此 provider、严格零校验或确认调用任一失败，
        都会停车并把读取器恢复为未授权状态。重复绑定同一对象不会注册重复回调。
        """

        operation_started = False
        with self._lifecycle_lock:
            if self._control is None or self._stopped or self._tcp_provider is None:
                try:
                    reader.revoke_tcp_confirmation()
                except Exception:
                    pass
                raise MotionCommandRejected("控制接口未连接，不能授予法兰位姿语义")
            control = self._control
            provider = self._tcp_provider
            generation = self._lifecycle_generation
            if not any(target is reader for target in self._authorized_state_readers):
                self._authorized_state_readers.append(reader)
            # 在 provider 可能阻塞之前先撤销旧授信，并登记到当前代次的撤销链。
            try:
                reader.revoke_tcp_confirmation()
            except Exception as error:
                self.stop()
                raise MotionCommandRejected(f"撤销旧 TCP 授信失败：{error}") from error
            # provider 会进入厂商 control 对象。必须在释放锁前登记，
            # disconnect 才能等它真正返回后再释放连接。
            self._active_operations += 1
            operation_started = True
        provider_error: Exception | None = None
        value: np.ndarray | None = None
        try:
            # 不能在这次可能阻塞的厂商调用期间持生命周期锁，否则 stop 无法取得优先权。
            value = np.asarray(provider(), dtype=np.float64)
            if not _exact_zero_tcp(value):
                raise MotionCommandRejected("UR 活动 TCP 非零，不能授予法兰位姿语义")
        except Exception as error:
            provider_error = error
        finally:
            if operation_started:
                self._finish_operation()
        if provider_error is not None:
            with self._lifecycle_lock:
                try:
                    reader.revoke_tcp_confirmation()
                except Exception:
                    pass
            self._stop_if_current(generation)
            if isinstance(provider_error, MotionCommandRejected):
                raise provider_error
            raise MotionCommandRejected(
                f"TCP 零值授权失败：{provider_error}"
            ) from provider_error
        assert value is not None
        with self._lifecycle_lock:
            still_registered = any(
                target is reader for target in self._authorized_state_readers
            )
            if not self._lifecycle_is_current_locked(generation, control) or not still_registered:
                try:
                    reader.revoke_tcp_confirmation()
                except Exception:
                    pass
                raise MotionCommandRejected("TCP 授权返回时控制生命周期已经停止")
            # confirm 必须和代次复核处于同一锁内；stop 只能在确认后进入并立即撤销。
            try:
                reader.confirm_zero_tcp(value)
            except Exception as error:
                try:
                    reader.revoke_tcp_confirmation()
                except Exception:
                    pass
                self.stop()
                raise MotionCommandRejected(f"状态读取器拒绝 TCP 授权：{error}") from error

    def verified_tcp_offset(self) -> np.ndarray:
        """读取并返回已验证的零 TCP，供只读状态线程显式确认法兰语义。"""

        with self._lifecycle_lock:
            if self._control is None or self._stopped or self._tcp_provider is None:
                raise MotionCommandRejected("控制接口未连接，无法验证活动 TCP")
            control = self._control
            provider = self._tcp_provider
            generation = self._lifecycle_generation
            self._active_operations += 1
        provider_error: Exception | None = None
        value: np.ndarray | None = None
        try:
            value = np.asarray(provider(), dtype=np.float64)
            if not _exact_zero_tcp(value):
                raise MotionCommandRejected("UR 活动 TCP 非零，撤销法兰位姿授信")
        except Exception as error:
            provider_error = error
        finally:
            self._finish_operation()
        if provider_error is not None:
            self._stop_if_current(generation)
            if isinstance(provider_error, MotionCommandRejected):
                raise provider_error
            raise MotionCommandRejected(
                f"读取活动 TCP 失败：{provider_error}"
            ) from provider_error
        assert value is not None
        with self._lifecycle_lock:
            if not self._lifecycle_is_current_locked(generation, control):
                raise MotionCommandRejected("读取 TCP 期间控制生命周期已经停止")
        return np.array(value, copy=True)

    def _revoke_state_reader_authorizations_locked(self) -> None:
        targets, self._authorized_state_readers = self._authorized_state_readers, []
        for target in targets:
            try:
                target.revoke_tcp_confirmation()
            except Exception:
                # 撤销链中的单个观察者不能阻止机器人本体继续停车。
                pass

    def _maybe_finish_stop_locked(self) -> None:
        stop_complete = {"servoStop", "stopScript"}.issubset(
            self._cleanup_succeeded
        )
        if stop_complete and self._active_operations == 0:
            self._stop_commands_sent = True
            if self._disconnect_requested:
                if "disconnect" in self._cleanup_succeeded:
                    self._connection_state = "DISCONNECTED"
                # 释放已申请但尚未完成时，不得降级成 STOPPED。
                # STOP_UNCONFIRMED 会保留到后台 disconnect 真正成功。
                elif self._connection_state != "STOP_UNCONFIRMED":
                    self._connection_state = "STOPPING"
            else:
                self._connection_state = "STOPPED"

    def _run_stop_cleanup(self, control: Any) -> None:
        failures: list[str] = []
        succeeded: set[str] = set()
        with self._lifecycle_lock:
            completed = set(self._cleanup_succeeded)
        try:
            succeeded = _cleanup_control(
                control,
                send_stop=True,
                disconnect=False,
                completed=completed,
            )
        except ControlCleanupError as error:
            succeeded = set(error.succeeded)
            failures.extend(error.failures)
        # servoStop/stopScript 允许为紧急制动与在途调用并发；
        # disconnect 会销毁 vendor control，必须等所有在途调用退出。
        with self._lifecycle_condition:
            if self._control is control:
                self._cleanup_succeeded.update(succeeded)
                self._cleanup_failures = list(failures)
                self._stop_commands_sent = {
                    "servoStop",
                    "stopScript",
                }.issubset(self._cleanup_succeeded)
                self._maybe_finish_stop_locked()
                self._lifecycle_condition.notify_all()
            while (
                self._control is control
                and self._disconnect_requested
                and self._active_operations > 0
            ):
                self._lifecycle_condition.wait()
            disconnect_requested = bool(
                self._control is control and self._disconnect_requested
            )
            completed = set(self._cleanup_succeeded)
        if disconnect_requested:
            try:
                succeeded = _cleanup_control(
                    control,
                    send_stop=False,
                    disconnect=True,
                    completed=completed,
                )
            except ControlCleanupError as error:
                succeeded = set(error.succeeded)
                failures.extend(error.failures)
        with self._lifecycle_condition:
            if self._control is control:
                self._cleanup_succeeded.update(succeeded)
                self._cleanup_failures = failures
                self._stop_commands_sent = {
                    "servoStop",
                    "stopScript",
                }.issubset(self._cleanup_succeeded)
                if "disconnect" in self._cleanup_succeeded:
                    self._control = None
                    if self._external_tcp_provider is None:
                        self._tcp_provider = None
                self._maybe_finish_stop_locked()
                if failures and self._connection_state != "STOPPED":
                    self._connection_state = "STOP_UNCONFIRMED"
            self._cleanup_thread = None
            self._lifecycle_condition.notify_all()

    def _start_stop_cleanup_locked(self, control: Any) -> None:
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            return
        self._cleanup_failures = []
        worker = threading.Thread(
            target=self._run_stop_cleanup,
            args=(control,),
            name="ur-stop-cleanup",
            daemon=True,
        )
        self._cleanup_thread = worker
        worker.start()

    def stop(self) -> None:
        deadline = time.monotonic() + self.stop_timeout_s
        with self._lifecycle_condition:
            # 先使所有在途授权/发送 token 失效，再启动独立清理线程。任何厂商
            # 调用都不持生命周期锁，因此 UI 停车入口不会被 waitPeriod 锁死。
            self._lifecycle_generation += 1
            self._revoke_state_reader_authorizations_locked()
            self._stopped = True
            control = self._control
            if control is None:
                self._connection_state = "STOPPED"
                return
            self._connection_state = "STOPPING"
            self._start_stop_cleanup_locked(control)
            while True:
                self._maybe_finish_stop_locked()
                if self._connection_state in {"STOPPED", "DISCONNECTED"}:
                    return
                if self._cleanup_thread is None and self._cleanup_failures:
                    self._connection_state = "STOP_UNCONFIRMED"
                    raise ControlCleanupError(
                        list(self._cleanup_failures), set(self._cleanup_succeeded)
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._connection_state = "STOP_UNCONFIRMED"
                    raise RobotStopTimeout(
                        [
                            "软件停车未能在时限内确认；厂商调用仍在进行，"
                            "请立即使用物理急停"
                        ],
                        set(self._cleanup_succeeded),
                    )
                self._lifecycle_condition.wait(min(remaining, 0.01))

    def disconnect(self) -> None:
        """可靠停车后显式释放控制连接，并自动撤销所有状态读取器授权。"""

        failures: list[str] = []
        try:
            self.stop()
        except ControlCleanupError as error:
            failures.extend(error.failures)
        deadline = time.monotonic() + self.stop_timeout_s
        with self._lifecycle_condition:
            self._lifecycle_generation += 1
            self._revoke_state_reader_authorizations_locked()
            self._stopped = True
            control = self._control
            self._disconnect_requested = True
            # stop 的失败已复制到局部 failures；disconnect 仍需启动一次独立
            # best-effort 清理，继续尝试其余步骤而不是被旧错误提前短路。
            self._cleanup_failures = []
            if control is None and self._active_operations == 0:
                self._connection_state = "DISCONNECTED"
                if failures:
                    raise ControlCleanupError(
                        list(dict.fromkeys(failures)), set(self._cleanup_succeeded)
                    )
                return
            while True:
                self._maybe_finish_stop_locked()
                if self._connection_state == "DISCONNECTED":
                    if failures:
                        raise ControlCleanupError(
                            list(dict.fromkeys(failures)),
                            set(self._cleanup_succeeded),
                        )
                    return
                if self._cleanup_thread is None and self._cleanup_failures:
                    failures.extend(self._cleanup_failures)
                    self._connection_state = "STOP_UNCONFIRMED"
                    raise ControlCleanupError(
                        list(dict.fromkeys(failures)), set(self._cleanup_succeeded)
                    )
                if self._cleanup_thread is None and self._control is not None:
                    self._start_stop_cleanup_locked(self._control)
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    failures.append(
                        "控制接口释放未能在时限内确认；请立即使用物理急停"
                    )
                    self._connection_state = "STOP_UNCONFIRMED"
                    raise RobotStopTimeout(
                        list(dict.fromkeys(failures)), set(self._cleanup_succeeded)
                    )
                self._lifecycle_condition.wait(min(remaining, 0.01))

    def __enter__(self) -> UrRtdeServoSender:
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        # with 作用域结束表示本次使用生命周期结束：先停车，再释放 RTDE 连接。
        self.disconnect()


@dataclass(slots=True)
class FakeRobotBackend:
    """离线测试/干运行后端；方法名与 ur_rtde 控制/接收接口保持一致。"""

    actual_tcp_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    actual_tcp_speed: np.ndarray = field(default_factory=lambda: np.zeros(6))
    actual_q: np.ndarray = field(default_factory=lambda: np.zeros(6))
    tcp_offset: np.ndarray = field(default_factory=lambda: np.zeros(6))
    inverse_kinematics_available: bool = True
    pose_within_safety_limits: bool = True
    fail_servo_l: bool = False
    fail_servo_stop: bool = False
    fail_stop_script: bool = False
    fail_disconnect: bool = False
    fail_wait_period: bool = False
    fail_tcp_offset: bool = False
    servo_commands: list[list[float]] = field(default_factory=list)
    period_wait_count: int = 0
    servo_stop_count: int = 0
    stop_script_count: int = 0
    disconnect_count: int = 0
    call_log: list[str] = field(default_factory=list)

    def initPeriod(self) -> object:
        self.call_log.append("initPeriod")
        return object()

    def waitPeriod(self, _token: object) -> None:
        self.call_log.append("waitPeriod")
        if self.fail_wait_period:
            raise RuntimeError("waitPeriod simulated failure")
        self.period_wait_count += 1

    def servoL(self, pose, *_args) -> bool:
        self.call_log.append("servoL")
        if self.fail_servo_l:
            raise RuntimeError("servoL simulated failure")
        self.servo_commands.append(list(pose))
        self.actual_tcp_pose = np.asarray(pose, dtype=np.float64)
        return True

    def servoStop(self) -> None:
        self.servo_stop_count += 1
        if self.fail_servo_stop:
            raise RuntimeError("servoStop simulated failure")

    def stopScript(self) -> None:
        self.stop_script_count += 1
        if self.fail_stop_script:
            raise RuntimeError("stopScript simulated failure")

    def isPoseWithinSafetyLimits(self, _pose) -> bool:
        self.call_log.append("safety")
        return self.pose_within_safety_limits

    def getInverseKinematicsHasSolution(self, _pose) -> bool:
        self.call_log.append("ik")
        return self.inverse_kinematics_available

    def getTCPOffset(self):
        self.call_log.append("tcp")
        if self.fail_tcp_offset:
            raise RuntimeError("getTCPOffset simulated failure")
        return self.tcp_offset.tolist()

    def getActualTCPPose(self):
        return self.actual_tcp_pose.tolist()

    def getActualTCPSpeed(self):
        return self.actual_tcp_speed.tolist()

    def getActualQ(self):
        return self.actual_q.tolist()

    def getRobotMode(self) -> int:
        return 7

    def getSafetyMode(self) -> int:
        return 1

    def getRuntimeState(self) -> int:
        return 2

    def disconnect(self) -> None:
        self.disconnect_count += 1
        if self.fail_disconnect:
            raise RuntimeError("disconnect simulated failure")

