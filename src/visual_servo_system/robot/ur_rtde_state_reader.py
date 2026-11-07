"""UR RTDE 只读状态采集与法兰位姿历史。"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from visual_servo_system.geometry.pose_transforms import ur_pose_to_transform
from visual_servo_system.geometry.robot_pose_history import RobotPoseHistory
from visual_servo_system.models.data_types import RobotStateSample
from visual_servo_system.runtime.device_clock_mapper import (
    ClockSampleStatus,
    DeviceClockMapper,
)
from visual_servo_system.runtime.latest_value_buffer import LatestValueBuffer


class RobotConnectionError(RuntimeError):
    """RTDE 连接或安全前提不满足。"""


class RobotShutdownTimeout(RuntimeError):
    """RTDE C 扩展读取仍阻塞，不能从另一线程并发断开。"""


@dataclass(frozen=True, slots=True)
class RtdeSampleDiagnostics:
    """控制器状态包分类计数，用于离线审计和界面诊断。"""

    received: int
    accepted: int
    duplicate: int
    delayed: int
    out_of_order: int
    clock_reset: int
    missed_deadlines: int


def _exact_zero_tcp(value: Any) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.shape == (6,) and np.all(np.isfinite(array)) and np.all(array == 0.0))


class UrRtdeStateReader:
    """125 Hz 读取 UR 状态；本类从不创建控制接口、从不发送运动。"""

    def __init__(
        self,
        host: str,
        *,
        frequency_hz: float = 125.0,
        receive_factory: Callable[[str, float], Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        history_capacity: int = 500,
        max_extrapolation_s: float = 0.05,
        read_timeout_s: float = 0.1,
        watchdog_check_period_s: float = 0.01,
        clock_mapper: DeviceClockMapper | None = None,
        performance_profiler: Any | None = None,
        perf_counter_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        timing_values = (frequency_hz, read_timeout_s, watchdog_check_period_s)
        if any(isinstance(value, (bool, np.bool_)) for value in timing_values) or not all(
            np.isfinite(value) and value > 0.0 for value in timing_values
        ):
            raise ValueError("frequency_hz、read_timeout_s 和 watchdog 周期必须为有限正数")
        if watchdog_check_period_s > read_timeout_s:
            raise ValueError("watchdog_check_period_s 不得大于 read_timeout_s")
        self.host = host
        self.frequency_hz = float(frequency_hz)
        self._factory = receive_factory or self._default_factory
        self._clock = monotonic
        self._read_timeout_s = float(read_timeout_s)
        self._watchdog_check_period_s = float(watchdog_check_period_s)
        self._clock_mapper = clock_mapper or DeviceClockMapper()
        self.performance_profiler = performance_profiler
        self._perf_counter_ns = perf_counter_ns
        self.pose_history = RobotPoseHistory(history_capacity, max_extrapolation_s)
        self.latest_state: LatestValueBuffer[RobotStateSample] = LatestValueBuffer()
        self.last_error: Exception | None = None
        self._receive: Any | None = None
        self._sequence = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self.watchdog_fault_event = threading.Event()
        self._status_lock = threading.Lock()
        self._last_success_s: float | None = None
        self._received_samples = 0
        self._accepted_samples = 0
        self._duplicate_samples = 0
        self._delayed_samples = 0
        self._out_of_order_samples = 0
        self._clock_reset_samples = 0
        self._missed_deadlines = 0
        self._tcp_zero_confirmed = False
        # confirm_zero_tcp 后先关闭历史入口；由上层在获得授权完成
        # 的主机时刻后显式开启新代次，避免授权途中样本混入。
        self._pose_epoch_not_before_s = float("inf")
        self.last_actual_tcp_pose_ur: np.ndarray | None = None

    def _record_performance(
        self,
        stage: str,
        kind: str,
        duration_ns: int,
        *,
        sequence_number: int | None = None,
        **details: Any,
    ) -> None:
        if self.performance_profiler is None:
            return
        self.performance_profiler.record(
            source="robot",
            stage=stage,
            kind=kind,
            duration_ns=int(max(0, duration_ns)),
            sequence_number=sequence_number,
            **details,
        )

    @staticmethod
    def _default_factory(host: str, frequency_hz: float) -> Any:
        import rtde_receive

        return rtde_receive.RTDEReceiveInterface(host, frequency_hz)

    def connect(self) -> None:
        if self._receive is not None:
            return
        try:
            receive = self._factory(self.host, self.frequency_hz)
        except RobotConnectionError:
            raise
        except Exception as error:
            raise RobotConnectionError(f"RTDE 只读连接失败：{error}") from error
        self._receive = receive
        self._sequence = 0

    def confirm_zero_tcp(self, tcp_offset_ur: Any) -> None:
        """接收控制接口 ``getTCPOffset()`` 的结果，显式授信 TCP=0。

        ``RTDEReceiveInterface`` 1.6.5 没有这个 getter，因此严禁只读线程自行猜测。调用者
        必须从 ``RTDEControlInterface`` 读取并传入；断连或任何控制侧异常应调用撤销方法。
        """

        if not _exact_zero_tcp(tcp_offset_ur):
            with self._status_lock:
                self._tcp_zero_confirmed = False
                self._pose_epoch_not_before_s = float("inf")
                self.pose_history.clear()
            raise RobotConnectionError("控制接口报告的活动 TCP 不是六维精确零值")
        with self._status_lock:
            self._tcp_zero_confirmed = True
            self._pose_epoch_not_before_s = float("inf")
            self.pose_history.clear()

    def begin_authorized_pose_epoch(self, not_before_host_s: float) -> None:
        """清除授权途中样本，只接受不早于指定时刻的可靠状态。"""

        threshold = float(not_before_host_s)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("not_before_host_s must be finite and non-negative")
        with self._status_lock:
            if not self._tcp_zero_confirmed:
                raise RobotConnectionError("未确认 TCP=0，不能开始法兰位姿代次")
            self._pose_epoch_not_before_s = threshold
            self.pose_history.clear()

    def revoke_tcp_confirmation(self) -> None:
        """撤销 TCP=0 授信；之后的读数不再写入法兰位姿历史。"""

        with self._status_lock:
            self._tcp_zero_confirmed = False
            self._pose_epoch_not_before_s = float("inf")
            self.pose_history.clear()

    @property
    def tcp_zero_confirmed(self) -> bool:
        with self._status_lock:
            return self._tcp_zero_confirmed

    @property
    def connected(self) -> bool:
        return self._receive is not None

    @property
    def sample_diagnostics(self) -> RtdeSampleDiagnostics:
        with self._status_lock:
            return RtdeSampleDiagnostics(
                received=self._received_samples,
                accepted=self._accepted_samples,
                duplicate=self._duplicate_samples,
                delayed=self._delayed_samples,
                out_of_order=self._out_of_order_samples,
                clock_reset=self._clock_reset_samples,
                missed_deadlines=self._missed_deadlines,
            )

    def read_once(self) -> RobotStateSample | None:
        cycle = getattr(self.performance_profiler, "cycle", None)
        if callable(cycle):
            with cycle():
                return self._read_once_profiled()
        return self._read_once_profiled()

    def _read_once_profiled(self) -> RobotStateSample | None:
        if self._receive is None:
            raise RobotConnectionError("RTDE 只读接口尚未连接")
        started = self._clock()
        getters_started_ns = self._perf_counter_ns()
        try:
            controller_timestamp_getter = getattr(self._receive, "getTimestamp", None)
            controller_timestamp_s = (
                float(controller_timestamp_getter())
                if callable(controller_timestamp_getter)
                else None
            )
            pose = np.asarray(self._receive.getActualTCPPose(), dtype=np.float64)
            speed = np.asarray(self._receive.getActualTCPSpeed(), dtype=np.float64)
            joints = np.asarray(self._receive.getActualQ(), dtype=np.float64)
            robot_mode = self._receive.getRobotMode()
            safety_mode = self._receive.getSafetyMode()
            runtime_state = self._receive.getRuntimeState()
            receive_timestamp = float(self._clock())
        except RobotConnectionError:
            raise
        except Exception as error:
            raise RobotConnectionError(f"RTDE 状态读取失败：{error}") from error
        with self._status_lock:
            self._received_samples += 1
        self._record_performance(
            "rtde_getters",
            "io",
            self._perf_counter_ns() - getters_started_ns,
            sequence_number=self._sequence,
        )
        read_duration_s = receive_timestamp - started
        if read_duration_s > self._read_timeout_s:
            raise RobotConnectionError("RTDE 状态读取超时")
        midpoint_timestamp = (started + receive_timestamp) * 0.5
        mapping_started_ns = self._perf_counter_ns()
        residual_rms_s = float("nan")
        clock_sample_count = self._clock_mapper.sample_count
        clock_reset_count = self._clock_mapper.reset_count
        clock_sample_status = "host_only"
        reject_repeated_state = False
        if controller_timestamp_s is None:
            timestamp = midpoint_timestamp
            timestamp_source = "host_midpoint"
            timestamp_reliable = False
            timestamp_uncertainty_s = max(0.0, read_duration_s * 0.5)
        else:
            mapping = self._clock_mapper.observe(
                controller_timestamp_s, midpoint_timestamp
            )
            clock_sample_status = mapping.status.value
            timestamp = mapping.host_timestamp_s
            timestamp_source = "ur_controller"
            timestamp_uncertainty_s = mapping.uncertainty_s + max(
                0.0, read_duration_s * 0.5
            )
            timestamp_reliable = bool(
                mapping.reliable
                and timestamp_uncertainty_s <= self._clock_mapper.max_uncertainty_s
            )
            residual_rms_s = mapping.residual_rms_s
            clock_sample_count = mapping.sample_count
            clock_reset_count = mapping.reset_count
            with self._status_lock:
                if mapping.status is ClockSampleStatus.DUPLICATE:
                    self._duplicate_samples += 1
                    reject_repeated_state = True
                elif mapping.status is ClockSampleStatus.OUT_OF_ORDER:
                    self._out_of_order_samples += 1
                    reject_repeated_state = True
                elif mapping.status is ClockSampleStatus.DELAYED:
                    self._delayed_samples += 1
                elif mapping.status is ClockSampleStatus.RESET:
                    self._clock_reset_samples += 1
        self._record_performance(
            "timestamp_mapping",
            "compute",
            self._perf_counter_ns() - mapping_started_ns,
            sequence_number=self._sequence,
            frame_host_timestamp_s=timestamp,
            receive_timestamp_s=receive_timestamp,
            timestamp_source=timestamp_source,
            timestamp_reliable=timestamp_reliable,
            timestamp_uncertainty_ms=timestamp_uncertainty_s * 1000.0,
            mapped_transport_delay_ms=max(0.0, receive_timestamp - timestamp) * 1000.0,
            clock_residual_rms_ms=residual_rms_s * 1000.0,
            clock_sample_count=clock_sample_count,
            clock_reset_count=clock_reset_count,
            detail=clock_sample_status,
        )
        if reject_repeated_state:
            # 读到同一控制器包或乱序旧包时，不增加序号、不更新
            # latest_state，也不刷新看门狗的“新状态成功”时刻。
            return None
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise RobotConnectionError("RTDE 返回的 actual_TCP_pose 不是有限六维向量")
        self.last_actual_tcp_pose_ur = np.array(pose, copy=True)
        sample_started_ns = self._perf_counter_ns()
        with self._status_lock:
            tcp_confirmed = self._tcp_zero_confirmed
            pose_epoch_not_before_s = self._pose_epoch_not_before_s
            sample = RobotStateSample(
                timestamp,
                ur_pose_to_transform(pose) if tcp_confirmed else np.eye(4),
                speed,
                joints,
                self._sequence,
                (
                    f"robot={robot_mode};safety={safety_mode};runtime={runtime_state};"
                    f"tcp={'zero_confirmed' if tcp_confirmed else 'unverified'}"
                ),
                bool(tcp_confirmed),
                1.0 if tcp_confirmed else 0.0,
                receive_timestamp_s=receive_timestamp,
                timestamp_source=timestamp_source,
                timestamp_reliable=timestamp_reliable,
                timestamp_uncertainty_s=float(timestamp_uncertainty_s),
                controller_timestamp_s=controller_timestamp_s,
                sample_identity=(
                    f"ur-controller:{controller_timestamp_s:.9f}"
                    if controller_timestamp_s is not None
                    else f"host-read:{self._sequence}"
                ),
                clock_sample_status=clock_sample_status,
            )
            self._sequence += 1
            self._accepted_samples += 1
            self._last_success_s = receive_timestamp
            if (
                sample.valid
                and sample.timestamp_reliable
                and sample.host_timestamp_s >= pose_epoch_not_before_s
                and sample.receive_timestamp_s >= pose_epoch_not_before_s
            ):
                self.pose_history.add(sample)
        self._record_performance(
            "state_sample_construction",
            "compute",
            self._perf_counter_ns() - sample_started_ns,
            sequence_number=sample.sequence_number,
        )
        publish_started_ns = self._perf_counter_ns()
        self.latest_state.publish(sample)
        self._record_performance(
            "latest_state_publish",
            "lock",
            self._perf_counter_ns() - publish_started_ns,
            sequence_number=sample.sequence_number,
        )
        return sample

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.connect()
        self._stop.clear()
        self.watchdog_fault_event.clear()
        self.last_error = None
        with self._status_lock:
            self._last_success_s = self._clock()
        self._thread = threading.Thread(target=self._run, name="ur-rtde-state", daemon=True)
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_run, name="ur-rtde-watchdog", daemon=True
        )
        self._thread.start()
        self._watchdog_thread.start()

    def _run(self) -> None:
        period = 1.0 / self.frequency_hz
        deadline = self._clock()
        try:
            while not self._stop.is_set():
                self.read_once()
                deadline += period
                now = self._clock()
                if now >= deadline:
                    # 读取超期后从“现在”重建下一个节拍，禁止用连续
                    # 零等待轮询追赶旧 deadline，否则会重复读取同一 RTDE 包。
                    missed = max(1, int((now - deadline) // period) + 1)
                    with self._status_lock:
                        self._missed_deadlines += missed
                    deadline = now + period
                wait_started_ns = self._perf_counter_ns()
                self._stop.wait(max(0.0, deadline - now))
                self._record_performance(
                    "rate_limit_wait",
                    "wait",
                    self._perf_counter_ns() - wait_started_ns,
                )
        except Exception as error:
            self.last_error = error
            self.watchdog_fault_event.set()
            self.revoke_tcp_confirmation()
            self._stop.set()
        finally:
            self._disconnect_from_reader_thread()

    def _watchdog_run(self) -> None:
        """独立看门狗：即使 C getter 卡死，也会在期限后把系统置为故障。"""

        while not self._stop.wait(self._watchdog_check_period_s):
            with self._status_lock:
                last_success = self._last_success_s
            now = self._clock()
            if last_success is None or now - last_success > self._read_timeout_s:
                self.last_error = RobotConnectionError(
                    "RTDE 超过 last-success 时限；读取线程可能阻塞，必须立即停车"
                )
                self.watchdog_fault_event.set()
                self.revoke_tcp_confirmation()
                self._stop.set()
                return

    @property
    def communication_healthy(self) -> bool:
        """只有连接存在、看门狗未故障且最近一次成功读取未过期才为真。"""

        with self._status_lock:
            last_success = self._last_success_s
        if self._receive is None or self.watchdog_fault_event.is_set() or last_success is None:
            return False
        return bool(self._clock() - last_success <= self._read_timeout_s)

    def _disconnect_from_reader_thread(self) -> None:
        receive, self._receive = self._receive, None
        if receive is not None:
            try:
                receive.disconnect()
            except Exception:
                pass

    def close(self) -> None:
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            raise RobotShutdownTimeout(
                "RTDE 读取线程仍在运行，拒绝从另一线程并发 disconnect"
            )
        self._stop.set()
        self._disconnect_from_reader_thread()
        self.revoke_tcp_confirmation()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)
        if thread is not None and thread.is_alive():
            raise RobotShutdownTimeout(
                "RTDE 读取线程仍在运行，底层连接保持打开；上层必须故障停车并稍后重试"
            )
        watchdog = self._watchdog_thread
        if watchdog is not None and watchdog is not threading.current_thread():
            watchdog.join(timeout_s)
        self.close()

    def __enter__(self) -> UrRtdeStateReader:
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()

