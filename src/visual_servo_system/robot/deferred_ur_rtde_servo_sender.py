"""在使能时根据实际法兰位姿建立软件运动包络。"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from visual_servo_system.geometry.pose_transforms import transform_to_ur_pose
from visual_servo_system.robot.ur_rtde_servo_sender import (
    ServoMotionEnvelope,
    UrRtdeServoSender,
)


class DeferredUrRtdeServoSender:
    """延迟创建 RTDEControlInterface，防止程序启动就连接控制器。"""

    def __init__(
        self,
        host: str,
        state_reader: Any,
        *,
        sender_factory: Callable[..., Any] = UrRtdeServoSender,
    ) -> None:
        self.host = host
        self.state_reader = state_reader
        self._sender_factory = sender_factory
        self._lifecycle_lock = threading.RLock()
        self._lifecycle_generation = 0
        self._connection_state = "DISCONNECTED"
        self._sender: Any | None = None
        self._pending_sender: Any | None = None
        # stop 会摘掉 pending 引用以阻止业务使用，但 connect 调用真正返回前仍在这里
        # 留痕。shutdown 据此明确报告厂商连接线程尚未退出，不能假报关闭成功。
        self._inflight_connections: dict[int, Any] = {}

    @property
    def connection_state(self) -> str:
        """返回当前外层连接状态，供状态界面和关闭诊断读取。"""

        with self._lifecycle_lock:
            return self._connection_state

    @property
    def pending_connection_count(self) -> int:
        """返回尚未从 ``inner.connect`` 返回的调用数量。"""

        with self._lifecycle_lock:
            return len(self._inflight_connections)

    def connect(self) -> None:
        # 构造、登记 pending 和切换 CONNECTING 必须处在同一把锁内。这样
        # inner.connect 一旦在厂商工厂中阻塞，另一个线程仍能找到 inner 并停车。
        with self._lifecycle_lock:
            if self._connection_state == "CONNECTED" and self._sender is not None:
                return
            if self._connection_state == "CONNECTING" or self._inflight_connections:
                raise RuntimeError("servo sender 正在连接，不允许并发创建控制接口")
            latest = self.state_reader.latest_state.get_latest()
            if latest is None:
                raise RuntimeError("无 UR5 当前位姿，不能建立运动包络")
            raw_tcp_pose = getattr(self.state_reader, "last_actual_tcp_pose_ur", None)
            if raw_tcp_pose is not None:
                # TCP 是否为零紧接着由 inner.connect 的 getTCPOffset 确认。
                # 在此之前不把 raw pose 公布为法兰历史，只用来初始化包络。
                pose_ur = np.asarray(raw_tcp_pose, dtype=np.float64)
                if pose_ur.shape != (6,) or not np.all(np.isfinite(pose_ur)):
                    raise RuntimeError("UR5 原始 actual_TCP_pose 无效")
            else:
                pose_ur = transform_to_ur_pose(latest[1].T_base_flange)
            envelope = ServoMotionEnvelope(
                start_pose_ur=pose_ur,
                max_translation_m=0.100,
                max_rotation_rad=np.deg2rad(15.0),
                max_step_translation_m=0.0012,
                max_step_rotation_rad=np.deg2rad(0.30),
            )
            self._lifecycle_generation += 1
            generation = self._lifecycle_generation
            self._connection_state = "CONNECTING"
            try:
                sender = self._sender_factory(
                    self.host,
                    envelope=envelope,
                    frequency_hz=125.0,
                    lookahead_time_s=0.1,
                    gain=300.0,
                    velocity=0.05,
                    acceleration=0.25,
                )
            except Exception:
                self._lifecycle_generation += 1
                self._connection_state = "DISCONNECTED"
                raise
            self._pending_sender = sender
            self._inflight_connections[generation] = sender

        try:
            sender.connect()
        except Exception:
            with self._lifecycle_lock:
                self._inflight_connections.pop(generation, None)
                if (
                    generation == self._lifecycle_generation
                    and self._pending_sender is sender
                    and self._connection_state == "CONNECTING"
                ):
                    self._lifecycle_generation += 1
                    self._pending_sender = None
                    self._connection_state = "DISCONNECTED"
            # inner.connect 应自行清理部分建立的对象；这里再做一次幂等释放，
            # 覆盖第三方 sender 没有完整异常清理的情况。
            self._disconnect_senders((sender,))
            raise

        with self._lifecycle_lock:
            self._inflight_connections.pop(generation, None)
            can_commit = bool(
                generation == self._lifecycle_generation
                and self._pending_sender is sender
                and self._connection_state == "CONNECTING"
            )
            if can_commit:
                self._pending_sender = None
                self._sender = sender
                self._connection_state = "CONNECTED"
        if not can_commit:
            self._disconnect_senders((sender,))
            raise RuntimeError("servo sender 连接返回前生命周期已停止，已清理迟到对象")

    def _require(self) -> Any:
        with self._lifecycle_lock:
            if self._sender is None or self._connection_state != "CONNECTED":
                raise RuntimeError("servo sender is not connected")
            return self._sender

    def authorize_state_reader(self, reader: Any) -> None:
        self._require().authorize_state_reader(reader)

    def send_pose(self, pose_ur: np.ndarray) -> None:
        self._require().send_pose(pose_ur)

    def stop(self) -> None:
        # 一次停车就结束本代控制连接。锁内先让业务引用和代次失效，锁外再调
        # 厂商对象，避免慢速 stop 阻塞其他生命周期观察者。
        with self._lifecycle_lock:
            self._lifecycle_generation += 1
            active = self._sender
            pending = self._pending_sender
            senders = self._unique_senders(active, pending)
            self._connection_state = "STOPPING"
        failures, failed_senders = self._stop_senders(senders)
        with self._lifecycle_lock:
            failed_ids = {id(sender) for sender in failed_senders}
            self._sender = active if active is not None and id(active) in failed_ids else None
            self._pending_sender = (
                pending if pending is not None and id(pending) in failed_ids else None
            )
            self._connection_state = "STOP_UNCONFIRMED" if failures else "STOPPED"
        if failures:
            raise RuntimeError("；".join(failures))

    def disconnect(self) -> None:
        with self._lifecycle_lock:
            self._lifecycle_generation += 1
            active = self._sender
            pending = self._pending_sender
            senders = self._unique_senders(
                active,
                pending,
                *self._inflight_connections.values(),
            )
            self._connection_state = "DISCONNECTING"
        failures, failed_senders = self._disconnect_senders(senders)
        with self._lifecycle_lock:
            failed_ids = {id(sender) for sender in failed_senders}
            self._sender = active if active is not None and id(active) in failed_ids else None
            self._pending_sender = (
                pending if pending is not None and id(pending) in failed_ids else None
            )
            pending_count = len(self._inflight_connections)
            self._connection_state = (
                "STOP_UNCONFIRMED"
                if failures or pending_count
                else "DISCONNECTED"
            )
        if pending_count:
            failures.append(f"{pending_count} 个控制连接仍在进行，厂商 connect 尚未返回")
        if failures:
            raise RuntimeError("；".join(failures))

    @staticmethod
    def _unique_senders(*senders: Any | None) -> tuple[Any, ...]:
        unique: list[Any] = []
        seen: set[int] = set()
        for sender in senders:
            if sender is None or id(sender) in seen:
                continue
            seen.add(id(sender))
            unique.append(sender)
        return tuple(unique)

    @staticmethod
    def _stop_senders(senders: tuple[Any, ...]) -> tuple[list[str], list[Any]]:
        failures: list[str] = []
        failed_senders: list[Any] = []
        for sender in senders:
            stop = getattr(sender, "stop", None)
            # 简单测试/兼容后端可能只有 disconnect；同样必须使它失效。
            operation = stop if callable(stop) else getattr(sender, "disconnect", None)
            if not callable(operation):
                failures.append("控制发送器既没有 stop 也没有 disconnect")
                failed_senders.append(sender)
                continue
            try:
                operation()
            except Exception as error:
                failures.append(f"控制发送器停车失败：{error}")
                failed_senders.append(sender)
        return failures, failed_senders

    @staticmethod
    def _disconnect_senders(
        senders: tuple[Any, ...],
    ) -> tuple[list[str], list[Any]]:
        failures: list[str] = []
        failed_senders: list[Any] = []
        for sender in DeferredUrRtdeServoSender._unique_senders(*senders):
            disconnect = getattr(sender, "disconnect", None)
            if not callable(disconnect):
                failures.append("控制发送器没有 disconnect")
                failed_senders.append(sender)
                continue
            try:
                disconnect()
            except Exception as error:
                failures.append(f"控制发送器释放失败：{error}")
                failed_senders.append(sender)
        return failures, failed_senders
