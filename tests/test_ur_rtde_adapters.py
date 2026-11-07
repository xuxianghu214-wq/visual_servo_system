from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from visual_servo_system.geometry.pose_transforms import ur_pose_to_transform
from visual_servo_system.models.data_types import RobotStateSample
from visual_servo_system.robot.deferred_ur_rtde_servo_sender import (
    DeferredUrRtdeServoSender,
)
from visual_servo_system.robot.ur_rtde_servo_sender import (
    ControlCleanupError,
    FakeRobotBackend,
    MotionCommandRejected,
    RobotStopTimeout,
    ServoMotionEnvelope,
    UrRtdeServoSender,
)
from visual_servo_system.robot.ur_rtde_state_reader import (
    RobotConnectionError,
    RobotShutdownTimeout,
    UrRtdeStateReader,
)


class _Receive:
    def __init__(self, host: str, frequency: float):
        self.host = host
        self.frequency = frequency
        self.disconnected = False

    def getActualTCPPose(self):
        return [0.1, 0.2, 0.3, 0.0, 0.0, 0.1]

    def getActualTCPSpeed(self):
        return [0.01, 0.0, 0.0, 0.0, 0.0, 0.02]

    def getActualQ(self):
        return [1, 2, 3, 4, 5, 6]

    def getRobotMode(self):
        return 7

    def getSafetyMode(self):
        return 1

    def getRuntimeState(self):
        return 2

    def disconnect(self):
        self.disconnected = True


def test_state_reader_maps_pose_modes_and_history() -> None:
    receive = _Receive("x", 1)
    reader = UrRtdeStateReader(
        "192.168.0.10",
        frequency_hz=125.0,
        receive_factory=lambda host, frequency: receive,
        monotonic=lambda: 2.5,
    )
    reader.connect()
    reader.confirm_zero_tcp(np.zeros(6))
    reader.begin_authorized_pose_epoch(0.0)
    sample = reader.read_once()
    expected = ur_pose_to_transform(np.array(receive.getActualTCPPose()))
    assert np.allclose(sample.T_base_flange, expected)
    assert sample.host_timestamp_s == pytest.approx(2.5)
    assert sample.timestamp_source == "host_midpoint"
    assert not sample.timestamp_reliable
    assert sample.robot_mode == "robot=7;safety=1;runtime=2;tcp=zero_confirmed"
    # 无控制器设备时间时样本只供状态显示，不得进入位姿历史。
    assert len(reader.pose_history) == 0
    reader.close()
    assert receive.disconnected


def test_state_reader_maps_controller_timestamp_and_reports_reliable_time() -> None:
    class TimestampedReceive(_Receive):
        def __init__(self):
            super().__init__("x", 125.0)
            self.controller_time = 1.0

        def getTimestamp(self):
            return self.controller_time

    receive = TimestampedReceive()
    host_ticks = iter(
        (11.0000, 11.0020, 12.0000, 12.0010, 13.0000, 13.0030, 14.0000, 14.0010)
    )
    reader = UrRtdeStateReader(
        "192.168.0.10",
        receive_factory=lambda *_args: receive,
        monotonic=lambda: next(host_ticks),
    )
    reader.connect()
    reader.confirm_zero_tcp(np.zeros(6))
    reader.begin_authorized_pose_epoch(0.0)
    samples = []
    for device_second in (1.0, 2.0, 3.0, 4.0):
        receive.controller_time = device_second
        samples.append(reader.read_once())
    assert samples[-1].timestamp_source == "ur_controller"
    assert samples[-1].timestamp_reliable
    assert samples[-1].host_timestamp_s == pytest.approx(14.0, abs=0.004)
    assert len(reader.pose_history) == 1
    reader.close()


def test_state_reader_does_not_publish_duplicate_controller_state() -> None:
    class TimestampedReceive(_Receive):
        controller_time = 1.0

        def getTimestamp(self):
            return self.controller_time

    receive = TimestampedReceive("x", 125.0)
    host_time = 10.0

    def monotonic():
        nonlocal host_time
        host_time += 0.0005
        return host_time

    reader = UrRtdeStateReader(
        "192.168.0.10",
        receive_factory=lambda *_args: receive,
        monotonic=monotonic,
    )
    reader.connect()

    first = reader.read_once()
    published_before = reader.latest_state.get_latest()
    duplicate = reader.read_once()
    published_after = reader.latest_state.get_latest()

    assert first is not None
    assert duplicate is None
    assert published_after == published_before
    assert reader.sample_diagnostics.duplicate == 1
    assert reader.sample_diagnostics.accepted == 1
    assert reader.sample_diagnostics.clock_reset == 0
    reader.close()


def test_state_reader_exposes_controller_sample_identity() -> None:
    class TimestampedReceive(_Receive):
        def getTimestamp(self):
            return 42.125

    reader = UrRtdeStateReader(
        "192.168.0.10",
        receive_factory=lambda *_args: TimestampedReceive("x", 125.0),
        monotonic=lambda: 100.0,
    )
    reader.connect()
    sample = reader.read_once()

    assert sample is not None
    assert sample.controller_timestamp_s == pytest.approx(42.125)
    assert sample.sample_identity == "ur-controller:42.125000000"
    reader.close()


def test_deferred_sender_stop_releases_generation_and_reconnects_fresh() -> None:
    sample = RobotStateSample(1.0, np.eye(4), valid=False)

    class Buffer:
        def get_latest(self):
            return 1, sample

    class Reader:
        latest_state = Buffer()
        last_actual_tcp_pose_ur = np.zeros(6)

    created = []

    class Inner:
        def __init__(self):
            self.connected = 0
            self.disconnected = 0

        def connect(self):
            self.connected += 1

        def disconnect(self):
            self.disconnected += 1

    def factory(*_args, **_kwargs):
        inner = Inner()
        created.append(inner)
        return inner

    sender = DeferredUrRtdeServoSender("x", Reader(), sender_factory=factory)
    sender.connect()
    sender.stop()
    sender.connect()
    assert len(created) == 2
    assert created[0].disconnected == 1
    assert created[1].connected == 1


def test_deferred_stop_immediately_invalidates_a_pending_real_sender() -> None:
    """外层 Deferred 必须在内层 RTDE 工厂阻塞时仍能立即取得停车权。"""

    sample = RobotStateSample(1.0, np.eye(4), valid=True)

    class Buffer:
        def get_latest(self):
            return 1, sample

    class Reader:
        latest_state = Buffer()
        last_actual_tcp_pose_ur = np.zeros(6)

    factory_entered = threading.Event()
    factory_release = threading.Event()
    controls: list[FakeRobotBackend] = []
    inners: list[UrRtdeServoSender] = []
    errors: list[Exception] = []

    def blocking_control_factory(_host: str) -> FakeRobotBackend:
        factory_entered.set()
        assert factory_release.wait(1.0)
        control = FakeRobotBackend()
        controls.append(control)
        return control

    def sender_factory(host: str, **kwargs) -> UrRtdeServoSender:
        inner = UrRtdeServoSender(
            host,
            control_factory=blocking_control_factory,
            **kwargs,
        )
        inners.append(inner)
        return inner

    sender = DeferredUrRtdeServoSender("x", Reader(), sender_factory=sender_factory)
    connect_thread = threading.Thread(
        target=lambda: _capture_exception(sender.connect, errors),
        daemon=True,
    )
    connect_thread.start()
    assert factory_entered.wait(1.0)
    inner = inners[0]
    generation_while_connecting = inner._lifecycle_generation

    sender.stop()

    assert inner._connection_state == "STOPPED"
    assert inner._lifecycle_generation > generation_while_connecting
    assert sender.connection_state == "STOPPED"
    factory_release.set()
    connect_thread.join(1.0)
    assert not connect_thread.is_alive()
    assert len(errors) == 1
    assert controls[0].call_log == []
    assert controls[0].servo_commands == []
    assert controls[0].servo_stop_count == 1
    assert controls[0].stop_script_count == 1
    assert controls[0].disconnect_count == 1


def test_deferred_rejects_a_second_connect_while_first_is_pending() -> None:
    sample = RobotStateSample(1.0, np.eye(4), valid=True)

    class Buffer:
        def get_latest(self):
            return 1, sample

    class Reader:
        latest_state = Buffer()
        last_actual_tcp_pose_ur = np.zeros(6)

    entered = threading.Event()
    release = threading.Event()
    created = 0

    class Inner:
        def connect(self):
            entered.set()
            assert release.wait(1.0)

        def stop(self):
            pass

        def disconnect(self):
            pass

    def factory(*_args, **_kwargs):
        nonlocal created
        created += 1
        return Inner()

    sender = DeferredUrRtdeServoSender("x", Reader(), sender_factory=factory)
    thread = threading.Thread(target=sender.connect, daemon=True)
    thread.start()
    assert entered.wait(1.0)
    with pytest.raises(RuntimeError, match="正在连接"):
        sender.connect()
    assert created == 1
    release.set()
    thread.join(1.0)
    assert not thread.is_alive()
    sender.disconnect()


def test_deferred_disconnect_retains_failed_active_sender_for_retry() -> None:
    sample = RobotStateSample(1.0, np.eye(4), valid=True)

    class Buffer:
        def get_latest(self):
            return 1, sample

    class Reader:
        latest_state = Buffer()
        last_actual_tcp_pose_ur = np.zeros(6)

    class Inner:
        def __init__(self):
            self.disconnect_calls = 0

        def connect(self):
            pass

        def stop(self):
            pass

        def disconnect(self):
            self.disconnect_calls += 1
            if self.disconnect_calls == 1:
                raise RuntimeError("disconnect blocked")

    inner = Inner()
    sender = DeferredUrRtdeServoSender(
        "x", Reader(), sender_factory=lambda *_args, **_kwargs: inner
    )
    sender.connect()

    with pytest.raises(RuntimeError, match="disconnect blocked"):
        sender.disconnect()
    assert sender.connection_state == "STOP_UNCONFIRMED"
    sender.disconnect()
    assert sender.connection_state == "DISCONNECTED"
    assert inner.disconnect_calls == 2


def _capture_exception(action, errors: list[Exception]) -> None:
    try:
        action()
    except Exception as error:
        errors.append(error)


def test_state_reader_is_fail_closed_until_control_side_confirms_zero_tcp() -> None:
    receive = _Receive("x", 1)
    reader = UrRtdeStateReader("192.168.0.10", receive_factory=lambda *_args: receive)
    reader.connect()
    unverified = reader.read_once()
    assert not unverified.valid
    assert np.array_equal(unverified.T_base_flange, np.eye(4))
    assert len(reader.pose_history) == 0
    with pytest.raises(RobotConnectionError, match="TCP"):
        reader.confirm_zero_tcp(np.array([0, 0, 0.001, 0, 0, 0], dtype=float))
    reader.confirm_zero_tcp(np.zeros(6))
    reader.begin_authorized_pose_epoch(0.0)
    assert reader.read_once().valid
    assert len(reader.pose_history) == 0
    reader.revoke_tcp_confirmation()
    assert not reader.read_once().valid
    assert len(reader.pose_history) == 0


def test_tcp_authorization_and_revocation_clear_pose_history_epochs() -> None:
    reader = UrRtdeStateReader("192.168.0.10", receive_factory=lambda *_args: _Receive("x", 1))

    def reliable_sample(timestamp: float) -> RobotStateSample:
        return RobotStateSample(
            timestamp,
            np.eye(4),
            valid=True,
            receive_timestamp_s=timestamp + 0.001,
            timestamp_source="ur_controller",
            timestamp_reliable=True,
            timestamp_uncertainty_s=0.001,
        )

    reader.pose_history.add(reliable_sample(1.0))
    reader.confirm_zero_tcp(np.zeros(6))
    assert len(reader.pose_history) == 0

    reader.pose_history.add(reliable_sample(2.0))
    reader.begin_authorized_pose_epoch(2.1)
    assert len(reader.pose_history) == 0

    reader.pose_history.add(reliable_sample(3.0))
    reader.revoke_tcp_confirmation()
    assert len(reader.pose_history) == 0


def test_state_reader_rejects_a_blocking_read_timeout() -> None:
    receive = _Receive("x", 1)
    ticks = iter((1.0, 1.2))
    reader = UrRtdeStateReader(
        "192.168.0.10",
        read_timeout_s=0.1,
        receive_factory=lambda *_args: receive,
        monotonic=lambda: next(ticks),
    )
    reader.connect()
    reader.confirm_zero_tcp(np.zeros(6))
    with pytest.raises(RobotConnectionError, match="超时"):
        reader.read_once()
    assert len(reader.pose_history) == 0


def test_reader_watchdog_faults_and_stop_never_disconnects_blocked_getter() -> None:
    class BlockingReceive(_Receive):
        def __init__(self, host, frequency):
            super().__init__(host, frequency)
            self.entered = threading.Event()
            self.release = threading.Event()
            self.disconnect_threads: list[int] = []

        def getActualTCPPose(self):
            self.entered.set()
            self.release.wait()
            return super().getActualTCPPose()

        def disconnect(self):
            self.disconnect_threads.append(threading.get_ident())

    receive = BlockingReceive("x", 125)
    reader = UrRtdeStateReader(
        "192.168.0.10",
        receive_factory=lambda *_args: receive,
        read_timeout_s=0.03,
        watchdog_check_period_s=0.005,
    )
    reader.confirm_zero_tcp(np.zeros(6))
    reader.start()
    assert receive.entered.wait(1.0)
    assert reader.watchdog_fault_event.wait(1.0)
    assert not reader.communication_healthy
    caller_thread = threading.get_ident()
    with pytest.raises(RobotShutdownTimeout, match="仍在运行"):
        reader.stop(timeout_s=0.01)
    assert receive.disconnect_threads == []
    receive.release.set()
    reader.stop(timeout_s=1.0)
    assert receive.disconnect_threads
    assert all(thread_id != caller_thread for thread_id in receive.disconnect_threads)


def _sender(backend: FakeRobotBackend) -> UrRtdeServoSender:
    envelope = ServoMotionEnvelope(
        start_pose_ur=np.zeros(6),
        max_translation_m=0.1,
        max_rotation_rad=np.deg2rad(15.0),
        max_step_translation_m=0.001,
        max_step_rotation_rad=0.01,
    )
    return UrRtdeServoSender(
        "192.168.0.10",
        control_factory=lambda _host: backend,
        envelope=envelope,
        tcp_offset_provider=backend.getTCPOffset,
    )


def _reader(receive: _Receive) -> UrRtdeStateReader:
    return UrRtdeStateReader(
        "192.168.0.10", receive_factory=lambda *_args: receive
    )


def test_sender_authorization_owns_reader_tcp_lifecycle() -> None:
    receive = _Receive("x", 125)
    reader = _reader(receive)
    backend = FakeRobotBackend()
    sender = _sender(backend)
    reader.connect()
    sender.connect()
    assert not reader.read_once().valid
    sender.authorize_state_reader(reader)
    assert reader.read_once().valid
    history_length = len(reader.pose_history)
    sender.stop()
    assert not reader.tcp_zero_confirmed
    assert not reader.read_once().valid
    assert len(reader.pose_history) == history_length


def test_tcp_provider_failure_during_authorization_stops_and_revokes() -> None:
    reader = _reader(_Receive("x", 125))
    backend = FakeRobotBackend()
    sender = _sender(backend)
    reader.connect()
    sender.connect()
    backend.fail_tcp_offset = True
    with pytest.raises(MotionCommandRejected, match="TCP"):
        sender.authorize_state_reader(reader)
    assert not reader.tcp_zero_confirmed
    assert backend.servo_stop_count == 1
    assert backend.stop_script_count == 1


def test_verified_tcp_provider_failure_revokes_existing_authorization() -> None:
    reader = _reader(_Receive("x", 125))
    backend = FakeRobotBackend()
    sender = _sender(backend)
    reader.connect()
    sender.connect()
    sender.authorize_state_reader(reader)
    backend.fail_tcp_offset = True
    with pytest.raises(MotionCommandRejected, match="TCP"):
        sender.verified_tcp_offset()
    assert not reader.tcp_zero_confirmed
    assert backend.servo_stop_count == 1
    assert backend.stop_script_count == 1


@pytest.mark.parametrize("failure", ["servo", "wait"])
def test_send_failures_automatically_revoke_reader_authorization(failure: str) -> None:
    reader = _reader(_Receive("x", 125))
    backend = FakeRobotBackend()
    sender = _sender(backend)
    reader.connect()
    sender.connect()
    sender.authorize_state_reader(reader)
    if failure == "servo":
        backend.fail_servo_l = True
    else:
        backend.fail_wait_period = True
    with pytest.raises(RuntimeError):
        sender.send_pose(np.array([0.0005, 0, 0, 0, 0, 0], dtype=float))
    assert not reader.tcp_zero_confirmed


def test_repeated_reader_authorization_and_stop_are_idempotent() -> None:
    reader = _reader(_Receive("x", 125))
    backend = FakeRobotBackend()
    sender = _sender(backend)
    reader.connect()
    sender.connect()
    sender.authorize_state_reader(reader)
    sender.authorize_state_reader(reader)
    assert reader.tcp_zero_confirmed
    sender.stop()
    sender.stop()
    assert not reader.tcp_zero_confirmed
    assert backend.servo_stop_count == 1
    assert backend.stop_script_count == 1


def test_disconnect_revokes_reader_authorization() -> None:
    reader = _reader(_Receive("x", 125))
    backend = FakeRobotBackend()
    sender = _sender(backend)
    reader.connect()
    sender.connect()
    sender.authorize_state_reader(reader)
    sender.disconnect()
    assert not reader.tcp_zero_confirmed


def test_stop_wins_when_tcp_authorization_provider_is_blocked() -> None:
    reader = _reader(_Receive("x", 125))
    backend = FakeRobotBackend()
    provider_entered = threading.Event()
    provider_release = threading.Event()
    block_provider = False

    def controlled_provider():
        if block_provider:
            provider_entered.set()
            assert provider_release.wait(1.0)
        return backend.getTCPOffset()

    envelope = ServoMotionEnvelope(np.zeros(6), 0.1, 0.2, 0.001, 0.01)
    sender = UrRtdeServoSender(
        "192.168.0.10",
        control_factory=lambda _host: backend,
        envelope=envelope,
        tcp_offset_provider=controlled_provider,
    )
    reader.connect()
    sender.connect()
    block_provider = True
    authorization_errors: list[Exception] = []

    def authorize() -> None:
        try:
            sender.authorize_state_reader(reader)
        except Exception as error:
            authorization_errors.append(error)

    thread = threading.Thread(target=authorize)
    thread.start()
    assert provider_entered.wait(1.0)
    with pytest.raises(RobotStopTimeout, match="物理急停"):
        sender.stop()
    assert sender.connection_state == "STOP_UNCONFIRMED"
    assert not reader.tcp_zero_confirmed
    provider_release.set()
    thread.join(1.0)
    assert not thread.is_alive()
    assert len(authorization_errors) == 1
    assert isinstance(authorization_errors[0], MotionCommandRejected)
    assert not reader.tcp_zero_confirmed
    assert not reader.read_once().valid
    deadline = time.monotonic() + 1.0
    while sender.connection_state != "STOPPED" and time.monotonic() < deadline:
        time.sleep(0.005)
    assert sender.connection_state == "STOPPED"
    sender.stop()
    assert backend.servo_stop_count == 1
    assert backend.stop_script_count == 1


def test_stop_invalidates_a_factory_result_that_returns_late() -> None:
    factory_entered = threading.Event()
    factory_release = threading.Event()
    created: list[FakeRobotBackend] = []
    connect_errors: list[Exception] = []

    def factory(_host):
        factory_entered.set()
        assert factory_release.wait(1.0)
        backend = FakeRobotBackend()
        created.append(backend)
        return backend

    sender = UrRtdeServoSender(
        "192.168.0.10",
        control_factory=factory,
        envelope=ServoMotionEnvelope(np.zeros(6), 0.1, 0.2, 0.001, 0.01),
    )

    def connect() -> None:
        try:
            sender.connect()
        except Exception as error:
            connect_errors.append(error)

    thread = threading.Thread(target=connect)
    thread.start()
    assert factory_entered.wait(1.0)
    sender.stop()
    factory_release.set()
    thread.join(1.0)
    assert not thread.is_alive()
    assert len(connect_errors) == 1
    assert isinstance(connect_errors[0], MotionCommandRejected)
    assert len(created) == 1
    assert created[0].servo_stop_count == 1
    assert created[0].stop_script_count == 1
    assert created[0].disconnect_count == 1
    # stop 已使代次失效；迟到对象除停车/释放外，不得再做 TCP 授信读取。
    assert "tcp" not in created[0].call_log


def test_second_concurrent_connect_is_rejected_without_creating_a_device() -> None:
    factory_entered = threading.Event()
    factory_release = threading.Event()
    factory_count = 0

    def factory(_host):
        nonlocal factory_count
        factory_count += 1
        factory_entered.set()
        assert factory_release.wait(1.0)
        return FakeRobotBackend()

    sender = UrRtdeServoSender(
        "192.168.0.10",
        control_factory=factory,
        envelope=ServoMotionEnvelope(np.zeros(6), 0.1, 0.2, 0.001, 0.01),
    )
    thread = threading.Thread(target=sender.connect)
    thread.start()
    assert factory_entered.wait(1.0)
    with pytest.raises(MotionCommandRejected, match="正在连接"):
        sender.connect()
    factory_release.set()
    thread.join(1.0)
    assert not thread.is_alive()
    assert factory_count == 1
    sender.disconnect()


def test_connect_requires_complete_control_safety_api_and_cleans_up() -> None:
    class IncompleteControl(FakeRobotBackend):
        getInverseKinematicsHasSolution = None

    backend = IncompleteControl()
    sender = UrRtdeServoSender(
        "192.168.0.10",
        control_factory=lambda _host: backend,
        envelope=ServoMotionEnvelope(np.zeros(6), 0.1, 0.2, 0.001, 0.01),
    )
    with pytest.raises(MotionCommandRejected, match="getInverseKinematicsHasSolution"):
        sender.connect()
    assert backend.servo_stop_count == 1
    assert backend.stop_script_count == 1
    assert backend.disconnect_count == 1


def test_servo_sender_checks_command_then_uses_period_api() -> None:
    backend = FakeRobotBackend()
    sender = _sender(backend)
    sender.connect()
    backend.call_log.clear()
    sender.send_pose(np.array([0.0005, 0, 0, 0, 0, 0], dtype=float))
    assert len(backend.servo_commands) == 1
    assert backend.period_wait_count == 1
    assert backend.call_log[:4] == ["initPeriod", "tcp", "safety", "ik"]
    assert backend.call_log[-2:] == ["servoL", "waitPeriod"]


def test_concurrent_send_uses_latest_committed_pose_for_step_gate() -> None:
    """两条并发命令不得共享同一个旧 last_pose 通过单周期门禁。"""

    backend = FakeRobotBackend()
    provider_lock = threading.Lock()
    first_provider_entered = threading.Event()
    second_provider_entered = threading.Event()
    release_first_provider = threading.Event()
    start_barrier = threading.Barrier(3)
    count_send_provider_calls = False
    send_provider_calls = 0

    def controlled_provider():
        nonlocal send_provider_calls
        if count_send_provider_calls:
            with provider_lock:
                send_provider_calls += 1
                call_number = send_provider_calls
            if call_number == 1:
                first_provider_entered.set()
                assert release_first_provider.wait(1.0)
            elif call_number == 2:
                second_provider_entered.set()
        return backend.getTCPOffset()

    sender = UrRtdeServoSender(
        "192.168.0.10",
        control_factory=lambda _host: backend,
        envelope=ServoMotionEnvelope(np.zeros(6), 0.1, 0.2, 0.0011, 0.01),
        tcp_offset_provider=controlled_provider,
    )
    sender.connect()
    count_send_provider_calls = True
    errors: list[Exception] = []

    def send(x_m: float) -> None:
        start_barrier.wait()
        try:
            sender.send_pose(np.array([x_m, 0, 0, 0, 0, 0], dtype=float))
        except Exception as error:
            errors.append(error)

    threads = [
        threading.Thread(target=send, args=(0.001,)),
        threading.Thread(target=send, args=(-0.001,)),
    ]
    for thread in threads:
        thread.start()
    start_barrier.wait()
    assert first_provider_entered.wait(1.0)
    # 在第一条 provider 被卡住时，第二条不得进入命令事务。
    second_entered_before_release = second_provider_entered.wait(0.05)
    release_first_provider.set()
    for thread in threads:
        thread.join(1.0)
        assert not thread.is_alive()

    assert not second_entered_before_release
    assert len(backend.servo_commands) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], MotionCommandRejected)
    assert backend.servo_stop_count == 1
    assert backend.stop_script_count == 1


def test_stop_is_not_blocked_by_a_send_tcp_provider() -> None:
    backend = FakeRobotBackend()
    provider_entered = threading.Event()
    provider_release = threading.Event()
    block_provider = False

    def controlled_provider():
        if block_provider:
            provider_entered.set()
            assert provider_release.wait(1.0)
        return backend.getTCPOffset()

    sender = UrRtdeServoSender(
        "192.168.0.10",
        control_factory=lambda _host: backend,
        envelope=ServoMotionEnvelope(np.zeros(6), 0.1, 0.2, 0.0011, 0.01),
        tcp_offset_provider=controlled_provider,
    )
    sender.connect()
    block_provider = True
    send_errors: list[Exception] = []

    def send() -> None:
        try:
            sender.send_pose(np.array([0.001, 0, 0, 0, 0, 0], dtype=float))
        except Exception as error:
            send_errors.append(error)

    send_thread = threading.Thread(target=send)
    send_thread.start()
    assert provider_entered.wait(1.0)
    stop_finished = threading.Event()
    stop_errors = []

    def stop():
        try:
            sender.stop()
        except Exception as error:
            stop_errors.append(error)
        finally:
            stop_finished.set()

    stop_thread = threading.Thread(target=stop)
    stop_thread.start()
    assert stop_finished.wait(0.2)
    assert len(stop_errors) == 1
    assert isinstance(stop_errors[0], RobotStopTimeout)
    assert sender.connection_state == "STOP_UNCONFIRMED"
    provider_release.set()
    send_thread.join(1.0)
    stop_thread.join(1.0)
    assert not send_thread.is_alive()
    assert not stop_thread.is_alive()
    assert len(send_errors) == 1
    assert isinstance(send_errors[0], MotionCommandRejected)
    assert backend.servo_commands == []
    assert backend.servo_stop_count == 1
    assert backend.stop_script_count == 1


@pytest.mark.parametrize(
    "pose",
    [
        np.array([np.nan, 0, 0, 0, 0, 0]),
        np.array([0.2, 0, 0, 0, 0, 0]),
        np.array([0.01, 0, 0, 0, 0, 0]),
    ],
)
def test_servo_sender_rejects_nonfinite_envelope_and_step(pose: np.ndarray) -> None:
    backend = FakeRobotBackend()
    sender = _sender(backend)
    sender.connect()
    with pytest.raises(MotionCommandRejected):
        sender.send_pose(pose)
    assert backend.servo_commands == []
    assert backend.servo_stop_count == 1
    assert backend.stop_script_count == 1


def test_servo_sender_refuses_nonzero_tcp_and_ik_or_safety_failures() -> None:
    for setup in ("tcp", "ik", "safety"):
        backend = FakeRobotBackend()
        if setup == "tcp":
            backend.tcp_offset[2] = 0.001
        elif setup == "ik":
            backend.inverse_kinematics_available = False
        else:
            backend.pose_within_safety_limits = False
        sender = _sender(backend)
        if setup == "tcp":
            with pytest.raises(MotionCommandRejected):
                sender.connect()
            assert backend.servo_stop_count == 1
            assert backend.stop_script_count == 1
        else:
            sender.connect()
            with pytest.raises(MotionCommandRejected):
                sender.send_pose(np.array([0.0005, 0, 0, 0, 0, 0], dtype=float))
            assert backend.servo_stop_count == 1
            assert backend.stop_script_count == 1


def test_stop_is_idempotent_and_calls_both_even_if_first_fails() -> None:
    backend = FakeRobotBackend(fail_servo_stop=True)
    sender = _sender(backend)
    sender.connect()
    with pytest.raises(ControlCleanupError):
        sender.stop()
    with pytest.raises(ControlCleanupError):
        sender.stop()
    assert backend.servo_stop_count == 2
    assert backend.stop_script_count == 1


def test_disconnect_attempts_stop_script_and_disconnect_even_after_stop_error() -> None:
    backend = FakeRobotBackend(fail_servo_stop=True, fail_stop_script=True)
    sender = _sender(backend)
    sender.connect()
    with pytest.raises(ControlCleanupError):
        sender.disconnect()
    assert backend.servo_stop_count == 2
    assert backend.stop_script_count == 2
    assert backend.disconnect_count == 1


def test_sender_can_only_resume_after_explicit_reconnect() -> None:
    backend = FakeRobotBackend()
    sender = _sender(backend)
    sender.connect()
    sender.stop()
    with pytest.raises(MotionCommandRejected):
        sender.send_pose(np.zeros(6))
    sender.connect()
    sender.send_pose(np.array([0.0005, 0, 0, 0, 0, 0], dtype=float))
    assert len(backend.servo_commands) == 1


def test_context_manager_stops_and_disconnects_after_send_exception() -> None:
    backend = FakeRobotBackend(fail_servo_l=True)
    sender = _sender(backend)
    with pytest.raises(RuntimeError, match="servoL"):
        with sender:
            sender.send_pose(np.array([0.0005, 0, 0, 0, 0, 0], dtype=float))
    assert backend.servo_stop_count == 1
    assert backend.stop_script_count == 1
    assert backend.disconnect_count == 1


def test_wait_period_failure_also_stops_robot() -> None:
    backend = FakeRobotBackend(fail_wait_period=True)
    sender = _sender(backend)
    sender.connect()
    with pytest.raises(RuntimeError, match="waitPeriod"):
        sender.send_pose(np.array([0.0005, 0, 0, 0, 0, 0], dtype=float))
    assert backend.servo_stop_count == 1
    assert backend.stop_script_count == 1


def test_blocked_wait_period_makes_stop_fast_but_explicitly_unconfirmed() -> None:
    class BlockingWaitBackend(FakeRobotBackend):
        def __init__(self):
            super().__init__()
            self.wait_entered = threading.Event()
            self.wait_release = threading.Event()

        def waitPeriod(self, _token):
            self.call_log.append("waitPeriod")
            self.wait_entered.set()
            self.wait_release.wait()
            self.period_wait_count += 1

    backend = BlockingWaitBackend()
    sender = UrRtdeServoSender(
        "192.168.0.10",
        control_factory=lambda _host: backend,
        envelope=ServoMotionEnvelope(np.zeros(6), 0.1, 0.2, 0.001, 0.01),
        tcp_offset_provider=backend.getTCPOffset,
        stop_timeout_s=0.03,
    )
    sender.connect()
    send_errors: list[Exception] = []
    send_thread = threading.Thread(
        target=lambda: _capture_exception(
            lambda: sender.send_pose(np.array([0.0005, 0, 0, 0, 0, 0])),
            send_errors,
        ),
        daemon=True,
    )
    send_thread.start()
    assert backend.wait_entered.wait(1.0)

    started = time.monotonic()
    with pytest.raises(RobotStopTimeout, match="物理急停"):
        sender.stop()
    assert time.monotonic() - started < 0.2
    assert sender.connection_state == "STOP_UNCONFIRMED"
    with pytest.raises(MotionCommandRejected):
        sender.send_pose(np.zeros(6))
    assert backend.servo_stop_count == 1
    assert backend.stop_script_count == 1

    backend.wait_release.set()
    send_thread.join(1.0)
    assert not send_thread.is_alive()
    assert send_errors
    deadline = time.monotonic() + 1.0
    while sender.connection_state != "STOPPED" and time.monotonic() < deadline:
        time.sleep(0.005)
    assert sender.connection_state == "STOPPED"


def test_blocked_servo_stop_cleanup_times_out_without_leaking_after_release() -> None:
    class BlockingStopBackend(FakeRobotBackend):
        def __init__(self):
            super().__init__()
            self.stop_entered = threading.Event()
            self.stop_release = threading.Event()

        def servoStop(self):
            self.servo_stop_count += 1
            self.stop_entered.set()
            self.stop_release.wait()

    backend = BlockingStopBackend()
    sender = UrRtdeServoSender(
        "192.168.0.10",
        control_factory=lambda _host: backend,
        envelope=ServoMotionEnvelope(np.zeros(6), 0.1, 0.2, 0.001, 0.01),
        tcp_offset_provider=backend.getTCPOffset,
        stop_timeout_s=0.03,
    )
    sender.connect()

    started = time.monotonic()
    with pytest.raises(RobotStopTimeout, match="物理急停"):
        sender.stop()
    assert time.monotonic() - started < 0.2
    assert backend.stop_entered.is_set()
    assert sender.connection_state == "STOP_UNCONFIRMED"

    backend.stop_release.set()
    deadline = time.monotonic() + 1.0
    while sender.connection_state != "STOPPED" and time.monotonic() < deadline:
        time.sleep(0.005)
    assert sender.connection_state == "STOPPED"
    assert sender._cleanup_thread is None
    assert backend.servo_stop_count == 1
    assert backend.stop_script_count == 1


def test_disconnect_waits_for_active_wait_period_before_releasing_control() -> None:
    """disconnect 不得与尚未返回的厂商 control 调用并发。"""

    class BlockingWaitBackend(FakeRobotBackend):
        def __init__(self):
            super().__init__()
            self.wait_entered = threading.Event()
            self.wait_release = threading.Event()
            self.vendor_call_active = False
            self.disconnect_during_active = False

        def waitPeriod(self, _token):
            self.vendor_call_active = True
            self.wait_entered.set()
            self.wait_release.wait()
            self.vendor_call_active = False

        def disconnect(self):
            if self.vendor_call_active:
                self.disconnect_during_active = True
            super().disconnect()

    backend = BlockingWaitBackend()
    sender = UrRtdeServoSender(
        "192.168.0.10",
        control_factory=lambda _host: backend,
        envelope=ServoMotionEnvelope(np.zeros(6), 0.1, 0.2, 0.001, 0.01),
        tcp_offset_provider=backend.getTCPOffset,
        stop_timeout_s=0.03,
    )
    sender.connect()
    send_errors: list[Exception] = []
    send_thread = threading.Thread(
        target=lambda: _capture_exception(
            lambda: sender.send_pose(np.array([0.0005, 0, 0, 0, 0, 0])),
            send_errors,
        ),
        daemon=True,
    )
    send_thread.start()
    assert backend.wait_entered.wait(1.0)

    with pytest.raises(RobotStopTimeout, match="未能在时限内确认"):
        sender.disconnect()
    assert sender.connection_state == "STOP_UNCONFIRMED"
    assert backend.disconnect_count == 0
    assert not backend.disconnect_during_active

    backend.wait_release.set()
    send_thread.join(1.0)
    assert not send_thread.is_alive()
    deadline = time.monotonic() + 1.0
    while backend.disconnect_count != 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert backend.disconnect_count == 1
    assert not backend.disconnect_during_active
    assert sender.connection_state == "DISCONNECTED"


def test_failed_stop_and_disconnect_steps_are_retryable_without_repeating_successes() -> None:
    backend = FakeRobotBackend(fail_servo_stop=True)
    sender = _sender(backend)
    sender.connect()

    with pytest.raises(ControlCleanupError, match="servoStop"):
        sender.stop()
    assert sender.connection_state == "STOP_UNCONFIRMED"
    assert backend.servo_stop_count == 1
    assert backend.stop_script_count == 1

    backend.fail_servo_stop = False
    sender.stop()
    assert sender.connection_state == "STOPPED"
    assert backend.servo_stop_count == 2
    assert backend.stop_script_count == 1

    backend.fail_disconnect = True
    with pytest.raises(ControlCleanupError, match="disconnect"):
        sender.disconnect()
    assert backend.disconnect_count == 1
    backend.fail_disconnect = False
    sender.disconnect()
    assert backend.disconnect_count == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"frequency_hz": 0.0},
        {"frequency_hz": np.nan},
        {"frequency_hz": True},
        {"velocity": 0.0},
        {"acceleration": np.inf},
        {"lookahead_time_s": 0.02},
        {"lookahead_time_s": 0.21},
        {"gain": 99.0},
        {"gain": 2001.0},
    ],
)
def test_servo_constructor_rejects_unsafe_timing_values(overrides) -> None:
    backend = FakeRobotBackend()
    kwargs = {
        "control_factory": lambda _host: backend,
        "envelope": ServoMotionEnvelope(np.zeros(6), 0.1, 0.2, 0.001, 0.01),
        "tcp_offset_provider": backend.getTCPOffset,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        UrRtdeServoSender("192.168.0.10", **kwargs)

