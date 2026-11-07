"""默认禁止构造实机控制接口的离线故障注入。"""

from __future__ import annotations

from dataclasses import dataclass

from visual_servo_system.safety.visual_servo_safety_monitor import (
    VisualServoSafetyMonitor,
    VisualServoState,
)

OFFLINE_FAULT_SCENARIOS = (
    "camera_disconnect",
    "rtde_interrupt",
    "stale_frame_backlog",
    "time_mapper_failure",
    "ekf_observation_timeout",
    "invalid_calibration",
    "diagnostic_buffer_overflow",
    "worker_thread_exception",
)


@dataclass(frozen=True, slots=True)
class OfflineFaultResult:
    scenario: str
    fault_latched: bool
    motion_enabled: bool
    motion_commands_sent: int
    control_interface_constructed: bool


class OfflineFaultHarness:
    """只验证故障闭锁逻辑；设计上不接受任何控制工厂。"""

    def __init__(self) -> None:
        self.motion_enabled = False
        self.motion_commands_sent = 0
        self.control_interface_constructed = False
        self.safety_monitor = VisualServoSafetyMonitor(clock=lambda: 1.0)
        self.safety_monitor.connect_preview()

    def send_motion_forbidden(self) -> None:
        raise RuntimeError("offline validation forbids every robot motion command")

    def inject(self, scenario: str) -> OfflineFaultResult:
        if scenario not in OFFLINE_FAULT_SCENARIOS:
            raise ValueError(f"unsupported offline fault scenario: {scenario}")
        self.safety_monitor.fault(f"offline injected: {scenario}")
        self.motion_enabled = False
        fault_latched = self.safety_monitor.state is VisualServoState.FAULT
        return OfflineFaultResult(
            scenario=scenario,
            fault_latched=fault_latched,
            motion_enabled=self.motion_enabled,
            motion_commands_sent=self.motion_commands_sent,
            control_interface_constructed=self.control_interface_constructed,
        )
