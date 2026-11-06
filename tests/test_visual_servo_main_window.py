from __future__ import annotations

import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt

from visual_servo_system.ui.visual_servo_main_window import VisualServoMainWindow


class _Runtime:
    def __init__(self):
        self.shutdown_calls = 0
        self.mode_label = "预览"
        self.speed_profile = "commissioning"
        self.state = "DISCONNECTED"
        self.robot_read_only_connected = False
        self.message = "等待连接"
        self.stop_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1

    def emergency_stop(self, _reason):
        self.stop_calls += 1

    def ui_snapshot(self):
        return {
            "state": self.state,
            "mode": self.mode_label,
            "speed_profile": self.speed_profile,
            "message": self.message,
            "marker_id": None,
            "distance_m": None,
            "reprojection_error_px": None,
            "recognition_rate_hz": 0.0,
            "latency_ms": 0.0,
            "image": None,
            "ekf_status": "等待观测",
            "mpc_status": "等待求解",
            "robot_status": "未连接",
            "safety_status": "DISCONNECTED",
            "robot_read_only_connected": self.robot_read_only_connected,
        }


def test_main_window_starts_safe_and_close_calls_shutdown(qtbot):
    runtime = _Runtime()
    window = VisualServoMainWindow(runtime, confirm_callback=lambda *_args: True)
    qtbot.addWidget(window)
    assert window.connect_camera_button.isEnabled()
    assert not window.connect_robot_button.isEnabled()
    assert not window.system_check_button.isEnabled()
    assert not window.enable_button.isEnabled()
    assert not window.start_follow_button.isEnabled()
    assert "BFS-U3-13Y3C" in window.camera_identity_label.text()
    assert "16276663" in window.camera_identity_label.text()
    assert window.ekf_value.text() == "等待观测"
    assert window.mpc_value.text() == "等待求解"
    assert window.robot_value.text() == "未连接"
    assert window.safety_value.text() == "DISCONNECTED"
    runtime.state = "PREVIEW"
    window.refresh_snapshot()
    assert window.connect_robot_button.isEnabled()
    assert not window.system_check_button.isEnabled()
    runtime.robot_read_only_connected = True
    window.refresh_snapshot()
    assert not window.connect_robot_button.isEnabled()
    assert window.system_check_button.isEnabled()
    window.close()
    qtbot.waitUntil(lambda: runtime.shutdown_calls == 1)


def test_blocking_lifecycle_action_runs_off_ui_thread_and_stop_stays_responsive(qtbot):
    class BlockingRuntime(_Runtime):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def start_preview(self):
            self.entered.set()
            self.release.wait(1.0)
            self.state = "PREVIEW"

    runtime = BlockingRuntime()
    window = VisualServoMainWindow(runtime, confirm_callback=lambda *_args: True)
    qtbot.addWidget(window)
    window.show()
    qtbot.mouseClick(window.connect_camera_button, Qt.MouseButton.LeftButton)
    assert runtime.entered.wait(1.0)
    assert not window.connect_camera_button.isEnabled()
    runtime.message = "定时器仍响应"
    qtbot.waitUntil(lambda: window.message_value.text() == "定时器仍响应")
    assert window.stop_button.isEnabled()
    qtbot.mouseClick(window.stop_button, Qt.MouseButton.LeftButton)
    assert runtime.stop_calls == 1
    runtime.release.set()
    qtbot.waitUntil(lambda: not window._busy)
    window.close()
    qtbot.waitUntil(lambda: runtime.shutdown_calls == 1)


def test_close_during_blocking_action_stops_immediately_then_shuts_down(qtbot):
    class BlockingRuntime(_Runtime):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def start_preview(self):
            self.entered.set()
            self.release.wait(1.0)

    runtime = BlockingRuntime()
    window = VisualServoMainWindow(runtime, confirm_callback=lambda *_args: True)
    qtbot.addWidget(window)
    window.show()
    qtbot.mouseClick(window.connect_camera_button, Qt.MouseButton.LeftButton)
    assert runtime.entered.wait(1.0)

    window.close()
    assert runtime.stop_calls == 1
    assert window.isVisible()
    # 关闭资源使用独立 worker；不能等待原生命周期动作返回。
    qtbot.waitUntil(lambda: runtime.shutdown_calls == 1)

    runtime.release.set()
    qtbot.waitUntil(lambda: not window.isVisible())


def test_close_failure_is_visible_and_can_be_retried(qtbot):
    class FailingCloseRuntime(_Runtime):
        def shutdown(self):
            self.shutdown_calls += 1
            if self.shutdown_calls == 1:
                raise RuntimeError("相机线程仍阻塞")

    runtime = FailingCloseRuntime()
    window = VisualServoMainWindow(runtime, confirm_callback=lambda *_args: True)
    qtbot.addWidget(window)
    window.show()
    window.close()
    qtbot.waitUntil(lambda: runtime.shutdown_calls == 1)
    qtbot.waitUntil(lambda: "关闭失败" in window.message_value.text())
    assert window.isVisible()
    window.close()
    qtbot.waitUntil(lambda: runtime.shutdown_calls == 2)


def test_unconfirmed_stop_close_warns_physical_estop_and_remains_retryable(qtbot):
    class PendingStopRuntime(_Runtime):
        def __init__(self):
            super().__init__()
            self.fail = True

        def shutdown(self):
            self.shutdown_calls += 1
            if self.fail:
                raise RuntimeError("停车未确认：厂商调用仍阻塞，请立即使用物理急停")

    runtime = PendingStopRuntime()
    window = VisualServoMainWindow(runtime, confirm_callback=lambda *_args: True)
    qtbot.addWidget(window)
    window.show()
    window.close()
    qtbot.waitUntil(lambda: runtime.shutdown_calls == 1)
    qtbot.waitUntil(lambda: "物理急停" in window.message_value.text())
    assert window.isVisible()
    assert window.stop_button.isEnabled()

    runtime.fail = False
    window.close()
    qtbot.waitUntil(lambda: runtime.shutdown_calls == 2)
    qtbot.waitUntil(lambda: not window.isVisible())


def test_failed_shutdown_keeps_original_worker_tracked_until_it_exits(qtbot):
    """shutdown 失败不得把仍阻塞的原生命周期 worker 伪装成已结束。"""

    class BlockingRuntime(_Runtime):
        def __init__(self):
            super().__init__()
            self.action_entered = threading.Event()
            self.action_release = threading.Event()
            self.fail_shutdown = True

        def start_preview(self):
            self.action_entered.set()
            self.action_release.wait()

        def shutdown(self):
            self.shutdown_calls += 1
            if self.fail_shutdown:
                raise RuntimeError("停车未确认：原生命周期调用仍阻塞")

    runtime = BlockingRuntime()
    window = VisualServoMainWindow(runtime, confirm_callback=lambda *_args: True)
    qtbot.addWidget(window)
    window.show()
    qtbot.mouseClick(window.connect_camera_button, Qt.MouseButton.LeftButton)
    assert runtime.action_entered.wait(1.0)

    window.close()
    qtbot.waitUntil(lambda: runtime.shutdown_calls == 1)
    qtbot.waitUntil(lambda: "关闭失败" in window.message_value.text())
    assert window.isVisible()
    assert window._busy
    assert len(window._workers) == 1

    # 重试 shutdown 成功也不能隐藏窗口；原 worker 仍可能回调 UI。
    runtime.fail_shutdown = False
    window.close()
    qtbot.waitUntil(lambda: runtime.shutdown_calls == 2)
    qtbot.wait(30)
    assert window.isVisible()
    assert window._busy

    runtime.action_release.set()
    qtbot.waitUntil(lambda: not window.isVisible())


def test_speed_profile_worker_captures_plain_value_on_ui_thread(qtbot):
    class SpeedRuntime(_Runtime):
        def __init__(self):
            super().__init__()
            self.selected = []

        def select_speed_profile(self, value):
            self.selected.append(value)

    runtime = SpeedRuntime()
    window = VisualServoMainWindow(runtime, confirm_callback=lambda *_args: True)
    qtbot.addWidget(window)
    captured = []
    window.speed_combo.blockSignals(True)
    window.speed_combo.setCurrentIndex(1)
    window.speed_combo.blockSignals(False)
    window._run_action = captured.append

    window._select_speed()
    window.speed_combo.setCurrentIndex(0)
    captured[0]()

    assert runtime.selected == ["experiment"]
