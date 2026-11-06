"""PySide6 中文操作界面。

界面只发出高层操作请求和显示最新快照；图像处理、MPC 和 servoL 从不在
Qt 主线程中运行。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class _WorkerSignals(QObject):
    """后台生命周期动作只通过排队信号回到 Qt 主线程。"""

    succeeded = Signal(object)
    failed = Signal(str)


class _LifecycleAction(QRunnable):
    def __init__(self, action: Callable[[], Any]) -> None:
        super().__init__()
        self.action = action
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.action()
        except Exception as error:
            self.signals.failed.emit(str(error))
        else:
            self.signals.succeeded.emit(result)


class VisualServoMainWindow(QMainWindow):
    def __init__(
        self,
        orchestrator: Any,
        *,
        confirm_callback: Callable[[str, str], bool] | None = None,
    ) -> None:
        super().__init__()
        self.orchestrator = orchestrator
        self._confirm_callback = confirm_callback or self._ask_confirmation
        self._thread_pool = QThreadPool(self)
        self._busy = False
        self._workers: set[_LifecycleAction] = set()
        # 普通生命周期动作与 shutdown worker 分开追踪。任何一个
        # shutdown 的成败都不能覆盖仍在运行的原 worker 状态。
        self._lifecycle_workers: set[_LifecycleAction] = set()
        self._closing = False
        self._close_ready = False
        self._shutdown_worker_active = False
        self._shutdown_completed_while_busy = False
        self._close_failure_message: str | None = None
        self.setWindowTitle("UR5 眼在手上视觉伺服")
        self.resize(1180, 760)
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self.video_label = QLabel("尚未连接相机")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(720, 560)
        self.video_label.setStyleSheet("background:#16191d;color:#d8dee9;border:1px solid #4c566a")
        root.addWidget(self.video_label, 3)

        side = QVBoxLayout()
        root.addLayout(side, 2)
        self.camera_identity_label = QLabel(
            "相机：Point Grey Blackfly S BFS-U3-13Y3C\n序列号：16276663"
        )
        self.mode_label = QLabel("模式：预览（机器人禁止运动）")
        self.mode_label.setStyleSheet("font-weight:bold;color:#bf616a")
        side.addWidget(self.camera_identity_label)
        side.addWidget(self.mode_label)

        status_box = QGroupBox("实时状态")
        status_form = QFormLayout(status_box)
        self.state_value = QLabel("DISCONNECTED")
        self.marker_value = QLabel("--")
        self.distance_value = QLabel("--")
        self.reprojection_value = QLabel("--")
        self.rate_value = QLabel("0.0 Hz")
        self.latency_value = QLabel("0.0 ms")
        self.ekf_value = QLabel("等待观测")
        self.mpc_value = QLabel("等待求解")
        self.robot_value = QLabel("未连接")
        self.safety_value = QLabel("DISCONNECTED")
        self.message_value = QLabel("等待连接")
        self.message_value.setWordWrap(True)
        for label, widget in (
            ("状态机", self.state_value),
            ("标签 ID", self.marker_value),
            ("目标距离", self.distance_value),
            ("重投影误差", self.reprojection_value),
            ("识别率", self.rate_value),
            ("端到端延迟", self.latency_value),
            ("EKF", self.ekf_value),
            ("MPC", self.mpc_value),
            ("机器人", self.robot_value),
            ("安全门", self.safety_value),
            ("提示", self.message_value),
        ):
            status_form.addRow(label, widget)
        side.addWidget(status_box)

        operation_box = QGroupBox("安全操作顺序")
        operation = QGridLayout(operation_box)
        self.connect_camera_button = QPushButton("1. 连接相机/预览")
        self.connect_robot_button = QPushButton("2. 连接 UR5 只读状态")
        self.system_check_button = QPushButton("3. 系统检查")
        self.dry_run_button = QPushButton("4. 干运行")
        self.speed_combo = QComboBox()
        self.speed_combo.addItem("联调档（50 mm/s）", "commissioning")
        self.speed_combo.addItem("试验档（120 mm/s）", "experiment")
        self.execution_combo = QComboBox()
        self.execution_combo.addItem("仿真/干运行", "dry_run")
        self.execution_combo.addItem("实机（需确认）", "hardware")
        self.enable_button = QPushButton("5. 使能")
        self.start_follow_button = QPushButton("6. 开始跟随")
        self.stop_button = QPushButton("立即停止")
        self.acknowledge_button = QPushButton("人工确认/重置")
        self.save_video_checkbox = QCheckBox("保存视频（默认关闭）")
        widgets = [
            self.connect_camera_button,
            self.connect_robot_button,
            self.system_check_button,
            self.dry_run_button,
        ]
        for row, widget in enumerate(widgets):
            operation.addWidget(widget, row, 0, 1, 2)
        operation.addWidget(QLabel("限速档"), 4, 0)
        operation.addWidget(self.speed_combo, 4, 1)
        operation.addWidget(QLabel("执行后端"), 5, 0)
        operation.addWidget(self.execution_combo, 5, 1)
        operation.addWidget(self.enable_button, 6, 0)
        operation.addWidget(self.start_follow_button, 6, 1)
        operation.addWidget(self.stop_button, 7, 0)
        operation.addWidget(self.acknowledge_button, 7, 1)
        operation.addWidget(self.save_video_checkbox, 8, 0, 1, 2)
        side.addWidget(operation_box)
        side.addStretch(1)

        self.connect_camera_button.clicked.connect(self._connect_camera)
        self.connect_robot_button.clicked.connect(self._connect_robot)
        self.system_check_button.clicked.connect(self._system_check)
        self.dry_run_button.clicked.connect(self._dry_run)
        self.speed_combo.currentIndexChanged.connect(self._select_speed)
        self.execution_combo.currentIndexChanged.connect(self._select_execution)
        self.enable_button.clicked.connect(self._enable)
        self.start_follow_button.clicked.connect(self._start_follow)
        self.stop_button.clicked.connect(self._stop)
        self.acknowledge_button.clicked.connect(self._acknowledge)
        self.save_video_checkbox.toggled.connect(self._toggle_video)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh_snapshot)
        self._timer.start(50)
        self.refresh_snapshot()

    def _ask_confirmation(self, title: str, text: str) -> bool:
        return (
            QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No)
            == QMessageBox.Yes
        )

    def _run_action(self, action: Callable[[], Any]) -> None:
        """相机/网络/系统检查等动作必须离开 UI 线程执行。"""

        if self._lifecycle_workers:
            return
        self._busy = True
        worker = _LifecycleAction(action)
        self._workers.add(worker)
        self._lifecycle_workers.add(worker)
        worker.signals.succeeded.connect(
            lambda _result, item=worker: self._finish_action(item)
        )
        worker.signals.failed.connect(
            lambda message, item=worker: self._fail_action(item, message)
        )
        self.refresh_snapshot()
        self._thread_pool.start(worker)

    def _finish_action(self, worker: _LifecycleAction) -> None:
        self._workers.discard(worker)
        self._lifecycle_workers.discard(worker)
        self._busy = bool(self._lifecycle_workers)
        if self._closing:
            if self._shutdown_completed_while_busy and not self._lifecycle_workers:
                self._closing = False
                self._close_ready = True
                self.close()
                return
            self._start_shutdown_worker()
            return
        self.refresh_snapshot()

    def _fail_action(self, worker: _LifecycleAction, message: str) -> None:
        self._workers.discard(worker)
        self._lifecycle_workers.discard(worker)
        self._busy = bool(self._lifecycle_workers)
        if self._closing:
            self.message_value.setText(f"关闭前的后台操作失败：{message}")
            if self._shutdown_completed_while_busy and not self._lifecycle_workers:
                self._closing = False
                self._close_ready = True
                self.close()
                return
            self._start_shutdown_worker()
            return
        self.message_value.setText(f"操作失败：{message}")
        self.refresh_snapshot()
        QMessageBox.critical(self, "操作失败", message)

    def _connect_camera(self) -> None:
        self._run_action(self.orchestrator.start_preview)

    def _connect_robot(self) -> None:
        self._run_action(self.orchestrator.connect_robot_read_only)

    def _system_check(self) -> None:
        self._run_action(self.orchestrator.system_check)

    def _dry_run(self) -> None:
        self._run_action(self.orchestrator.start_dry_run)

    def _select_speed(self) -> None:
        if hasattr(self.orchestrator, "select_speed_profile"):
            # QWidget 只能在主线程读取；worker 仅捕获不可变的普通字符串。
            profile = str(self.speed_combo.currentData())
            self._run_action(
                lambda value=profile: self.orchestrator.select_speed_profile(value)
            )

    def _select_execution(self) -> None:
        if not hasattr(self.orchestrator, "select_execution_mode"):
            return
        mode = self.execution_combo.currentData()
        confirmed = mode != "hardware" or self._confirm_callback(
            "实机确认",
            "将切换到 UR5 实机后端。请确认物理急停可用、工作区无人。",
        )
        if mode == "hardware" and not confirmed:
            self.execution_combo.setCurrentIndex(0)
            return
        self._run_action(
            lambda: self.orchestrator.select_execution_mode(mode, confirmed=confirmed)
        )

    def _enable(self) -> None:
        if self._confirm_callback(
            "使能确认", "确认标定、TCP=0、限速档和物理安全设施均正确？"
        ):
            self._run_action(self.orchestrator.enable)

    def _start_follow(self) -> None:
        if self._confirm_callback("开始跟随", "机器人将开始跟随固定 ID 标签，是否继续？"):
            self._run_action(self.orchestrator.start_following)

    def _stop(self) -> None:
        # 停止不能排在耗时连接任务后面；直接触发运行器的并发安全停车入口。
        failure_message = None
        try:
            self.orchestrator.emergency_stop("用户点击停止")
        except Exception as error:
            failure_message = f"停车未确认：{error}。请立即使用物理急停并检查机器人状态。"
        self.refresh_snapshot()
        if failure_message is not None:
            self.message_value.setText(failure_message)

    def _acknowledge(self) -> None:
        if self._confirm_callback("人工确认", "确认故障原因已查清？确认后仍需重新系统检查。"):
            self._run_action(self.orchestrator.acknowledge_stop)

    def _toggle_video(self, enabled: bool) -> None:
        recorder = getattr(self.orchestrator, "video_recorder", None)
        if recorder is not None:
            recorder.set_enabled(bool(enabled))

    @staticmethod
    def _to_pixmap(image: np.ndarray) -> QPixmap:
        array = np.ascontiguousarray(image)
        if array.ndim == 2:
            qimage = QImage(
                array.data,
                array.shape[1],
                array.shape[0],
                array.strides[0],
                QImage.Format_Grayscale8,
            ).copy()
        else:
            qimage = QImage(
                array.data, array.shape[1], array.shape[0], array.strides[0], QImage.Format_BGR888
            ).copy()
        return QPixmap.fromImage(qimage)

    def refresh_snapshot(self) -> None:
        snapshot = self.orchestrator.ui_snapshot()
        state = snapshot.get("state", "DISCONNECTED")
        self.state_value.setText(state)
        self.mode_label.setText(
            f"模式：{snapshot.get('mode', '预览')} | 限速：{snapshot.get('speed_profile', '--')}"
        )
        marker_id = snapshot.get("marker_id")
        self.marker_value.setText("--" if marker_id is None else str(marker_id))
        distance = snapshot.get("distance_m")
        self.distance_value.setText("--" if distance is None else f"{distance:.3f} m")
        reprojection = snapshot.get("reprojection_error_px")
        self.reprojection_value.setText(
            "--" if reprojection is None else f"{reprojection:.2f} px"
        )
        self.rate_value.setText(f"{float(snapshot.get('recognition_rate_hz', 0.0)):.1f} Hz")
        self.latency_value.setText(f"{float(snapshot.get('latency_ms', 0.0)):.1f} ms")
        self.ekf_value.setText(str(snapshot.get("ekf_status", "等待观测")))
        self.mpc_value.setText(str(snapshot.get("mpc_status", "等待求解")))
        self.robot_value.setText(str(snapshot.get("robot_status", "未连接")))
        self.safety_value.setText(str(snapshot.get("safety_status", state)))
        self.message_value.setText(
            self._close_failure_message
            if self._close_failure_message is not None
            else str(snapshot.get("message", ""))
        )
        image = snapshot.get("image")
        if image is not None:
            pixmap = self._to_pixmap(image)
            self.video_label.setPixmap(
                pixmap.scaled(
                    self.video_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        self.connect_camera_button.setEnabled(state == "DISCONNECTED")
        robot_connected = bool(snapshot.get("robot_read_only_connected", False))
        self.connect_robot_button.setEnabled(state == "PREVIEW" and not robot_connected)
        self.system_check_button.setEnabled(state == "PREVIEW" and robot_connected)
        dry_run_completed = bool(snapshot.get("dry_run_completed", False))
        self.dry_run_button.setEnabled(state == "VALIDATED" and not dry_run_completed)
        self.speed_combo.setEnabled(state == "VALIDATED" and dry_run_completed)
        self.execution_combo.setEnabled(state == "VALIDATED" and dry_run_completed)
        self.enable_button.setEnabled(state == "VALIDATED" and dry_run_completed)
        self.start_follow_button.setEnabled(state == "ENABLED")
        self.stop_button.setEnabled(state in {"ENABLED", "FOLLOWING", "OCCLUSION_DECAY"})
        self.acknowledge_button.setEnabled(state in {"STOPPED", "FAULT"})
        if self._busy:
            for widget in (
                self.connect_camera_button,
                self.connect_robot_button,
                self.system_check_button,
                self.dry_run_button,
                self.speed_combo,
                self.execution_combo,
                self.enable_button,
                self.start_follow_button,
                self.acknowledge_button,
            ):
                widget.setEnabled(False)
            # 即使后台厂商调用阻塞，也始终保留人工停车入口。
            self.stop_button.setEnabled(True)
        if self._close_failure_message is not None:
            # 关闭未完成时窗口必须保持可交互，并允许操作员再次请求停车。
            self.stop_button.setEnabled(True)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API name
        if self._close_ready:
            self._timer.stop()
            event.accept()
            return
        event.ignore()
        if self._closing:
            return
        self._closing = True
        self._close_failure_message = None
        # 关闭窗口也先触发停车，再在工作线程等待相机/RTDE C 扩展退出。
        try:
            self.orchestrator.emergency_stop("窗口关闭/系统退出")
        except Exception as error:
            self.message_value.setText(
                f"停车未确认：{error}。请立即使用物理急停；正在尝试关闭其余资源。"
            )
        # shutdown 使用独立 worker，即使已有连接/使能动作阻塞也必须立即启动。
        self._start_shutdown_worker()

    def _start_shutdown_worker(self) -> None:
        if self._shutdown_worker_active:
            return
        self._shutdown_worker_active = True
        worker = _LifecycleAction(self.orchestrator.shutdown)
        self._workers.add(worker)
        worker.signals.succeeded.connect(
            lambda _result, item=worker: self._finish_close(item)
        )
        worker.signals.failed.connect(
            lambda message, item=worker: self._fail_close(item, message)
        )
        self._thread_pool.start(worker)

    def _finish_close(self, worker: _LifecycleAction) -> None:
        self._workers.discard(worker)
        self._shutdown_worker_active = False
        if self._lifecycle_workers:
            # 资源关闭已经执行，但原 worker 仍可能发回 Qt 信号；保留窗口对象，
            # 待该 worker 返回后再销毁，避免回调访问已删除的 QWidget。
            self._shutdown_completed_while_busy = True
            self.message_value.setText("资源关闭完成，等待当前后台调用退出……")
            return
        self._closing = False
        self._close_ready = True
        self.close()

    def _fail_close(self, worker: _LifecycleAction, message: str) -> None:
        self._workers.discard(worker)
        self._shutdown_worker_active = False
        self._shutdown_completed_while_busy = False
        self._closing = False
        self._busy = bool(self._lifecycle_workers)
        if "停车未确认" in message or "连接仍在进行" in message:
            message = f"{message}；请立即使用物理急停，并在解除阻塞后重试关闭"
        self._close_failure_message = f"关闭失败，可再次关闭重试：{message}"
        self.message_value.setText(self._close_failure_message)
        self.stop_button.setEnabled(True)
        self.show()
