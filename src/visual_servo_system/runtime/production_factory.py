"""生产对象工厂：只构造对象，不连接相机或机器人。"""

from __future__ import annotations

from pathlib import Path

from visual_servo_system.camera.camera_gentl_capture import (
    CameraCaptureWorker,
    GentlCameraCapture,
)
from visual_servo_system.config.config_loader import load_system_config
from visual_servo_system.logging.optional_video_recorder import OptionalVideoRecorder
from visual_servo_system.logging.session_csv_logger import SessionCsvLogger
from visual_servo_system.robot.deferred_ur_rtde_servo_sender import DeferredUrRtdeServoSender
from visual_servo_system.robot.ur_rtde_state_reader import UrRtdeStateReader
from visual_servo_system.runtime.visual_servo_orchestrator import VisualServoOrchestrator


def build_production_orchestrator(project_root: str | Path) -> VisualServoOrchestrator:
    root = Path(project_root).resolve()
    config = load_system_config(root / "config_files")
    capture = GentlCameraCapture(config.camera)
    camera_worker = CameraCaptureWorker(capture)
    state_reader = UrRtdeStateReader(
        config.robot.host,
        frequency_hz=config.robot.state_frequency_hz,
    )
    sender = DeferredUrRtdeServoSender(config.robot.host, state_reader)
    logger = SessionCsvLogger(
        root / "experiment_logs",
        camera_model="BFS-U3-13Y3C",
        camera_serial=config.camera.device_serial or "",
        config_snapshot={
            "robot_host": config.robot.host,
            "marker_id": config.marker.marker_id,
            "marker_size_m": config.marker.marker_size_m,
            "marker_detection_scale": config.marker.detection_scale,
            "camera_resolution_px": [
                config.camera.image_width_px,
                config.camera.image_height_px,
            ],
        },
    )
    recorder = OptionalVideoRecorder(
        logger.session_directory / "preview.mp4",
        frame_size=(config.camera.image_width_px, config.camera.image_height_px),
        fps=config.camera.frame_rate_hz,
        enabled=False,
    )
    return VisualServoOrchestrator(
        camera_worker=camera_worker,
        state_reader=state_reader,
        real_sender=sender,
        detector=None,
        project_root=root,
        logger=logger,
        video_recorder=recorder,
    )
