"""对已采集图像运行与生产一致的低分辨率搜索+原图 PnP。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from visual_servo_system.config.config_loader import (  # noqa: E402
    load_camera_intrinsics,
    load_system_config,
)
from visual_servo_system.diagnostics.performance_profiler import (  # noqa: E402
    PerformanceProfiler,
)
from visual_servo_system.offline.image_folder_replay import (  # noqa: E402
    ImageReplayConfig,
    replay_image_folder,
)
from visual_servo_system.vision.marker_pose_detector import (  # noqa: E402
    MarkerPoseDetector,
    MarkerPoseDetectorConfig,
)


def _percentiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AprilTag/ArUco 图像文件夹离线回放")
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--drop-probability", type=float, default=0.0)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    parser.add_argument("--jitter-ms", type=float, default=0.0)
    parser.add_argument(
        "--performance-mode", choices=("OFF", "LIGHT", "FULL_ASYNC"), default="LIGHT"
    )
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    config = load_system_config(ROOT / "config_files")
    intrinsics = load_camera_intrinsics(
        ROOT / "calibration_files" / "camera_intrinsics.yaml",
        expected_resolution=(
            config.camera.image_width_px,
            config.camera.image_height_px,
        ),
    )
    detector = MarkerPoseDetector(
        MarkerPoseDetectorConfig(
            detector=config.marker.detector,
            family=config.marker.family,
            aruco_dictionary=config.marker.aruco_dictionary,
            marker_id=config.marker.marker_id,
            marker_size_m=config.marker.marker_size_m,
            detection_scale=config.marker.detection_scale,
        ),
        intrinsics.camera_matrix,
        intrinsics.distortion_coefficients,
    )
    profiler = PerformanceProfiler(
        arguments.output_directory / "performance_stages.csv",
        mode=arguments.performance_mode,
        queue_capacity=config.diagnostics.queue_capacity,
        flush_every_records=config.diagnostics.batch_size,
        flush_interval_s=config.diagnostics.flush_interval_s,
        light_summary_interval_s=config.diagnostics.light_summary_interval_s,
    )
    replay_config = ImageReplayConfig(
        fps=arguments.fps,
        fixed_delay_s=arguments.delay_ms / 1000.0,
        jitter_s=arguments.jitter_ms / 1000.0,
        random_drop_probability=arguments.drop_probability,
    )
    result_path = arguments.output_directory / "image_replay_results.csv"
    durations_ms: list[float] = []
    valid_count = 0
    frame_count = 0
    try:
        with result_path.open("w", encoding="utf-8-sig", newline="") as stream:
            fields = [
                "sequence_number",
                "capture_host_s",
                "receive_host_s",
                "valid",
                "reprojection_error_px",
                "processing_ms",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for event in replay_image_folder(arguments.image_directory, replay_config):
                frame = event.frame
                expected_shape = (
                    config.camera.image_height_px,
                    config.camera.image_width_px,
                )
                if frame.image.shape[:2] != expected_shape:
                    raise ValueError(
                        f"frame {frame.sequence_number} resolution {frame.image.shape[:2]} "
                        f"does not match calibrated {expected_shape}"
                    )
                started_ns = time.perf_counter_ns()
                with profiler.cycle():
                    observation = detector.detect(frame.image, frame.host_timestamp_s)
                    elapsed_ns = time.perf_counter_ns() - started_ns
                    stages = [
                        (
                            stage,
                            "total" if stage == "detector_total" else "compute",
                            duration,
                        )
                        for stage, duration in detector.last_stage_timings_ns.items()
                    ]
                    stages.append(("image_replay_total", "total", elapsed_ns))
                    profiler.record_stages(
                        source="offline_image_replay",
                        sequence_number=frame.sequence_number,
                        stages=stages,
                        timestamp_source=frame.timestamp_source,
                        timestamp_reliable=frame.timestamp_reliable,
                    )
                elapsed_ms = elapsed_ns / 1_000_000.0
                durations_ms.append(elapsed_ms)
                frame_count += 1
                valid_count += int(observation.valid)
                writer.writerow(
                    {
                        "sequence_number": frame.sequence_number,
                        "capture_host_s": frame.host_timestamp_s,
                        "receive_host_s": frame.receive_timestamp_s,
                        "valid": observation.valid,
                        "reprojection_error_px": observation.reprojection_error_px,
                        "processing_ms": elapsed_ms,
                    }
                )
    finally:
        profiler.close()
    duration_summary = _percentiles(durations_ms)
    mean_ms = duration_summary["mean"]
    summary = {
        "frames": frame_count,
        "valid_frames": valid_count,
        "recognition_rate": valid_count / frame_count if frame_count else 0.0,
        "processing_ms": duration_summary,
        "processing_fps_from_compute_time": (
            None if mean_ms is None or mean_ms <= 0.0 else 1000.0 / mean_ms
        ),
        "diagnostic_dropped_records": profiler.dropped_records,
        "scope": "offline image replay; no robot control interface",
    }
    (arguments.output_directory / "image_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
