"""实验电脑专用：只读取相机帧，不创建任何 UR 对象。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from visual_servo_system.camera.camera_gentl_capture import GentlCameraCapture  # noqa: E402
from visual_servo_system.config.config_loader import load_system_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Point Grey 相机只读取帧测试")
    parser.add_argument("--frames", type=int, default=30)
    arguments = parser.parse_args()
    if arguments.frames <= 0:
        parser.error("frames must be positive")
    config = load_system_config(ROOT / "config_files")
    capture = GentlCameraCapture(config.camera)
    try:
        capture.connect()
        for _ in range(arguments.frames):
            frame = capture.fetch_frame()
            print(
                frame.sequence_number,
                frame.image.shape,
                frame.timestamp_source,
                frame.timestamp_reliable,
            )
    finally:
        capture.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
