"""安全地检查运行环境；默认不连接相机、不连接 UR5。"""

from __future__ import annotations

import argparse
import importlib
import platform
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-camera", action="store_true", help="显式请求相机探测")
    parser.add_argument("--probe-robot", action="store_true", help="显式请求 UR5 只读探测")
    args = parser.parse_args()
    failures: list[str] = []
    print(f"Python: {platform.python_version()} ({struct.calcsize('P') * 8}-bit)")
    if sys.version_info[:2] != (3, 11) or struct.calcsize("P") != 8:
        failures.append("需要 Python 3.11 x64")
    modules = (
        "numpy",
        "cv2",
        "PySide6",
        "harvesters",
        "genicam",
        "yaml",
        "scipy",
        "osqp",
        "rtde_control",
    )
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"[OK] {module}")
        except Exception as error:
            failures.append(f"{module}: {error}")
    try:
        from visual_servo_system.config.config_loader import (
            load_camera_intrinsics,
            load_hand_eye_calibration,
            load_system_config,
        )

        config = load_system_config(ROOT / "config_files")
        print(f"[OK] camera config: BFS-U3-13Y3C / {config.camera.device_serial}")
        try:
            load_camera_intrinsics(
                ROOT / "calibration_files" / "camera_intrinsics.yaml",
                expected_resolution=(config.camera.image_width_px, config.camera.image_height_px),
            )
            load_hand_eye_calibration(
                ROOT / "calibration_files" / "hand_eye_eye_in_hand.yaml"
            )
            print("[OK] 标定文件已完成")
        except Exception as error:
            print(f"[PREVIEW ONLY] 标定未完成：{error}")
    except Exception as error:
        failures.append(f"配置错误: {error}")
    try:
        from visual_servo_system.camera.spinnaker_cti_discovery import (
            discover_cti_files,
            require_spinnaker_cti,
        )

        cti = require_spinnaker_cti(discover_cti_files())
        print(f"[OK] Spinnaker CTI: {cti}")
    except Exception as error:
        failures.append(f"Spinnaker CTI: {error}")
    if args.probe_camera:
        print("[INFO] 相机探测请在界面中点击“连接相机/预览”执行。")
    if args.probe_robot:
        print("[INFO] UR5 探测请在界面中点击“连接 UR5 只读状态”执行。")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("环境基础检查通过。未执行任何机器人运动。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
