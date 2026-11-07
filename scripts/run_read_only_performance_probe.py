"""只读运行相机、RTDE状态和视觉线程，采集性能数据；绝不使能机器人。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from visual_servo_system.runtime.production_factory import (  # noqa: E402
    build_production_orchestrator,
)


def _wait_for_value(buffer, timeout_s: float, name: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if buffer.get_latest() is not None:
            return
        time.sleep(0.02)
    raise TimeoutError(f"等待{name}超时")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="UR5+Blackfly S 只读性能探针（不执行干运行、使能或 servoL）"
    )
    parser.add_argument("--duration-s", type=float, default=30.0, help="采样时长，默认30秒")
    parser.add_argument("--startup-timeout-s", type=float, default=8.0)
    parser.add_argument(
        "--performance-mode",
        choices=("OFF", "LIGHT", "FULL_ASYNC"),
        default="FULL_ASYNC",
        help="只读性能探针默认详细异步记录",
    )
    arguments = parser.parse_args()
    if arguments.duration_s <= 0.0 or arguments.startup_timeout_s <= 0.0:
        parser.error("时长必须为正数")

    orchestrator = build_production_orchestrator(
        ROOT, performance_mode_override=arguments.performance_mode
    )
    session_directory = orchestrator.logger.session_directory
    print(f"性能会话目录：{session_directory}")
    print("安全模式：只连接相机和 UR5 只读状态，不创建控制连接，不发送运动。")
    exit_code = 0
    try:
        orchestrator.start_preview()
        _wait_for_value(
            orchestrator.camera_worker.latest_frame,
            arguments.startup_timeout_s,
            "第一帧相机图像",
        )
        orchestrator.connect_robot_read_only()
        _wait_for_value(
            orchestrator.state_reader.latest_state,
            arguments.startup_timeout_s,
            "第一帧UR5只读状态",
        )
        if not orchestrator.system_check():
            raise RuntimeError(orchestrator.ui_snapshot()["message"])

        deadline = time.monotonic() + arguments.duration_s
        while time.monotonic() < deadline:
            if orchestrator.camera_worker.last_error is not None:
                raise RuntimeError(
                    f"相机线程异常：{orchestrator.camera_worker.last_error}"
                )
            if orchestrator.state_reader.last_error is not None:
                raise RuntimeError(f"RTDE线程异常：{orchestrator.state_reader.last_error}")
            if orchestrator.state.name == "FAULT":
                raise RuntimeError(orchestrator.ui_snapshot()["message"])
            time.sleep(0.05)
    except Exception as error:
        exit_code = 2
        print(f"性能探针失败：{error}", file=sys.stderr)
    finally:
        diagnostics = orchestrator.state_reader.sample_diagnostics
        print(
            "RTDE分类："
            f"accepted={diagnostics.accepted}, duplicate={diagnostics.duplicate}, "
            f"delayed={diagnostics.delayed}, out_of_order={diagnostics.out_of_order}, "
            f"reset={diagnostics.clock_reset}, missed_deadlines={diagnostics.missed_deadlines}"
        )
        try:
            orchestrator.shutdown()
        except Exception as error:
            exit_code = 3
            print(f"关闭失败：{error}", file=sys.stderr)

    print(f"采样结束。请保留整个目录：{session_directory}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
