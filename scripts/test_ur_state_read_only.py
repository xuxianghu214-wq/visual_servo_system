"""实验电脑专用：只构造 RTDEReceiveInterface，绝不构造控制接口。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from visual_servo_system.config.config_loader import load_system_config  # noqa: E402
from visual_servo_system.robot.ur_rtde_state_reader import UrRtdeStateReader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="UR RTDE 只读状态与时钟分类测试")
    parser.add_argument("--samples", type=int, default=250)
    arguments = parser.parse_args()
    if arguments.samples <= 0:
        parser.error("samples must be positive")
    config = load_system_config(ROOT / "config_files")
    reader = UrRtdeStateReader(
        config.robot.host,
        frequency_hz=config.robot.state_frequency_hz,
    )
    try:
        reader.connect()
        accepted = 0
        while accepted < arguments.samples:
            sample = reader.read_once()
            if sample is not None:
                accepted += 1
    finally:
        reader.close()
    print(json.dumps(asdict(reader.sample_diagnostics), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
