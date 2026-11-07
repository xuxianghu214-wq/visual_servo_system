"""运行时钟回放和无运动故障注入，不导入 UR 控制接口。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from visual_servo_system.offline.fault_injection import (  # noqa: E402
    OFFLINE_FAULT_SCENARIOS,
    OfflineFaultHarness,
)
from visual_servo_system.offline.robot_state_replay import (  # noqa: E402
    RobotReplayConfig,
    generate_robot_replay,
)
from visual_servo_system.runtime.device_clock_mapper import (  # noqa: E402
    DeviceClockMapper,
)


def _mapping_case(name: str, config: RobotReplayConfig) -> dict[str, object]:
    mapper = DeviceClockMapper(
        min_samples=4,
        max_uncertainty_s=0.005,
        reset_backjump_s=0.5,
    )
    statuses: Counter[str] = Counter()
    sample_count = 0
    for sample in generate_robot_replay(config):
        estimate = mapper.observe(sample.controller_timestamp_s, sample.receive_host_s)
        statuses[estimate.status.value] += 1
        sample_count += 1
    return {
        "name": name,
        "input_samples": sample_count,
        "status_counts": dict(statuses),
        "mapper_reset_count": mapper.reset_count,
        "duplicate_count": mapper.duplicate_count,
        "delayed_count": mapper.delayed_count,
        "out_of_order_count": mapper.out_of_order_count,
        "provenance": "synthetic RTDE timing; logic test only",
    }


def run_validation(output_directory: Path) -> dict[str, object]:
    mapping_cases = [
        _mapping_case("normal_125_hz", RobotReplayConfig(frequency_hz=125.0)),
        _mapping_case("normal_77_hz", RobotReplayConfig(frequency_hz=77.0)),
        _mapping_case(
            "fault_mix",
            RobotReplayConfig(
                frequency_hz=50.0,
                duplicate_indices=(10, 20),
                drop_indices=(25, 26),
                reorder_pairs=((30, 31),),
                delay_indices=(35,),
                injected_delay_s=0.050,
                jitter_s=0.0005,
                drift_ppm=100.0,
                reset_index=40,
                reset_to_s=0.0,
            ),
        ),
    ]
    fault_results = []
    for scenario in OFFLINE_FAULT_SCENARIOS:
        result = OfflineFaultHarness().inject(scenario)
        fault_results.append(
            {
                "scenario": result.scenario,
                "fault_latched": result.fault_latched,
                "motion_enabled": result.motion_enabled,
                "motion_commands_sent": result.motion_commands_sent,
                "control_interface_constructed": result.control_interface_constructed,
            }
        )
    report = {
        "mapping_cases": mapping_cases,
        "fault_results": fault_results,
        "all_faults_no_motion": all(
            result["fault_latched"]
            and not result["motion_enabled"]
            and result["motion_commands_sent"] == 0
            and not result["control_interface_constructed"]
            for result in fault_results
        ),
        "hardware_claims": "none; current computer has no camera or UR5",
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "offline_validation_summary.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="UR5 视觉伺服离线无运动验证")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "runtime_data" / "offline_validation",
    )
    arguments = parser.parse_args()
    report = run_validation(arguments.output_directory)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_faults_no_motion"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
