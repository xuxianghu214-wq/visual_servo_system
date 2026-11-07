from __future__ import annotations

import numpy as np

from visual_servo_system.offline.image_folder_replay import (
    ImageReplayConfig,
    replay_images,
)
from visual_servo_system.offline.robot_state_replay import (
    RobotReplayConfig,
    generate_robot_replay,
)
from visual_servo_system.runtime.device_clock_mapper import (
    ClockSampleStatus,
    DeviceClockMapper,
)


def test_image_replay_preserves_capture_identity_through_drop_duplicate_and_reorder() -> None:
    images = [np.full((4, 6), index, np.uint8) for index in range(6)]
    events = list(
        replay_images(
            images,
            ImageReplayConfig(
                fps=30.0,
                start_host_s=10.0,
                drop_indices=(1,),
                duplicate_indices=(2,),
                reorder_pairs=((3, 4),),
            ),
        )
    )

    identities = [event.frame.sequence_number for event in events]
    assert 1 not in identities
    assert identities.count(2) == 2
    assert identities.index(4) < identities.index(3)
    assert all(event.simulated for event in events)
    assert all(event.frame.timestamp_source == "offline_simulated_camera" for event in events)
    assert events[0].frame.host_timestamp_s == 10.0
    assert events[0].frame.receive_timestamp_s >= events[0].frame.host_timestamp_s


def test_robot_replay_generates_requested_nominal_frequencies() -> None:
    samples_125 = list(
        generate_robot_replay(RobotReplayConfig(frequency_hz=125.0, duration_s=1.0))
    )
    samples_77 = list(
        generate_robot_replay(RobotReplayConfig(frequency_hz=77.0, duration_s=1.0))
    )

    assert len(samples_125) == 125
    assert len(samples_77) == 77
    assert all(sample.simulated for sample in samples_125 + samples_77)


def test_robot_replay_exercises_duplicate_out_of_order_delay_drift_and_reset() -> None:
    samples = list(
        generate_robot_replay(
            RobotReplayConfig(
                frequency_hz=20.0,
                duration_s=1.0,
                duplicate_indices=(4,),
                reorder_pairs=((7, 8),),
                delay_indices=(10,),
                fixed_delay_s=0.001,
                injected_delay_s=0.050,
                drift_ppm=100.0,
                reset_index=15,
                reset_to_s=0.0,
            )
        )
    )
    mapper = DeviceClockMapper(
        min_samples=3,
        max_uncertainty_s=0.005,
        reset_backjump_s=0.5,
    )
    statuses = [
        mapper.observe(sample.controller_timestamp_s, sample.receive_host_s).status
        for sample in samples
    ]

    assert ClockSampleStatus.DUPLICATE in statuses
    assert ClockSampleStatus.OUT_OF_ORDER in statuses
    assert ClockSampleStatus.DELAYED in statuses
    assert ClockSampleStatus.RESET in statuses
    assert mapper.reset_count == 1
