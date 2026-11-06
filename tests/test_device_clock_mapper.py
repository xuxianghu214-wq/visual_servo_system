from __future__ import annotations

import pytest

from visual_servo_system.runtime.device_clock_mapper import DeviceClockMapper


def test_mapper_tracks_drift_with_positive_transport_jitter() -> None:
    mapper = DeviceClockMapper(min_samples=5, max_uncertainty_s=0.004)
    delays = (0.0030, 0.0010, 0.0020, 0.0005, 0.0015, 0.0008)
    estimate = None
    for index, delay in enumerate(delays):
        device_s = 100.0 + index
        true_host_s = 20.0 + device_s * 1.0001
        estimate = mapper.observe(device_s, true_host_s + delay)

    assert estimate is not None
    assert estimate.sample_count == len(delays)
    assert estimate.reliable
    assert estimate.host_timestamp_s == pytest.approx(20.0 + 105.0 * 1.0001, abs=0.003)
    assert estimate.uncertainty_s <= 0.004
    assert estimate.residual_rms_s >= 0.0


def test_mapper_resets_on_device_clock_rollback() -> None:
    mapper = DeviceClockMapper(min_samples=3)
    for index in range(3):
        assert mapper.observe(10.0 + index, 50.0 + index).sample_count == index + 1
    rolled_back = mapper.observe(1.0, 60.0)
    assert rolled_back.sample_count == 1
    assert not rolled_back.reliable
    assert rolled_back.reset_count == 1
