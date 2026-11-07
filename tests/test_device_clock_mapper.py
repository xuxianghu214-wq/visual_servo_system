from __future__ import annotations

import pytest

from visual_servo_system.runtime.device_clock_mapper import (
    ClockSampleStatus,
    DeviceClockMapper,
)


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
    assert rolled_back.status is ClockSampleStatus.RESET


def test_duplicate_timestamp_is_rejected_without_reset_or_new_mapping_sample() -> None:
    mapper = DeviceClockMapper(min_samples=3)
    for index in range(3):
        mapper.observe(10.0 + index, 50.0 + index)

    duplicate = mapper.observe(12.0, 52.01)

    assert duplicate.status is ClockSampleStatus.DUPLICATE
    assert duplicate.sample_count == 3
    assert duplicate.reset_count == 0
    assert mapper.duplicate_count == 1


def test_small_backward_step_is_out_of_order_but_material_step_is_reset() -> None:
    mapper = DeviceClockMapper(min_samples=3, reset_backjump_s=1.0)
    for index in range(3):
        mapper.observe(10.0 + index, 50.0 + index)

    out_of_order = mapper.observe(11.75, 52.01)
    assert out_of_order.status is ClockSampleStatus.OUT_OF_ORDER
    assert out_of_order.sample_count == 3
    assert out_of_order.reset_count == 0
    assert mapper.out_of_order_count == 1

    reset = mapper.observe(1.0, 53.0)
    assert reset.status is ClockSampleStatus.RESET
    assert reset.sample_count == 1
    assert reset.reset_count == 1


def test_excessively_delayed_unique_sample_is_classified_without_poisoning_fit() -> None:
    mapper = DeviceClockMapper(min_samples=3, max_uncertainty_s=0.005)
    for index in range(4):
        mapper.observe(10.0 + index, 50.001 + index)

    delayed = mapper.observe(14.0, 54.050)

    assert delayed.status is ClockSampleStatus.DELAYED
    assert not delayed.reliable
    assert delayed.sample_count == 4
    assert mapper.delayed_count == 1
