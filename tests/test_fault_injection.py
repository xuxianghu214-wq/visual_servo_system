from __future__ import annotations

import pytest

from visual_servo_system.offline.fault_injection import (
    OFFLINE_FAULT_SCENARIOS,
    OfflineFaultHarness,
)


@pytest.mark.parametrize("scenario", OFFLINE_FAULT_SCENARIOS)
def test_every_offline_fault_latches_stop_without_motion(scenario: str) -> None:
    harness = OfflineFaultHarness()

    result = harness.inject(scenario)

    assert result.fault_latched
    assert result.motion_commands_sent == 0
    assert not result.control_interface_constructed
    assert not result.motion_enabled


def test_offline_fault_harness_rejects_motion_even_before_fault() -> None:
    harness = OfflineFaultHarness()

    with pytest.raises(RuntimeError, match="offline"):
        harness.send_motion_forbidden()
