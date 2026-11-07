from __future__ import annotations

import csv
import time

from visual_servo_system.diagnostics.performance_profiler import (
    PerformanceMode,
    PerformanceProfiler,
)


def test_performance_profiler_writes_high_resolution_stage_records(tmp_path) -> None:
    output = tmp_path / "performance_stages.csv"
    profiler = PerformanceProfiler(output, queue_capacity=16)

    profiler.record(
        source="vision",
        stage="apriltag_search",
        kind="compute",
        duration_ns=1_234_567,
        sequence_number=7,
        timestamp_uncertainty_ms=24.1,
        detail="full_frame",
    )
    profiler.close()
    profiler.close()

    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["source"] == "vision"
    assert rows[0]["stage"] == "apriltag_search"
    assert rows[0]["kind"] == "compute"
    assert int(rows[0]["duration_ns"]) == 1_234_567
    assert float(rows[0]["duration_ms"]) == 1.234567
    assert int(rows[0]["sequence_number"]) == 7
    assert float(rows[0]["timestamp_uncertainty_ms"]) == 24.1
    assert rows[0]["detail"] == "full_frame"
    assert profiler.dropped_records == 0


def test_off_mode_has_no_writer_and_creates_no_diagnostic_file(tmp_path) -> None:
    output = tmp_path / "performance_stages.csv"
    profiler = PerformanceProfiler(output, mode=PerformanceMode.OFF)

    assert not profiler.enabled
    profiler.record(source="vision", stage="total", kind="total", duration_ns=10)
    profiler.close()

    assert not output.exists()
    assert not output.with_suffix(".meta.json").exists()


def test_full_async_submits_an_entire_cycle_with_one_queue_operation(tmp_path) -> None:
    output = tmp_path / "performance_stages.csv"
    profiler = PerformanceProfiler(output, queue_capacity=8)

    accepted = profiler.record_stages(
        source="vision",
        sequence_number=9,
        stages=(
            ("grayscale", "compute", 100),
            ("apriltag_search", "compute", 200),
            ("total", "total", 350),
        ),
        timestamp_reliable=True,
    )
    profiler.close()

    assert accepted
    assert profiler.submitted_batches == 1
    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["stage"] for row in rows] == [
        "grayscale",
        "apriltag_search",
        "total",
    ]


def test_cycle_coalesces_legacy_stage_calls_into_one_queue_batch(tmp_path) -> None:
    profiler = PerformanceProfiler(tmp_path / "performance_stages.csv")

    with profiler.cycle():
        profiler.record(source="vision", stage="detect", kind="compute", duration_ns=10)
        profiler.record(source="vision", stage="pnp", kind="compute", duration_ns=20)
        profiler.record(source="vision", stage="total", kind="total", duration_ns=40)
    profiler.close()

    assert profiler.submitted_batches == 1


def test_bounded_queue_drops_ordinary_diagnostics_without_blocking(tmp_path) -> None:
    output = tmp_path / "performance_stages.csv"
    profiler = PerformanceProfiler(
        output,
        queue_capacity=1,
        writer_start_gate=lambda: time.sleep(0.05),
    )

    started = time.perf_counter()
    for sequence in range(200):
        profiler.record(
            source="vision",
            stage="total",
            kind="total",
            duration_ns=sequence,
        )
    submission_elapsed = time.perf_counter() - started
    profiler.close(timeout_s=1.0)

    assert submission_elapsed < 0.05
    assert profiler.dropped_records > 0


def test_light_mode_writes_periodic_style_summary_not_per_event_rows(tmp_path) -> None:
    output = tmp_path / "performance_stages.csv"
    profiler = PerformanceProfiler(output, mode="LIGHT", queue_capacity=8)
    for duration in (1_000_000, 2_000_000, 3_000_000):
        profiler.record(
            source="vision", stage="total", kind="total", duration_ns=duration
        )
    profiler.close()

    summary = output.with_name("performance_summary.csv")
    with summary.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["source"] == "vision"
    assert rows[0]["stage"] == "total"
    assert int(rows[0]["count"]) == 3
    assert float(rows[0]["mean_ms"]) == 2.0


def test_writer_failure_is_observable_but_never_raised_in_producer(tmp_path) -> None:
    output = tmp_path / "missing" / "performance_stages.csv"
    profiler = PerformanceProfiler(output, create_parent=False)
    time.sleep(0.02)

    # 诊断落盘失败不能把视觉、RTDE或控制线程变成故障线程。
    profiler.record(source="vision", stage="total", kind="total", duration_ns=1)
    profiler.close()

    assert profiler.writer_error is not None
