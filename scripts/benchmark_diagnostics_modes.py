"""离线比较旧式同步元数据写入与三种新诊断模式。"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from visual_servo_system.diagnostics.performance_profiler import (  # noqa: E402
    PerformanceMode,
    PerformanceProfiler,
)

STAGES = (
    ("frame_queue_age", "queue", 0),
    ("gray_prepare", "compute", 1_000_000),
    ("scaled_search", "compute", 8_000_000),
    ("corner_refinement", "compute", 2_000_000),
    ("pnp", "compute", 3_000_000),
    ("ekf_update", "compute", 2_000_000),
    ("vision_step_total", "total", 16_000_000),
)


def _stats_ms(values_ns: list[int]) -> dict[str, float]:
    values = np.asarray(values_ns, dtype=np.float64) / 1_000_000.0
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
    }


def _run_mode(mode: PerformanceMode, cycles: int, directory: Path) -> dict[str, object]:
    profiler = PerformanceProfiler(
        directory / f"{mode.value.lower()}_performance_stages.csv",
        mode=mode,
        queue_capacity=4096,
        flush_every_records=256,
    )
    submission_ns: list[int] = []
    tracemalloc.start()
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    for sequence in range(cycles):
        started_ns = time.perf_counter_ns()
        profiler.record_stages(
            source="offline_synthetic_vision",
            sequence_number=sequence,
            stages=STAGES,
            timestamp_source="offline_simulated_camera",
            timestamp_reliable=True,
        )
        submission_ns.append(time.perf_counter_ns() - started_ns)
    profiler.close(timeout_s=5.0)
    wall_s = time.perf_counter() - wall_started
    cpu_s = time.process_time() - cpu_started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "mode": mode.value,
        "cycles": cycles,
        "synthetic_cycle_rate_hz": cycles / wall_s,
        "submission": _stats_ms(submission_ns),
        "dropped_records": profiler.dropped_records,
        "submitted_batches": profiler.submitted_batches,
        "cpu_time_s": cpu_s,
        "wall_time_s": wall_s,
        "single_process_cpu_percent": 100.0 * cpu_s / wall_s,
        "python_peak_memory_mib": peak_bytes / (1024.0 * 1024.0),
        "writer_error": None if profiler.writer_error is None else repr(profiler.writer_error),
        "scope": "offline synthetic diagnostics submission; not camera or UR hardware FPS",
    }


def _run_legacy_simulation(cycles: int, directory: Path) -> dict[str, object]:
    """复现修复前每阶段在生产线程重写 meta JSON 的关键开销。"""

    output = directory / "legacy_meta.json"
    submission_ns: list[int] = []
    tracemalloc.start()
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    for sequence in range(cycles):
        started_ns = time.perf_counter_ns()
        for stage, kind, duration_ns in STAGES:
            output.write_text(
                json.dumps(
                    {
                        "sequence": sequence,
                        "stage": stage,
                        "kind": kind,
                        "duration_ns": duration_ns,
                        "closed_cleanly": True,
                        "dropped_records": 0,
                    }
                ),
                encoding="utf-8",
            )
        submission_ns.append(time.perf_counter_ns() - started_ns)
    wall_s = time.perf_counter() - wall_started
    cpu_s = time.process_time() - cpu_started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "mode": "LEGACY_SYNC_META_SIMULATION",
        "cycles": cycles,
        "synthetic_cycle_rate_hz": cycles / wall_s,
        "submission": _stats_ms(submission_ns),
        "dropped_records": 0,
        "submitted_batches": cycles * len(STAGES),
        "cpu_time_s": cpu_s,
        "wall_time_s": wall_s,
        "single_process_cpu_percent": 100.0 * cpu_s / wall_s,
        "python_peak_memory_mib": peak_bytes / (1024.0 * 1024.0),
        "writer_error": None,
        "scope": "offline reproduction of removed synchronous meta-file writes only",
    }


def run_benchmark(cycles: int, output_directory: Path) -> dict[str, object]:
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="visual_servo_diag_", dir=output_directory
    ) as temporary:
        scratch = Path(temporary)
        results = [_run_legacy_simulation(cycles, scratch)]
        results.extend(
            _run_mode(mode, cycles, scratch)
            for mode in (
                PerformanceMode.OFF,
                PerformanceMode.LIGHT,
                PerformanceMode.FULL_ASYNC,
            )
        )
    report = {
        "generated_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "median_submission_improvement_vs_legacy": {
            result["mode"]: (
                results[0]["submission"]["p50_ms"]
                / max(result["submission"]["p50_ms"], 1e-12)
            )
            for result in results[1:]
        },
        "disclaimer": (
            "All values are offline diagnostics microbenchmarks. They do not prove 30 FPS "
            "camera processing, 125 Hz RTDE, or closed-loop robot performance."
        ),
        "repeatability_note": f"cycles={cycles}; stage_count={len(STAGES)}",
        "submission_p50_median_ms": statistics.median(
            result["submission"]["p50_ms"] for result in results[1:]
        ),
    }
    (output_directory / "diagnostics_modes_benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="三种性能诊断模式的离线微基准")
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "runtime_data" / "offline_validation",
    )
    arguments = parser.parse_args()
    report = run_benchmark(arguments.cycles, arguments.output_directory)
    for result in report["results"]:
        print(
            f"{result['mode']}: P50={result['submission']['p50_ms']:.4f} ms, "
            f"P95={result['submission']['p95_ms']:.4f} ms, "
            f"drop={result['dropped_records']}"
        )
    print(arguments.output_directory / "diagnostics_modes_benchmark.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
