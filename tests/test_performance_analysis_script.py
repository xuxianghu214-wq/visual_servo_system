from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from visual_servo_system.diagnostics.performance_profiler import PerformanceProfiler

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_performance_analysis_script_summarizes_stages_queue_and_timestamps(
    tmp_path: Path,
) -> None:
    session = tmp_path / "session"
    profiler = PerformanceProfiler(session / "performance_stages.csv")
    profiler.record(
        source="vision",
        stage="frame_queue_age",
        kind="queue",
        duration_ns=0,
        sequence_number=10,
        queue_age_ms=4.0,
        capture_age_ms=12.0,
        frames_skipped=2,
    )
    profiler.record(
        source="vision_detector",
        stage="apriltag_search",
        kind="compute",
        duration_ns=20_000_000,
        sequence_number=10,
    )
    profiler.record(
        source="camera",
        stage="timestamp_mapping",
        kind="compute",
        duration_ns=50_000,
        sequence_number=10,
        timestamp_reliable=True,
        timestamp_uncertainty_ms=3.0,
        mapped_transport_delay_ms=2.0,
        clock_residual_rms_ms=1.0,
    )
    profiler.close()

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "analyze_performance_log.py"),
            str(session),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(
        (session / "performance_analysis_summary.json").read_text(encoding="utf-8")
    )
    assert result["queue"]["frames_skipped"] == 2
    assert result["queue"]["queue_age_ms"]["p50"] == 4.0
    assert result["timestamps"]["camera"]["reliable_rate"] == 1.0
    assert (session / "performance_stage_summary.csv").is_file()
    assert (session / "performance_analysis.md").is_file()
