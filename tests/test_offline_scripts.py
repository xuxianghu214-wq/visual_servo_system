from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_offline_validation_script_never_constructs_motion_interface(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_offline_validation.py"),
            "--output-directory",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(
        (tmp_path / "offline_validation_summary.json").read_text(encoding="utf-8")
    )
    assert report["all_faults_no_motion"]
    assert all(
        not item["control_interface_constructed"] for item in report["fault_results"]
    )


def test_diagnostics_benchmark_exports_all_modes_and_percentiles(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "benchmark_diagnostics_modes.py"),
            "--cycles",
            "20",
            "--output-directory",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(
        (tmp_path / "diagnostics_modes_benchmark.json").read_text(encoding="utf-8")
    )
    assert [item["mode"] for item in report["results"]] == [
        "LEGACY_SYNC_META_SIMULATION",
        "OFF",
        "LIGHT",
        "FULL_ASYNC",
    ]
    assert all("p99_ms" in item["submission"] for item in report["results"])
