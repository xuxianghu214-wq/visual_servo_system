"""汇总 performance_stages.csv，并输出可审计的统计表和 Markdown 报告。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _stats(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _format(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def analyze(session_directory: Path) -> dict[str, object]:
    performance_path = session_directory / "performance_stages.csv"
    rows = _read_csv(performance_path)
    if not rows:
        raise ValueError(f"没有可分析的性能记录：{performance_path}")

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        duration = _finite_float(row.get("duration_ms"))
        if duration is not None:
            grouped[(row["source"], row["stage"], row["kind"])].append(duration)
    stage_summary = []
    for (source, stage, kind), values in grouped.items():
        stage_summary.append(
            {"source": source, "stage": stage, "kind": kind, **_stats(values)}
        )
    stage_summary.sort(
        key=lambda item: (
            -float(item["mean"] if item["mean"] is not None else -1.0),
            str(item["source"]),
            str(item["stage"]),
        )
    )

    queue_rows = [row for row in rows if row["stage"] == "frame_queue_age"]
    queue_age = [
        value
        for row in queue_rows
        if (value := _finite_float(row.get("queue_age_ms"))) is not None
    ]
    capture_age = [
        value
        for row in queue_rows
        if (value := _finite_float(row.get("capture_age_ms"))) is not None
    ]
    skipped = sum(int(row.get("frames_skipped") or 0) for row in queue_rows)
    duplicate_rejections = sum(
        1 for row in rows if row["stage"] == "duplicate_frame_rejected"
    )

    timestamp_summary: dict[str, object] = {}
    for source in ("camera", "robot"):
        mapping_rows = [
            row
            for row in rows
            if row["source"] == source and row["stage"] == "timestamp_mapping"
        ]
        uncertainty = [
            value
            for row in mapping_rows
            if (value := _finite_float(row.get("timestamp_uncertainty_ms"))) is not None
        ]
        transport = [
            value
            for row in mapping_rows
            if (value := _finite_float(row.get("mapped_transport_delay_ms"))) is not None
        ]
        residual = [
            value
            for row in mapping_rows
            if (value := _finite_float(row.get("clock_residual_rms_ms"))) is not None
        ]
        reliable_count = sum(
            str(row.get("timestamp_reliable", "")).lower() == "true"
            for row in mapping_rows
        )
        timestamp_summary[source] = {
            "rows": len(mapping_rows),
            "reliable_rate": (
                reliable_count / len(mapping_rows) if mapping_rows else None
            ),
            "uncertainty_ms": _stats(uncertainty),
            "mapped_transport_delay_ms": _stats(transport),
            "clock_residual_rms_ms": _stats(residual),
        }

    metadata_path = performance_path.with_suffix(".meta.json")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {"closed_cleanly": False, "dropped_records": None}
    )
    legacy_timing = _read_csv(session_directory / "timing.csv")
    legacy_vision = _read_csv(session_directory / "vision.csv")
    legacy_vision_duration = [
        value
        for row in legacy_timing
        if row.get("stage") == "vision"
        and (value := _finite_float(row.get("duration_ms"))) is not None
    ]
    legacy_latency = [
        value
        for row in legacy_timing
        if row.get("stage") == "vision"
        and (value := _finite_float(row.get("latency_ms"))) is not None
    ]

    return {
        "session_directory": str(session_directory.resolve()),
        "generated_local": datetime.now().isoformat(timespec="seconds"),
        "performance_rows": len(rows),
        "profiler_metadata": metadata,
        "stage_summary": stage_summary,
        "queue": {
            "processed_frames": len(queue_rows),
            "frames_skipped": skipped,
            "duplicate_rejections": duplicate_rejections,
            "queue_age_ms": _stats(queue_age),
            "capture_age_ms": _stats(capture_age),
        },
        "timestamps": timestamp_summary,
        "legacy": {
            "timing_rows": len(legacy_timing),
            "vision_rows": len(legacy_vision),
            "vision_duration_ms": _stats(legacy_vision_duration),
            "capture_to_control_latency_ms": _stats(legacy_latency),
        },
    }


def write_outputs(result: dict[str, object], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "performance_stage_summary.csv"
    fields = [
        "source",
        "stage",
        "kind",
        "count",
        "mean",
        "p50",
        "p95",
        "p99",
        "max",
    ]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["stage_summary"])
    (output_directory / "performance_analysis_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    queue_summary = result["queue"]
    timestamp_summary = result["timestamps"]
    legacy = result["legacy"]
    queue_p50 = _format(queue_summary["queue_age_ms"]["p50"])
    queue_p95 = _format(queue_summary["queue_age_ms"]["p95"])
    capture_p50 = _format(queue_summary["capture_age_ms"]["p50"])
    capture_p95 = _format(queue_summary["capture_age_ms"]["p95"])
    lines = [
        "# 视觉伺服性能诊断自动汇总",
        "",
        f"- 会话：`{result['session_directory']}`",
        f"- 性能记录数：{result['performance_rows']}",
        f"- 诊断记录丢失数：{result['profiler_metadata'].get('dropped_records')}",
        "",
        "## 分阶段耗时（ms）",
        "",
        "| 来源 | 阶段 | 类型 | 样本 | 平均 | P50 | P95 | P99 | 最大 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result["stage_summary"]:
        row_template = (
            "| {source} | {stage} | {kind} | {count} | {mean} | {p50} | "
            "{p95} | {p99} | {max} |"
        )
        lines.append(
            row_template.format(
                source=item["source"],
                stage=item["stage"],
                kind=item["kind"],
                count=_format(item["count"]),
                mean=_format(item["mean"]),
                p50=_format(item["p50"]),
                p95=_format(item["p95"]),
                p99=_format(item["p99"]),
                max=_format(item["max"]),
            )
        )
    lines.extend(
        [
            "",
            "## 缓冲与时间戳",
            "",
            "- 处理帧数：{processed}；跳过旧帧：{skipped}；重复帧拒绝：{duplicates}。".format(
                processed=queue_summary["processed_frames"],
                skipped=queue_summary["frames_skipped"],
                duplicates=queue_summary["duplicate_rejections"],
            ),
            f"- 接收后排队年龄 P50/P95：{queue_p50}/{queue_p95} ms。",
            f"- 曝光到处理年龄 P50/P95：{capture_p50}/{capture_p95} ms。",
        ]
    )
    for source, item in timestamp_summary.items():
        reliable = item["reliable_rate"]
        reliable_text = "--" if reliable is None else f"{reliable * 100.0:.1f}%"
        uncertainty_p50 = _format(item["uncertainty_ms"]["p50"])
        uncertainty_p95 = _format(item["uncertainty_ms"]["p95"])
        lines.append(
            f"- {source} 时间戳可靠率 {reliable_text}；不确定度 P50/P95 "
            f"{uncertainty_p50}/{uncertainty_p95} ms。"
        )
    vision_duration = legacy["vision_duration_ms"]
    end_to_end_latency = legacy["capture_to_control_latency_ms"]
    lines.extend(
        [
            "",
            "## 旧版总耗时交叉核对",
            "",
            f"- vision 总耗时平均/P50/P95/最大：{_format(vision_duration['mean'])}/"
            f"{_format(vision_duration['p50'])}/{_format(vision_duration['p95'])}/"
            f"{_format(vision_duration['max'])} ms。",
            f"- 端到端延迟平均/P50/P95/最大：{_format(end_to_end_latency['mean'])}/"
            f"{_format(end_to_end_latency['p50'])}/"
            f"{_format(end_to_end_latency['p95'])}/"
            f"{_format(end_to_end_latency['max'])} ms。",
            "",
            "该文件是机械汇总，不自动判断安全门限，也不修改控制参数。",
        ]
    )
    (output_directory / "performance_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总视觉伺服性能诊断CSV")
    parser.add_argument("session_directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    arguments = parser.parse_args()
    output_directory = arguments.output_directory or arguments.session_directory
    result = analyze(arguments.session_directory)
    write_outputs(result, output_directory)
    print(output_directory / "performance_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
