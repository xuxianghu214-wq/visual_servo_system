"""有界、低扰动的性能诊断记录器。

``OFF`` 不创建写盘线程；``LIGHT`` 在消费线程内聚合汇总；
``FULL_ASYNC`` 批量写出详细阶段记录。生产线程只做一次非阻塞入队，
永不在相机、视觉、RTDE 或控制关键路径内写文件。

该通道只用于可丢弃的普通性能诊断。安全故障和状态转移仍由
``SessionCsvLogger`` 的独立安全事件通道记录。
"""

from __future__ import annotations

import csv
import json
import queue
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np


class PerformanceMode(StrEnum):
    """性能诊断运行模式。"""

    OFF = "OFF"
    LIGHT = "LIGHT"
    FULL_ASYNC = "FULL_ASYNC"


PERFORMANCE_FIELDS = [
    "recorded_monotonic_ns",
    "thread_name",
    "source",
    "stage",
    "kind",
    "sequence_number",
    "duration_ns",
    "duration_ms",
    "frame_host_timestamp_s",
    "receive_timestamp_s",
    "queue_age_ms",
    "capture_age_ms",
    "frames_skipped",
    "timestamp_source",
    "timestamp_reliable",
    "timestamp_uncertainty_ms",
    "mapped_transport_delay_ms",
    "clock_residual_rms_ms",
    "clock_sample_count",
    "clock_reset_count",
    "detail",
]

_SUPPORTED_KINDS = frozenset(
    {"compute", "wait", "lock", "copy", "io", "queue", "total"}
)


class PerformanceProfiler:
    """把一个工作周期的多个阶段作为一个有界队列项提交。"""

    def __init__(
        self,
        output_path: str | Path,
        *,
        mode: PerformanceMode | str = PerformanceMode.FULL_ASYNC,
        queue_capacity: int = 4096,
        flush_every_records: int = 256,
        flush_interval_s: float = 1.0,
        light_summary_interval_s: float = 1.0,
        light_reservoir_size: int = 2048,
        close_timeout_s: float = 5.0,
        perf_counter_ns: Callable[[], int] = time.perf_counter_ns,
        create_parent: bool = True,
        writer_start_gate: Callable[[], None] | None = None,
    ) -> None:
        try:
            self.mode = PerformanceMode(mode)
        except ValueError as error:
            raise ValueError("mode must be OFF, LIGHT, or FULL_ASYNC") from error
        if type(queue_capacity) is not int or queue_capacity <= 0:
            raise ValueError("queue_capacity must be a positive integer")
        if type(flush_every_records) is not int or flush_every_records <= 0:
            raise ValueError("flush_every_records must be a positive integer")
        if not np.isfinite(flush_interval_s) or flush_interval_s <= 0.0:
            raise ValueError("flush_interval_s must be finite and positive")
        if not np.isfinite(light_summary_interval_s) or light_summary_interval_s <= 0.0:
            raise ValueError("light_summary_interval_s must be finite and positive")
        if type(light_reservoir_size) is not int or light_reservoir_size <= 0:
            raise ValueError("light_reservoir_size must be a positive integer")
        if not np.isfinite(close_timeout_s) or close_timeout_s <= 0.0:
            raise ValueError("close_timeout_s must be finite and positive")

        self.output_path = Path(output_path)
        self._queue_capacity = queue_capacity
        self._flush_every_records = flush_every_records
        self._flush_interval_s = float(flush_interval_s)
        self._light_summary_interval_s = float(light_summary_interval_s)
        self._light_reservoir_size = light_reservoir_size
        self._close_timeout_s = float(close_timeout_s)
        self._clock_ns = perf_counter_ns
        self._writer_start_gate = writer_start_gate
        self._queue: queue.Queue[tuple[dict[str, Any], ...]] | None = None
        self._closing = threading.Event()
        self._closed = False
        self._dropped_records = 0
        self._submitted_batches = 0
        self._written_records = 0
        self._writer_error: Exception | None = None
        self._thread: threading.Thread | None = None
        self._local = threading.local()

        if self.mode is not PerformanceMode.OFF:
            if create_parent:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._queue = queue.Queue(queue_capacity)
            self._thread = threading.Thread(
                target=self._writer_loop,
                name="performance-diagnostics-writer",
                daemon=True,
            )
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self.mode is not PerformanceMode.OFF and not self._closed

    @property
    def detailed(self) -> bool:
        return self.mode is PerformanceMode.FULL_ASYNC and not self._closed

    @property
    def dropped_records(self) -> int:
        return self._dropped_records

    @property
    def submitted_batches(self) -> int:
        return self._submitted_batches

    @property
    def writer_error(self) -> Exception | None:
        return self._writer_error

    def _validate_details(self, details: Mapping[str, Any]) -> None:
        unknown = set(details) - set(PERFORMANCE_FIELDS)
        if unknown:
            raise KeyError(f"unknown performance fields: {sorted(unknown)}")

    def record(
        self,
        *,
        source: str,
        stage: str,
        kind: str,
        duration_ns: int,
        sequence_number: int | None = None,
        **details: Any,
    ) -> bool:
        """兼容旧接口；新的高频路径应使用 :meth:`record_stages`。"""

        pending = getattr(self._local, "pending", None)
        if pending is not None and self.enabled:
            pending.append(
                (source, stage, kind, duration_ns, sequence_number, dict(details))
            )
            return True
        return self.record_stages(
            source=source,
            sequence_number=sequence_number,
            stages=((stage, kind, duration_ns),),
            **details,
        )

    @contextmanager
    def cycle(self):
        """把当前线程在一个工作周期内的多次 ``record`` 合并入队。"""

        if not self.enabled:
            yield
            return
        existing = getattr(self._local, "pending", None)
        if existing is not None:
            yield
            return
        pending: list[tuple[str, str, str, int, int | None, dict[str, Any]]] = []
        self._local.pending = pending
        try:
            yield
        finally:
            self._local.pending = None
            if pending:
                self._submit_entries(pending)

    def record_stages(
        self,
        *,
        source: str,
        stages: Sequence[tuple[str, str, int]] | Iterable[tuple[str, str, int]],
        sequence_number: int | None = None,
        **details: Any,
    ) -> bool:
        """非阻塞提交一个帧/样本的全部阶段，队列满时丢弃整批。"""

        if not self.enabled:
            return False
        stage_values = tuple(stages)
        if not stage_values:
            return False
        entries = [
            (source, stage, kind, duration_ns, sequence_number, dict(details))
            for stage, kind, duration_ns in stage_values
        ]
        return self._submit_entries(entries)

    def _submit_entries(
        self,
        entries: Sequence[tuple[str, str, str, int, int | None, dict[str, Any]]],
    ) -> bool:
        """验证一个周期后只执行一次 ``put_nowait``。"""

        if not self.enabled:
            return False
        recorded_ns = int(self._clock_ns())
        thread_name = threading.current_thread().name
        rows: list[dict[str, Any]] = []
        for source, stage, kind, duration_ns, sequence_number, details in entries:
            if not isinstance(source, str) or not source:
                raise ValueError("source must be a non-empty string")
            if sequence_number is not None and type(sequence_number) is not int:
                raise ValueError("sequence_number must be an integer or None")
            self._validate_details(details)
            if not isinstance(stage, str) or not stage:
                raise ValueError("stage must be a non-empty string")
            if kind not in _SUPPORTED_KINDS:
                raise ValueError("unsupported performance record kind")
            if type(duration_ns) is not int or duration_ns < 0:
                raise ValueError("duration_ns must be a non-negative integer")
            row = {field: "" for field in PERFORMANCE_FIELDS}
            row.update(details)
            row.update(
                {
                    "recorded_monotonic_ns": recorded_ns,
                    "thread_name": thread_name,
                    "source": source,
                    "stage": stage,
                    "kind": kind,
                    "sequence_number": "" if sequence_number is None else sequence_number,
                    "duration_ns": duration_ns,
                    "duration_ms": duration_ns / 1_000_000.0,
                }
            )
            rows.append(row)
        batch = tuple(rows)
        diagnostic_queue = self._queue
        if diagnostic_queue is None:
            return False
        try:
            diagnostic_queue.put_nowait(batch)
        except queue.Full:
            self._dropped_records += len(batch)
            return False
        self._submitted_batches += 1
        return True

    def _writer_loop(self) -> None:
        try:
            if self._writer_start_gate is not None:
                self._writer_start_gate()
            if self.mode is PerformanceMode.LIGHT:
                self._light_writer_loop()
            else:
                self._full_writer_loop()
        except Exception as error:  # 诊断故障只被记录，不传播到实时线程。
            self._writer_error = error

    def _next_batch(self) -> tuple[dict[str, Any], ...] | None:
        diagnostic_queue = self._queue
        if diagnostic_queue is None:
            return None
        try:
            return diagnostic_queue.get(timeout=min(self._flush_interval_s, 0.1))
        except queue.Empty:
            return None

    def _should_finish(self) -> bool:
        diagnostic_queue = self._queue
        return bool(
            self._closing.is_set()
            and (diagnostic_queue is None or diagnostic_queue.empty())
        )

    def _full_writer_loop(self) -> None:
        with self.output_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=PERFORMANCE_FIELDS)
            writer.writeheader()
            pending = 0
            last_flush_s = time.monotonic()
            while True:
                batch = self._next_batch()
                if batch is not None:
                    writer.writerows(batch)
                    pending += len(batch)
                    self._written_records += len(batch)
                now = time.monotonic()
                if pending and (
                    pending >= self._flush_every_records
                    or now - last_flush_s >= self._flush_interval_s
                ):
                    stream.flush()
                    pending = 0
                    last_flush_s = now
                if self._should_finish():
                    stream.flush()
                    return

    def _light_writer_loop(self) -> None:
        values: dict[tuple[str, str, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=self._light_reservoir_size)
        )
        counts: dict[tuple[str, str, str], int] = defaultdict(int)
        sums: dict[tuple[str, str, str], float] = defaultdict(float)
        maxima: dict[tuple[str, str, str], float] = defaultdict(float)
        last_summary_s = time.monotonic()
        while True:
            batch = self._next_batch()
            if batch is not None:
                for row in batch:
                    key = (str(row["source"]), str(row["stage"]), str(row["kind"]))
                    duration_ms = float(row["duration_ms"])
                    values[key].append(duration_ms)
                    counts[key] += 1
                    sums[key] += duration_ms
                    maxima[key] = max(maxima[key], duration_ms)
                    self._written_records += 1
            now = time.monotonic()
            if counts and now - last_summary_s >= self._light_summary_interval_s:
                self._write_light_summary(values, counts, sums, maxima)
                last_summary_s = now
            if self._should_finish():
                self._write_light_summary(values, counts, sums, maxima)
                return

    def _write_light_summary(
        self,
        values: Mapping[tuple[str, str, str], deque[float]],
        counts: Mapping[tuple[str, str, str], int],
        sums: Mapping[tuple[str, str, str], float],
        maxima: Mapping[tuple[str, str, str], float],
    ) -> None:
        path = self.output_path.with_name("performance_summary.csv")
        fields = [
            "source",
            "stage",
            "kind",
            "count",
            "mean_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "max_ms",
            "reservoir_samples",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for key in sorted(counts):
                sample = np.asarray(values[key], dtype=np.float64)
                writer.writerow(
                    {
                        "source": key[0],
                        "stage": key[1],
                        "kind": key[2],
                        "count": counts[key],
                        "mean_ms": sums[key] / counts[key],
                        "p50_ms": float(np.percentile(sample, 50)),
                        "p95_ms": float(np.percentile(sample, 95)),
                        "p99_ms": float(np.percentile(sample, 99)),
                        "max_ms": maxima[key],
                        "reservoir_samples": len(sample),
                    }
                )

    def _write_metadata(self) -> None:
        thread = self._thread
        metadata = {
            "mode": self.mode.value,
            "closed_cleanly": thread is None or not thread.is_alive(),
            "submitted_batches": self._submitted_batches,
            "written_records": self._written_records,
            "dropped_records": self._dropped_records,
            "queue_capacity_batches": self._queue_capacity,
            "writer_error": None if self._writer_error is None else repr(self._writer_error),
        }
        try:
            self.output_path.with_suffix(".meta.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as error:
            if self._writer_error is None:
                self._writer_error = error

    def close(self, timeout_s: float | None = None) -> None:
        """在有界时间内排空已接收批次；诊断故障不向实时线程抛出。"""

        timeout = self._close_timeout_s if timeout_s is None else float(timeout_s)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        if self._closed:
            return
        self._closed = True
        if self.mode is PerformanceMode.OFF:
            return
        self._closing.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive() and self._writer_error is None:
                self._writer_error = TimeoutError(
                    "performance diagnostics writer did not stop within the bounded timeout"
                )
        self._write_metadata()
