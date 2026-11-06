"""只保留最新值的线程安全缓冲区。"""

from __future__ import annotations

import math
import threading
import time
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestValueBuffer(Generic[T]):
    """容量恒为一的跨线程数据槽。

    生产者永不因慢消费者而堆积历史数据；每次 ``publish`` 都覆盖旧值并递增序号。
    消费者保存上次见过的序号，再用 ``wait_for_newer`` 等待真正的新值。
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._sequence = 0
        self._value: T | None = None
        self._has_value = False

    def publish(self, value: T) -> int:
        """原子地替换当前值，返回从 1 开始递增的序号。"""

        with self._condition:
            self._sequence += 1
            self._value = value
            self._has_value = True
            self._condition.notify_all()
            return self._sequence

    def clear(self) -> int:
        """清除当前快照并进入新代次。

        序号仍然递增，因此消费者不会把清除前的值误当成新代次数据。
        """

        with self._condition:
            self._sequence += 1
            self._value = None
            self._has_value = False
            self._condition.notify_all()
            return self._sequence

    def get_latest(self) -> tuple[int, T] | None:
        """立即取得 ``(序号, 最新值)``；尚未发布时返回 ``None``。"""

        with self._condition:
            if not self._has_value:
                return None
            return self._sequence, self._value  # type: ignore[return-value]

    def wait_for_newer(self, sequence: int, timeout_s: float | None) -> tuple[int, T] | None:
        """等待序号大于 ``sequence`` 的值，超时返回 ``None``。

        ``timeout_s=None`` 表示无限等待；零表示只检查一次、不阻塞。使用绝对截止时间可
        正确处理条件变量的伪唤醒，也不会在多次唤醒后把总超时时间意外延长。
        """

        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s < 0.0):
            raise ValueError("timeout_s must be finite and non-negative, or None")

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._condition:
            while not self._has_value or self._sequence <= sequence:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            return self._sequence, self._value  # type: ignore[return-value]

