"""Timeout history and analysis

Records the last 100 requests' actual timing, implements P95/P90 percentile calculation and dynamic timeout adjustment.
"""

import bisect
import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class TimeoutStatistics:
    """Timeout statistics"""

    p95: float = 0.0
    p90: float = 0.0
    average: float = 0.0
    maximum: float = 0.0
    minimum: float = float("inf")
    count: int = 0
    warning_count: int = 0

    def to_dict(self) -> dict:
        return {
            "p95": self.p95,
            "p90": self.p90,
            "average": self.average,
            "maximum": self.maximum,
            "minimum": self.minimum if self.minimum != float("inf") else 0.0,
            "count": self.count,
            "warning_count": self.warning_count,
        }


@dataclass
class RequestRecord:
    """Request record"""

    text_length: int
    actual_time: float
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    timeout_occurred: bool = False


class TimeoutHistory:
    """Timeout history tracker

    Features:
    - Records last 100 requests' actual timing
    - P95/P90 percentile calculation using bisect for O(log n) insertion
    - Dynamic timeout adjustment based on text length: +10s per 1000 chars
    - Timeout prediction: triggers warning when request exceeds P90
    - Thread-safe with lock protection
    """

    DEFAULT_HISTORY_SIZE = 100
    BASE_TIMEOUT_PER_1000_CHARS = 10.0

    def __init__(self, max_size: int | None = None):
        self._max_size = max_size if max_size is not None else self.DEFAULT_HISTORY_SIZE
        self._history: deque[RequestRecord] = deque(maxlen=self._max_size)
        self._sorted_times: list[float] = []
        self._lock = threading.Lock()
        self._warning_count: int = 0

    def record(
        self,
        text: str,
        actual_time: float,
        success: bool = True,
        timeout_occurred: bool = False,
    ) -> None:
        """Record request result

        Args:
            text: Request text content
            actual_time: Actual time taken (seconds)
            success: Whether the request was successful
            timeout_occurred: Whether a timeout occurred
        """
        record = RequestRecord(
            text_length=len(text),
            actual_time=actual_time,
            timestamp=time.time(),
            success=success,
            timeout_occurred=timeout_occurred,
        )

        with self._lock:
            # deque(maxlen=N) 在 append 时会自动驱逐最旧记录，
            # 但 _sorted_times 不会同步清理 → 内存泄漏且统计基于全量历史而非最近 N 条。
            # 在 append 前先取出将被驱逐的记录，并从 _sorted_times 中移除其 actual_time。
            if len(self._history) >= self._max_size and self._history:
                evicted = self._history[0]
                if evicted.success:
                    idx = bisect.bisect_left(self._sorted_times, evicted.actual_time)
                    if idx < len(self._sorted_times) and self._sorted_times[idx] == evicted.actual_time:
                        self._sorted_times.pop(idx)

            self._history.append(record)
            if success:
                bisect.insort(self._sorted_times, actual_time)

            p90 = self._percentile_unlocked(90)

            if actual_time > p90 and success:
                self._warning_count += 1

    def get_p95(self) -> float:
        """Get P95 percentile"""
        return self._percentile(95)

    def get_p90(self) -> float:
        """Get P90 percentile"""
        return self._percentile(90)

    def get_average(self) -> float:
        """Get average time"""
        with self._lock:
            if not self._sorted_times:
                return 0.0
            return sum(self._sorted_times) / len(self._sorted_times)

    def get_maximum(self) -> float:
        """Get maximum time"""
        with self._lock:
            if not self._sorted_times:
                return 0.0
            return self._sorted_times[-1]

    def get_minimum(self) -> float:
        """Get minimum time"""
        with self._lock:
            if not self._sorted_times:
                return float("inf")
            return self._sorted_times[0]

    def calculate_dynamic_timeout(self, text: str, per_1000_chars: float | None = None) -> float:
        """Calculate dynamic timeout based on text length

        Formula: base_timeout + (text_length / 1000) * per_1000_chars

        Args:
            text: Text to process
            per_1000_chars: 每 1000 字符加成秒数；None 时使用类常量
                ``BASE_TIMEOUT_PER_1000_CHARS``（10.0）。

        Returns:
            Dynamic timeout (seconds)
        """
        text_length = len(text)
        base_timeout = max(self.get_p95(), self.get_average())

        if base_timeout == 0.0:
            base_timeout = 60.0

        per_1000 = self.BASE_TIMEOUT_PER_1000_CHARS if per_1000_chars is None else per_1000_chars
        length_based_timeout = (text_length / 1000.0) * per_1000

        return base_timeout + length_based_timeout

    def predict_timeout(self, text: str, per_1000_chars: float | None = None) -> tuple:
        """Predict timeout and check if warning is needed

        Args:
            text: Text to process
            per_1000_chars: 每 1000 字符加成秒数，透传给
                :meth:`calculate_dynamic_timeout`。

        Returns:
            (predicted_timeout, should_warn): Predicted timeout and whether to warn
        """
        predicted = self.calculate_dynamic_timeout(text, per_1000_chars)
        p90 = self.get_p90()

        should_warn = p90 > 0 and predicted > p90

        return predicted, should_warn

    def recent_records(self, limit: int = 10) -> list[RequestRecord]:
        """返回最近 ``limit`` 条记录（最新在前），线程安全快照。"""
        with self._lock:
            return list(self._history)[-limit:][::-1]

    def get_statistics(self) -> TimeoutStatistics:
        """Get complete statistics

        Returns:
            TimeoutStatistics object
        """
        return TimeoutStatistics(
            p95=self.get_p95(),
            p90=self.get_p90(),
            average=self.get_average(),
            maximum=self.get_maximum(),
            minimum=self.get_minimum(),
            count=len(self._history),
            warning_count=self._warning_count,
        )

    def _percentile_unlocked(self, p: int) -> float:
        """Calculate percentile without acquiring lock (must be called with lock held)

        Args:
            p: Percentile (0-100)

        Returns:
            Value at the p-th percentile
        """
        if not self._sorted_times:
            return 0.0

        times = self._sorted_times
        k = (len(times) - 1) * p / 100.0
        f = int(k)
        c = f + 1

        if c >= len(times):
            return times[-1]

        return times[f] + (times[c] - times[f]) * (k - f)

    def _percentile(self, p: int) -> float:
        """Calculate percentile using sorted list for O(log n) access

        Args:
            p: Percentile (0-100)

        Returns:
            Value at the p-th percentile
        """
        with self._lock:
            return self._percentile_unlocked(p)

    def clear(self) -> None:
        """Clear history"""
        with self._lock:
            self._history.clear()
            self._sorted_times.clear()
        self._warning_count = 0

    @property
    def history_size(self) -> int:
        """Get current history size"""
        return len(self._history)
