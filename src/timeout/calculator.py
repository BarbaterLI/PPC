"""Timeout calculator

Implements dynamic timeout calculation based on historical data, task complexity, and system status.
"""

import threading
from dataclasses import dataclass, field
from typing import Any

from .history import TimeoutHistory


@dataclass
class TimeoutConfig:
    """Timeout calculator configuration"""

    base_timeout: float = 120.0
    min_timeout: float = 30.0
    max_timeout: float = 600.0
    text_factor: float = 0.01
    task_type_factors: dict[str, float] = field(
        default_factory=lambda: {
            "tts": 1.5,
            "voice_clone": 2.0,
            "transcription": 1.2,
        }
    )
    history_weight: float = 0.3
    p95_weight: float = 0.6
    avg_weight: float = 0.4
    max_history_timeout: float = 300.0
    safety_margin: float = 1.2
    timeout_mode: str = "auto"  # fixed | auto | adaptive


@dataclass
class TimeoutResult:
    """Timeout calculation result"""

    timeout: float
    base_timeout: float
    complexity_timeout: float
    history_timeout: float
    method: str
    warning: bool = False
    details: dict[str, Any] | None = None


class TimeoutCalculator:
    """Dynamic timeout calculator

    Supports calculation based on historical data, task complexity, and text length.
    """

    def __init__(self, config: TimeoutConfig | None = None):
        self.config = config or TimeoutConfig()
        self._history = TimeoutHistory()

    def calculate(
        self,
        task_type: str = "default",
        text_length: int = 0,
        audio_duration: float = 0.0,
    ) -> TimeoutResult:
        """Calculate timeout based on configured timeout_mode.

        Args:
            task_type: Task type
            text_length: Text length
            audio_duration: Audio duration in seconds

        Returns:
            TimeoutResult with calculated timeout
        """
        complexity_timeout = self._calculate_complexity_timeout(task_type, text_length, audio_duration)
        history_timeout = self._calculate_history_timeout()

        mode = self.config.timeout_mode
        if mode == "fixed":
            timeout = self.config.base_timeout
            method = "fixed"
        elif mode == "adaptive":
            if self._history.history_size >= 10 and history_timeout > 0:
                timeout = history_timeout
                method = "adaptive"
            else:
                # 历史不足时回退到 auto/hybrid 计算
                timeout = self._calculate_hybrid_timeout(complexity_timeout, history_timeout)
                method = "adaptive_fallback"
        else:  # auto
            timeout = self._calculate_hybrid_timeout(complexity_timeout, history_timeout)
            method = "hybrid"

        timeout = self._clamp(timeout)

        return TimeoutResult(
            timeout=timeout,
            base_timeout=self.config.base_timeout,
            complexity_timeout=complexity_timeout,
            history_timeout=history_timeout,
            method=method,
            warning=timeout >= self.config.max_timeout * 0.8,
            details={
                "task_type": task_type,
                "text_length": text_length,
                "audio_duration": audio_duration,
            },
        )

    def record_result(self, text: str, actual_time: float, success: bool = True) -> None:
        """Record request result

        Args:
            text: Request text
            actual_time: Actual time taken
            success: Whether the request was successful
        """
        self._history.record(text, actual_time, success)

    def get_history(self) -> TimeoutHistory:
        """Get timeout history"""
        return self._history

    def get_stats(self) -> dict[str, Any]:
        """Get timeout calculator statistics.

        Returns a snapshot of the current configuration and history statistics
        for observability / monitoring use.
        """
        history_stats = self._history.get_statistics()
        return {
            "base_timeout": self.config.base_timeout,
            "min_timeout": self.config.min_timeout,
            "max_timeout": self.config.max_timeout,
            "history_size": self._history.history_size,
            "count": history_stats.count,
            "average": history_stats.average,
            "p95": history_stats.p95,
        }

    def _calculate_complexity_timeout(
        self,
        task_type: str,
        text_length: int,
        audio_duration: float,
    ) -> float:
        """Calculate timeout based on task complexity"""
        base = self.config.base_timeout
        factor = self.config.task_type_factors.get(task_type, 1.0)
        text_timeout = text_length * self.config.text_factor
        audio_timeout = audio_duration * 0.5 if audio_duration > 0 else 0
        return (base + text_timeout + audio_timeout) * factor

    def _calculate_history_timeout(self) -> float:
        """Calculate timeout based on historical data"""
        if self._history.history_size < 10:
            return 0.0

        stats = self._history.get_statistics()

        if stats.count == 0:
            return self.config.base_timeout

        p95_timeout = stats.p95 * self.config.p95_weight
        avg_timeout = stats.average * self.config.avg_weight
        history_timeout = (p95_timeout + avg_timeout) * self.config.safety_margin

        return min(history_timeout, self.config.max_history_timeout)

    def _calculate_hybrid_timeout(self, complexity_timeout: float, history_timeout: float) -> float:
        """混合模式：历史数据不足时使用复杂度，否则按权重混合。"""
        if self._history.history_size < 10:
            return complexity_timeout

        base_weight = 1 - self.config.history_weight
        weights = (
            self.config.base_timeout * base_weight * 0.7,
            complexity_timeout * base_weight * 0.3,
            history_timeout * self.config.history_weight,
        )
        return sum(weights)

    def _clamp(self, timeout: float) -> float:
        """Clamp timeout to valid range"""
        return max(self.config.min_timeout, min(timeout, self.config.max_timeout))

    def reset(self) -> None:
        """Reset calculator state"""
        self._history.clear()


_calculator_instance: TimeoutCalculator | None = None
_calculator_lock = threading.Lock()


def get_calculator(config: TimeoutConfig | None = None) -> TimeoutCalculator:
    """Get timeout calculator instance with thread-safe singleton"""
    global _calculator_instance
    if _calculator_instance is None:
        with _calculator_lock:
            if _calculator_instance is None:
                _calculator_instance = TimeoutCalculator(config)
    return _calculator_instance


def calculate_timeout(
    task_type: str = "default",
    text_length: int = 0,
    audio_duration: float = 0.0,
) -> float:
    """Calculate timeout convenience function"""
    result = get_calculator().calculate(task_type, text_length, audio_duration)
    return result.timeout
