"""Timeout calculator

Implements dynamic timeout calculation based on historical data, task complexity, and system status.

超时模式（``timeout_mode``）：

- ``fixed``：恒定使用 ``base_timeout``，不做任何推算。
- ``adaptive``：历史样本充足时以 P95/avg 统计推算；不足时回退 hybrid。
- ``auto``（默认）：hybrid 混合（复杂度 + 历史加权）。
- ``synthetic``：fixed 基线 + adaptive 历史的合成模式，由轻量启发式
  :class:`TimeoutHeuristicController` 调控（冷启动/超时升级/过度宽松收缩/预警）。

设计约束（单一实现）：TTS 合成链路（``TTSClientWrapper``）不再自带超时公式，
统一委托本模块。历史样本由 :class:`TimeoutHistory` 单一实例承载，calculator
与 wrapper 共享同一实例，保证 adaptive/synthetic 的历史分支真实生效。
"""

import threading
from dataclasses import dataclass, field
from typing import Any

from .heuristics import TimeoutHeuristicController
from .history import TimeoutHistory


@dataclass
class TimeoutConfig:
    """Timeout calculator configuration

    默认值与 ``TTSConfig``（src/config/schemas/tts.py）及
    ``TTSEngineConfig``（src/engines/tts_engine_config.py）保持一致：
    base=120 / min=50 / max=720。
    """

    base_timeout: float = 120.0
    min_timeout: float = 50.0
    max_timeout: float = 720.0
    text_factor: float = 0.01
    audio_factor: float = 0.5
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
    # 历史样本达到该数量后才启用历史统计分支（原硬编码 10）
    history_min_samples: int = 10
    # synthetic 模式参数（启发式调控）
    length_bonus_per_1000: float = 10.0
    heuristic_grow_factor: float = 1.25
    heuristic_shrink_ratio: float = 0.8
    timeout_mode: str = "auto"  # fixed | auto | adaptive | synthetic


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

    def __init__(self, config: TimeoutConfig | None = None, history: TimeoutHistory | None = None):
        self.config = config or TimeoutConfig()
        self._history = history if history is not None else TimeoutHistory()
        self._heuristics = TimeoutHeuristicController(self.config)

    def calculate(
        self,
        task_type: str = "default",
        text_length: int = 0,
        audio_duration: float = 0.0,
        text: str | None = None,
    ) -> TimeoutResult:
        """Calculate timeout based on configured timeout_mode.

        Args:
            task_type: Task type
            text_length: Text length
            audio_duration: Audio duration in seconds
            text: 原始文本（synthetic 模式用于启发式长度预测）

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
            if self._history.history_size >= self.config.history_min_samples and history_timeout > 0:
                timeout = history_timeout
                method = "adaptive"
            else:
                # 历史不足时回退到 auto/hybrid 计算
                timeout = self._calculate_hybrid_timeout(complexity_timeout, history_timeout)
                method = "adaptive_fallback"
        elif mode == "synthetic":
            # fixed 基线 + adaptive 历史 + 启发式调控
            suggestion = self._heuristics.suggest(
                history=self._history,
                text_length=text_length,
                text=text,
                recent_timeout_count=self._recent_timeout_count(),
            )
            timeout = suggestion.timeout
            # reason → method 命名：synthetic / synthetic_cold / synthetic_escalation / synthetic_contraction
            method = {
                "history_synthetic": "synthetic",
                "cold_start": "synthetic_cold",
                "timeout_escalation": "synthetic_escalation",
                "contraction": "synthetic_contraction",
            }.get(suggestion.reason, suggestion.reason)
            warning = suggestion.warn
        else:  # auto
            timeout = self._calculate_hybrid_timeout(complexity_timeout, history_timeout)
            method = "hybrid"

        timeout = self._clamp(timeout)
        warning = locals().get("warning", False) or timeout >= self.config.max_timeout * 0.8

        return TimeoutResult(
            timeout=timeout,
            base_timeout=self.config.base_timeout,
            complexity_timeout=complexity_timeout,
            history_timeout=history_timeout,
            method=method,
            warning=warning,
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
            "timeout_mode": self.config.timeout_mode,
            "history_size": self._history.history_size,
            "count": history_stats.count,
            "average": history_stats.average,
            "p95": history_stats.p95,
            "p90": history_stats.p90,
            "warning_count": history_stats.warning_count,
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
        audio_timeout = audio_duration * self.config.audio_factor if audio_duration > 0 else 0
        return (base + text_timeout + audio_timeout) * factor

    def _calculate_history_timeout(self) -> float:
        """Calculate timeout based on historical data"""
        if self._history.history_size < self.config.history_min_samples:
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
        if self._history.history_size < self.config.history_min_samples:
            return complexity_timeout

        base_weight = 1 - self.config.history_weight
        weights = (
            self.config.base_timeout * base_weight * 0.7,
            complexity_timeout * base_weight * 0.3,
            history_timeout * self.config.history_weight,
        )
        return sum(weights)

    def _recent_timeout_count(self) -> int:
        """统计最近一批（history 窗口内）发生超时的请求数，供启发式升级使用。"""
        count = 0
        for record in self._history.recent_records(limit=10):
            if record.timeout_occurred:
                count += 1
        return count

    def _clamp(self, timeout: float) -> float:
        """Clamp timeout to valid range"""
        return max(self.config.min_timeout, min(timeout, self.config.max_timeout))

    def reset(self) -> None:
        """Reset calculator state"""
        self._history.clear()


_calculator_instance: TimeoutCalculator | None = None
_calculator_lock = threading.Lock()


def get_calculator(config: TimeoutConfig | None = None) -> TimeoutCalculator:
    """Get timeout calculator instance with thread-safe singleton.

    - ``config is None``：返回现有实例（不存在则用默认配置创建）。
    - 传入 config 且与现有实例配置不同：**重建实例并继承旧实例的历史**，
      避免配置热更新被单例固化吞掉（修复"首次调用后配置被锁定"问题）。
    """
    global _calculator_instance
    if config is None:
        if _calculator_instance is None:
            with _calculator_lock:
                if _calculator_instance is None:
                    _calculator_instance = TimeoutCalculator()
        return _calculator_instance

    with _calculator_lock:
        if _calculator_instance is None:
            _calculator_instance = TimeoutCalculator(config)
        elif _calculator_instance.config != config:
            old_history = _calculator_instance.get_history()
            _calculator_instance = TimeoutCalculator(config, history=old_history)
    return _calculator_instance


def calculate_timeout(
    task_type: str = "default",
    text_length: int = 0,
    audio_duration: float = 0.0,
) -> float:
    """Calculate timeout convenience function"""
    result = get_calculator().calculate(task_type, text_length, audio_duration)
    return result.timeout
