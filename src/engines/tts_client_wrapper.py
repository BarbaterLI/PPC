"""TTS client wrapper: dynamic timeout, error classification, stats.

Split out of the former ``tts_engine`` module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.core.exceptions import (
    ErrorCodes,
    NetworkError,
    PermanentError,
    QuotaError,
    TransientError,
)
from src.engines.client_base import EdgeTTSClient
from src.engines.tts_engine_config import TTSEngineConfig
from src.engines.tts_metrics import EngineStats
from src.reliability import ExecutionResult
from src.timeout import TimeoutCalculator, TimeoutHistory

logger = logging.getLogger(__name__)


class TTSClientWrapper:
    """EdgeTTSClient 包裹层：动态超时计算、错误分类、统计收集。"""

    def __init__(
        self,
        edge_client: EdgeTTSClient,
        *,
        timeout_calculator: TimeoutCalculator,
        timeout_history: TimeoutHistory,
        stats: EngineStats,
    ) -> None:
        self._edge_client = edge_client
        self._timeout_calculator = timeout_calculator
        self._timeout_history = timeout_history
        self._stats = stats

    @property
    def edge_client(self) -> EdgeTTSClient:
        return self._edge_client

    def calculate_timeout(self, config: TTSEngineConfig, text: str) -> float:
        """根据当前 timeout_mode 计算超时时间。

        统一委托 :class:`TimeoutCalculator`（唯一超时公式实现），
        支持 fixed / auto / adaptive / synthetic 四种模式，结果乘以
        ``timeout_multiplier``。calculator 与 wrapper 共享同一个
        :class:`TimeoutHistory` 实例，历史统计分支真实生效。
        """
        result = self._timeout_calculator.calculate(
            task_type="tts",
            text_length=len(text),
            audio_duration=0.0,
            text=text,
        )
        timeout = result.timeout * config.timeout_multiplier

        logger.debug(f"超时计算：{timeout:.1f}s (模式={result.method}, 倍率={config.timeout_multiplier}x)")
        return timeout

    def record_timeout(
        self,
        text: str,
        actual_time: float,
        success: bool,
        timeout_occurred: bool = False,
    ) -> None:
        """记录超时历史"""
        self._timeout_history.record(
            text=text,
            actual_time=actual_time,
            success=success,
            timeout_occurred=timeout_occurred,
        )

    @staticmethod
    def classify_error_result(exc: BaseException) -> ExecutionResult[Path]:
        """根据异常类型返回对应的 ExecutionResult。"""
        specific_error_code = getattr(exc, "error_code", None)
        if specific_error_code == "TTS_NO_AUDIO_RECEIVED":
            return ExecutionResult.fail(error=str(exc), error_code=ErrorCodes.TTS_NO_AUDIO_RECEIVED.value)
        if isinstance(exc, TransientError):
            return ExecutionResult.fail(error=str(exc), error_code=ErrorCodes.TTS_TRANSIENT_FAILED.value)
        if isinstance(exc, PermanentError):
            return ExecutionResult.fail(error=str(exc), error_code=ErrorCodes.TTS_PERMANENT_FAILED.value)
        if isinstance(exc, QuotaError):
            return ExecutionResult.fail(error=str(exc), error_code=ErrorCodes.TTS_QUOTA_EXCEEDED.value)
        if isinstance(exc, NetworkError):
            return ExecutionResult.fail(error=str(exc), error_code=ErrorCodes.TTS_NETWORK_FAILED.value)
        return ExecutionResult.fail(error=str(exc), error_code=ErrorCodes.TTS_SYNTHESIS_FAILED.value)

    def get_timeout_stats(self) -> dict[str, Any]:
        try:
            return self._timeout_calculator.get_stats()
        except Exception:  # noqa: BLE001
            return {}

    def get_history_stats(self) -> dict[str, Any]:
        history_stats = self._timeout_history.get_statistics()
        return {
            "p95": history_stats.p95,
            "p90": history_stats.p90,
            "average": history_stats.average,
            "maximum": history_stats.maximum,
            "minimum": history_stats.minimum if history_stats.minimum != float("inf") else 0.0,
            "count": history_stats.count,
            "warning_count": history_stats.warning_count,
        }

    async def cleanup(self) -> None:
        try:
            await self._edge_client.cleanup()
        except Exception as e:
            logger.warning("清理 edge client 失败: %s", e)
