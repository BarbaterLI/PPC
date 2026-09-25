"""轻量启发式超时调控系统（synthetic 模式核心）。

synthetic 模式 = fixed 基线 + adaptive 历史统计，再经一组轻量启发式规则
调节：

1. **冷启动（历史不足）**：使用 fixed 基线 + 文本长度加成（每 1000 字符
   ``length_bonus_per_1000`` 秒），保证合成任务从一开始就有合理的下限。
2. **历史充足**：以 fixed 基线为底、P95/P90 历史统计为参考加权合成，并
   叠加文本长度启发（复用 :meth:`TimeoutHistory.calculate_dynamic_timeout`
   的长度加成逻辑）。
3. **超时事件升级**：近期发生过超时（``timeout_occurred``）时按
   ``heuristic_grow_factor`` 放大，避免重复踩超时。
4. **过度宽松收缩**：当历史 P90 显著低于当前超时（小于
   ``heuristic_shrink_ratio`` 倍）时收缩到 ``P90 × safety_margin + 长度加成``，
   避免任务长时间空等。
5. **预警**：预测超时（``TimeoutHistory.predict_timeout``）超过历史 P90 时
   给出 warning，供上层观测/日志使用。

所有可调参数均来自 :class:`TimeoutConfig`，无魔法数字。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .history import TimeoutHistory

if TYPE_CHECKING:
    from .calculator import TimeoutConfig


@dataclass
class HeuristicSuggestion:
    """启发式调节建议。"""

    timeout: float
    reason: str  # cold_start | history_synthetic | timeout_escalation | contraction
    warn: bool = False
    details: dict | None = None


class TimeoutHeuristicController:
    """轻量启发式超时控制器。

    无状态设计：所有输入显式传入（history、文本长度、最近超时计数、当前值），
    便于单元测试与并发复用。
    """

    def __init__(self, config: TimeoutConfig):
        self.config = config

    def suggest(
        self,
        *,
        history: TimeoutHistory,
        text_length: int = 0,
        text: str | None = None,
        recent_timeout_count: int = 0,
    ) -> HeuristicSuggestion:
        """根据历史统计与文本长度给出超时建议。

        Args:
            history: 共享的合成历史（与 calculator 同一实例）。
            text_length: 本次合成的文本长度（字符数）。
            text: 原始文本（存在时用于激活 ``TimeoutHistory.predict_timeout``
                的长度启发；为空时退化为纯数值计算）。
            recent_timeout_count: 最近一批请求中超时发生的次数（>0 触发升级）。

        Returns:
            HeuristicSuggestion: 建议的超时值与原因。
        """
        config = self.config
        stats = history.get_statistics()

        # 1. 冷启动：历史不足 → fixed 基线 + 长度加成
        if history.history_size < config.history_min_samples:
            length_bonus = (text_length / 1000.0) * config.length_bonus_per_1000
            timeout = config.base_timeout + length_bonus
            return HeuristicSuggestion(
                timeout=timeout,
                reason="cold_start",
                warn=False,
                details={"text_length": text_length, "history_size": history.history_size},
            )

        # 2. 合成：fixed 基线权重 + adaptive 历史权重
        base_weight = 1.0 - config.history_weight
        history_timeout = self._history_synthetic_timeout(stats.p95, stats.average)
        length_bonus = (text_length / 1000.0) * config.length_bonus_per_1000
        timeout = config.base_timeout * base_weight + history_timeout * config.history_weight + length_bonus
        reason = "history_synthetic"

        # 3. 过度宽松收缩：P90 显著低于当前值时收窄到 P90 × safety_margin + 长度加成。
        #    先于升级执行，保证"超时事件"优先放大（升级覆盖收缩）。
        if stats.p90 > 0 and timeout > 0 and stats.p90 < timeout * config.heuristic_shrink_ratio:
            timeout = stats.p90 * config.safety_margin + length_bonus
            reason = "contraction"

        # 4. 超时事件升级：近期发生超时则放大，避免重复踩超时（优先级最高）
        if recent_timeout_count > 0:
            timeout *= config.heuristic_grow_factor
            reason = "timeout_escalation"

        # 5. 预警：激活 TimeoutHistory 自带的动态超时预测（死代码复活）。
        #    predict_timeout 内部即 ``max(P95, avg) + 长度加成``，与数值版等价；
        #    有原文时走原文，否则用等长空格串保持长度语义一致。
        if text is not None:
            predicted, should_warn = history.predict_timeout(text)
        else:
            predicted = max(stats.p95, stats.average) + length_bonus
            should_warn = predicted > stats.p90 > 0

        return HeuristicSuggestion(
            timeout=timeout,
            reason=reason,
            warn=should_warn,
            details={
                "text_length": text_length,
                "history_size": history.history_size,
                "p95": stats.p95,
                "p90": stats.p90,
                "average": stats.average,
                "recent_timeout_count": recent_timeout_count,
                "predicted": predicted,
            },
        )

    def _history_synthetic_timeout(self, p95: float, avg: float) -> float:
        """历史统计合成：``(P95 × p95_weight + avg × avg_weight) × safety_margin``。"""
        p95_timeout = p95 * self.config.p95_weight
        avg_timeout = avg * self.config.avg_weight
        return (p95_timeout + avg_timeout) * self.config.safety_margin
