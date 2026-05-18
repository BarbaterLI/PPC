"""Error pattern recognizer.

Detects reliability issues by checking circuit breaker states. Since there is
no persistent error history, this analyzer uses current runtime data as a proxy.
More comprehensive error pattern analysis would require persistent error history.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..engine import BaseAnalyzer
from ..models import AnalysisCategory, AnalysisIssue, Severity


class ErrorPatternAnalyzer(BaseAnalyzer):
    """Analyzer for error patterns and reliability issues."""

    def __init__(self) -> None:
        super().__init__(name="ErrorPatternAnalyzer")

    def get_categories(self) -> List[AnalysisCategory]:
        return [AnalysisCategory.RELIABILITY]

    async def analyze(self, context: Optional[Dict[str, Any]] = None) -> List[AnalysisIssue]:
        issues: List[AnalysisIssue] = []

        try:
            from ...reliability import get_circuit_breakers
            breakers = get_circuit_breakers()
        except Exception:
            breakers = {}

        for name, cb in breakers.items():
            state = cb.get_state()
            stats = cb.get_stats()
            if state != "closed":
                issues.append(
                    AnalysisIssue(
                        severity=Severity.CRITICAL if state == "open" else Severity.HIGH,
                        category=AnalysisCategory.RELIABILITY,
                        description=f"熔断器 '{name}' 处于 {state.upper()} 状态",
                        suggestion="检查下游服务健康状态，等待自动恢复或手动重置熔断器",
                        location=f"circuit_breaker:{name}",
                        details={
                            "state": state,
                            "total_calls": stats.total_calls,
                            "failed_calls": stats.failed_calls,
                            "failure_rate": stats.failure_rate,
                        },
                    )
                )

        return issues
