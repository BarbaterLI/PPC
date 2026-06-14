"""Unit tests for :mod:`src_m.analysis.analyzers.errors`.

覆盖 ErrorPatternAnalyzer 的熔断器状态检测能力。
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from src_m.analysis.analyzers.errors import ErrorPatternAnalyzer
from src_m.analysis.models import AnalysisCategory


def _run(coro):
    return asyncio.run(coro)


class _FakeCircuit:
    """可注入 state / stats 的熔断器替身。"""

    def __init__(self, state: str = "closed", failed: int = 0, total: int = 10) -> None:
        self._state = state
        self._failed = failed
        self._total = total

    def get_state(self) -> str:
        return self._state

    def get_stats(self) -> Any:
        class _Stats:
            total_calls: int = 0
            failed_calls: int = 0
            failure_rate: float = 0.0

        s = _Stats()
        s.total_calls = self._total
        s.failed_calls = self._failed
        s.failure_rate = self._failed / self._total if self._total else 0.0
        return s


# ---------------------------------------------------------------------------
# 基本能力
# ---------------------------------------------------------------------------


class TestErrorAnalyzerBasic:
    def test_name(self) -> None:
        a = ErrorPatternAnalyzer()
        assert a.name == "ErrorPatternAnalyzer"

    def test_categories(self) -> None:
        cats = ErrorPatternAnalyzer().get_categories()
        assert AnalysisCategory.RELIABILITY in cats


# ---------------------------------------------------------------------------
# 分析执行
# ---------------------------------------------------------------------------


class TestErrorAnalyzerRun:
    def test_no_breakers_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 让 get_circuit_breakers 返回空 dict
        import src_m.reliability as rel_mod

        monkeypatch.setattr(rel_mod, "get_circuit_breakers", lambda: {})
        issues = _run(ErrorPatternAnalyzer().analyze())
        assert issues == []

    def test_open_breaker_produces_critical(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src_m.reliability as rel_mod

        monkeypatch.setattr(
            rel_mod,
            "get_circuit_breakers",
            lambda: {"b1": _FakeCircuit(state="open", failed=5, total=10)},
        )
        issues = _run(ErrorPatternAnalyzer().analyze())
        assert any("OPEN" in (i.description or "").upper() for i in issues)

    def test_closed_breaker_produces_no_issue(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src_m.reliability as rel_mod

        monkeypatch.setattr(
            rel_mod,
            "get_circuit_breakers",
            lambda: {"b1": _FakeCircuit(state="closed", failed=0, total=10)},
        )
        issues = _run(ErrorPatternAnalyzer().analyze())
        assert issues == []
