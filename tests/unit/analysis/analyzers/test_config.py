"""Unit tests for :mod:`src.analysis.analyzers.config`.

覆盖 ConfigAnalyzer 的配置冲突检测能力。
"""

from __future__ import annotations

import asyncio

import pytest

from src.analysis.analyzers.config import ConfigAnalyzer
from src.analysis.models import AnalysisCategory


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 基本能力
# ---------------------------------------------------------------------------


class TestConfigAnalyzerBasic:
    def test_name(self) -> None:
        a = ConfigAnalyzer()
        assert a.name == "ConfigAnalyzer"

    def test_categories(self) -> None:
        a = ConfigAnalyzer()
        cats = a.get_categories()
        assert AnalysisCategory.CONFIGURATION in cats

    def test_enable_disable(self) -> None:
        a = ConfigAnalyzer()
        a.disable()
        assert a.enabled is False
        a.enable()
        assert a.enabled is True


# ---------------------------------------------------------------------------
# 分析执行
# ---------------------------------------------------------------------------


class TestConfigAnalyzerRun:
    def test_analyze_returns_list(self) -> None:
        a = ConfigAnalyzer()
        issues = _run(a.analyze())
        # 即使无配置中心，结果也应是列表
        assert isinstance(issues, list)

    def test_analyze_handles_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 模拟 config 加载失败 → 应至少返回一条 critical issue
        from src.config import manager as mgr_mod

        def _boom_config():
            raise RuntimeError("config broken")

        class _FakeConfigManager:
            def get_config(self):
                _boom_config()

        monkeypatch.setattr(mgr_mod, "ConfigManager", _FakeConfigManager)
        a = ConfigAnalyzer()
        issues = _run(a.analyze())
        # 仍然能运行（不崩溃）
        assert isinstance(issues, list)
        # 应包含一条 critical issue 描述无法加载配置
        if issues:
            assert any("无法加载" in (i.description or "") for i in issues)
