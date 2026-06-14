"""Unit tests for :mod:`src_m.analysis.repair`.

覆盖 RepairEngine 的策略注册、修复应用、备份与回滚能力。
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from src_m.analysis.repair import (
    BackupManager,
    CacheCleanupStrategy,
    NetworkRepairStrategy,
    RepairEngine,
    ResourceAdjustmentStrategy,
    StrategyInfo,
)
from src_m.analysis.models import (
    AnalysisIssue,
    AnalysisCategory,
    HealthReport,
    RepairResult,
    RepairSuggestion,
    RiskLevel,
    Severity,
)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _suggestion(action: str = "test", strategy: Optional[str] = None) -> RepairSuggestion:
    return RepairSuggestion(
        action=action,
        risk_level=RiskLevel.MEDIUM,
        strategy_name=strategy,
        auto_applicable=False,
    )


# ---------------------------------------------------------------------------
# StrategyInfo
# ---------------------------------------------------------------------------


class TestStrategyInfo:
    def test_info_construction(self) -> None:
        info = StrategyInfo(
            name="x",
            description="d",
            risk_level=RiskLevel.LOW,
            auto_applicable=True,
        )
        d = info.to_dict()
        assert d["name"] == "x"
        assert d["auto_applicable"] is True


# ---------------------------------------------------------------------------
# RepairEngine 基础
# ---------------------------------------------------------------------------


class TestRepairEngineBasic:
    def test_register_and_list(self) -> None:
        engine = RepairEngine(auto_backup=False)
        engine.register(CacheCleanupStrategy())
        names = engine.list_strategies()
        assert "cache_cleanup" in names

    def test_unregister(self) -> None:
        engine = RepairEngine(auto_backup=False)
        s = CacheCleanupStrategy()
        engine.register(s)
        assert engine.unregister("cache_cleanup") is s
        assert engine.unregister("cache_cleanup") is None

    def test_get_strategy_info(self) -> None:
        engine = RepairEngine(auto_backup=False)
        engine.register(NetworkRepairStrategy())
        info = engine.get_strategy_info()
        assert any(i.name == "network_repair" for i in info)

    def test_get_strategy(self) -> None:
        engine = RepairEngine(auto_backup=False)
        engine.register(ResourceAdjustmentStrategy())
        s = engine.get_strategy("resource_adjustment")
        assert isinstance(s, ResourceAdjustmentStrategy)
        assert engine.get_strategy("nope") is None


# ---------------------------------------------------------------------------
# 备份
# ---------------------------------------------------------------------------


class TestBackup:
    def test_backup_and_restore_file(self, tmp_path: Path) -> None:
        bm = BackupManager(backup_dir=str(tmp_path / "bk"))
        target = tmp_path / "config.json"
        target.write_text("hello", encoding="utf-8")
        bm.create_backup(str(target), backup_id="b1")
        target.write_text("world", encoding="utf-8")
        # restore_backup 接收 backup_id 与 target_path
        assert bm.restore_backup("b1", str(target)) is True
        assert target.read_text(encoding="utf-8") == "hello"

    def test_backup_missing_file_raises(self, tmp_path: Path) -> None:
        bm = BackupManager(backup_dir=str(tmp_path / "bk"))
        with pytest.raises(FileNotFoundError):
            bm.create_backup(str(tmp_path / "missing.txt"))

    def test_list_backups(self, tmp_path: Path) -> None:
        bm = BackupManager(backup_dir=str(tmp_path / "bk"))
        target = tmp_path / "f.txt"
        target.write_text("x", encoding="utf-8")
        bm.create_backup(str(target), backup_id="b1")
        bm.create_backup(str(target), backup_id="b2")
        items = bm.list_backups()
        assert len(items) >= 2


# ---------------------------------------------------------------------------
# 应用修复
# ---------------------------------------------------------------------------


class TestApplyRepair:
    def test_no_strategy_returns_failure(self) -> None:
        engine = RepairEngine(auto_backup=False)
        result = _run(engine.apply(_suggestion("unsupported-action-xyz")))
        assert result.success is False
        assert "No strategy" in (result.error or "")

    def test_apply_with_matching_strategy(self) -> None:
        engine = RepairEngine(auto_backup=False)
        engine.register(NetworkRepairStrategy())
        s = _suggestion("fix-network")
        result = _run(engine.apply(s))
        # 至少不应崩溃，结果由具体策略决定
        assert isinstance(result, RepairResult)

    def test_history_appended(self) -> None:
        engine = RepairEngine(auto_backup=False)
        _run(engine.apply(_suggestion("nope-aaa")))
        _run(engine.apply(_suggestion("nope-bbb")))
        history = engine.get_history()
        assert len(history) >= 2


# ---------------------------------------------------------------------------
# CacheCleanup 策略
# ---------------------------------------------------------------------------


class TestCacheCleanupStrategy:
    def test_strategy_info(self, tmp_path: Path) -> None:
        s = CacheCleanupStrategy()
        info = s.get_info()
        assert info.name == "cache_cleanup"
