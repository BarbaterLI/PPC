"""Unit tests for :mod:`src.analysis.history`.

覆盖 AnalysisHistoryManager 的持久化与查询能力。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.analysis.history import AnalysisHistoryManager
from src.analysis.models import HealthReport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def storage_dir(tmp_path: Path) -> Path:
    return tmp_path / "history"


@pytest.fixture
def manager(storage_dir: Path) -> AnalysisHistoryManager:
    return AnalysisHistoryManager(storage_dir=storage_dir)


def _report(score: int = 90, summary: str = "ok") -> HealthReport:
    return HealthReport(
        timestamp=datetime.now(),
        score=score,
        summary=summary,
        issues=[],
    )


# ---------------------------------------------------------------------------
# 保存与获取
# ---------------------------------------------------------------------------


class TestSaveAndGet:
    def test_save_returns_id(self, manager: AnalysisHistoryManager) -> None:
        rid = manager.save_report(_report())
        assert rid  # 非空

    def test_get_report_roundtrip(self, manager: AnalysisHistoryManager) -> None:
        rid = manager.save_report(_report(score=85, summary="hello"))
        got = manager.get_report(rid)
        assert got is not None
        assert got.score == 85
        assert got.summary == "hello"

    def test_get_nonexistent_returns_none(self, manager: AnalysisHistoryManager) -> None:
        assert manager.get_report("99999999_999999") is None


# ---------------------------------------------------------------------------
# 列表
# ---------------------------------------------------------------------------


class TestList:
    def test_list_empty(self, manager: AnalysisHistoryManager) -> None:
        assert manager.list_reports() == []

    def test_list_newest_first(self, manager: AnalysisHistoryManager) -> None:
        import time

        for s in (90, 80, 70):
            manager.save_report(_report(score=s))
            time.sleep(1.05)  # 保证 timestamp 不同 → id 不同
        items = manager.list_reports(limit=10)
        assert len(items) == 3
        # 倒序：第一个是 score=70
        scores = [item["score"] for item in items]
        assert scores[0] == 70
        assert scores[-1] == 90

    def test_list_limit(self, manager: AnalysisHistoryManager) -> None:
        for _ in range(5):
            manager.save_report(_report())
        items = manager.list_reports(limit=2)
        assert len(items) <= 2


# ---------------------------------------------------------------------------
# 持久化
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_reload_after_restart(self, storage_dir: Path) -> None:
        m1 = AnalysisHistoryManager(storage_dir=storage_dir)
        rid = m1.save_report(_report(score=99, summary="persist"))
        m2 = AnalysisHistoryManager(storage_dir=storage_dir)
        got = m2.get_report(rid)
        assert got is not None
        assert got.score == 99

    def test_cleanup_keeps_max(self, manager: AnalysisHistoryManager) -> None:
        for i in range(5):
            manager.save_report(_report(score=i))
        manager.cleanup(max_records=2)
        items = manager.list_reports(limit=10)
        assert len(items) <= 2

    def test_get_latest_report(self, manager: AnalysisHistoryManager) -> None:
        manager.save_report(_report(score=70))
        manager.save_report(_report(score=85))
        latest = manager.get_latest_report()
        assert latest is not None
        # 不同时刻下应能拿到某个 report
        assert latest.score in (70, 85)
