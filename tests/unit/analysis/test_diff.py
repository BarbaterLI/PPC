"""Unit tests for :mod:`src.analysis.diff`.

覆盖 AnalysisDiffer 对 HealthReport 的多维比较。
"""

from __future__ import annotations

from datetime import datetime

from src.analysis.diff import AnalysisDiffer, DiffResult, compute_diff
from src.analysis.models import (
    AnalysisCategory,
    AnalysisIssue,
    HealthReport,
    Severity,
)

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _report(
    score: int = 90,
    issues: list[AnalysisIssue] = None,
    summary: str = "",
) -> HealthReport:
    return HealthReport(
        timestamp=datetime.now(),
        score=score,
        summary=summary,
        issues=issues or [],
    )


def _issue(desc: str, sev: Severity = Severity.MEDIUM) -> AnalysisIssue:
    return AnalysisIssue(
        severity=sev,
        category=AnalysisCategory.UNKNOWN,
        description=desc,
        location="src/x.py",
        suggestion="fix it",
    )


# ---------------------------------------------------------------------------
# 基本对比
# ---------------------------------------------------------------------------


class TestBasicCompare:
    def test_identical_reports(self) -> None:
        r1 = _report(score=85, issues=[_issue("a")])
        r2 = _report(score=85, issues=[_issue("a")])
        result = AnalysisDiffer().compare(r1, r2)
        assert result.score_diff == 0
        assert result.new_issues == []
        assert result.fixed_issues == []
        assert result.persistent_issues == [_issue("a")] or len(result.persistent_issues) == 1

    def test_score_improved(self) -> None:
        prev = _report(score=70)
        cur = _report(score=90)
        result = AnalysisDiffer().compare(cur, prev)
        assert result.score_diff == 20
        assert "improved" in result.summary.lower()

    def test_score_declined(self) -> None:
        prev = _report(score=90)
        cur = _report(score=60)
        result = AnalysisDiffer().compare(cur, prev)
        assert result.score_diff == -30
        assert "declined" in result.summary.lower()


# ---------------------------------------------------------------------------
# 问题分类
# ---------------------------------------------------------------------------


class TestIssueClassification:
    def test_new_issues(self) -> None:
        prev = _report(issues=[])
        cur = _report(issues=[_issue("new-bug")])
        result = AnalysisDiffer().compare(cur, prev)
        assert len(result.new_issues) == 1
        assert result.new_issues[0].description == "new-bug"
        assert result.fixed_issues == []

    def test_fixed_issues(self) -> None:
        prev = _report(issues=[_issue("old-bug")])
        cur = _report(issues=[])
        result = AnalysisDiffer().compare(cur, prev)
        assert len(result.fixed_issues) == 1
        assert result.fixed_issues[0].description == "old-bug"
        assert result.new_issues == []

    def test_persistent_issues(self) -> None:
        prev = _report(issues=[_issue("p1")])
        cur = _report(issues=[_issue("p1")])
        result = AnalysisDiffer().compare(cur, prev)
        assert len(result.persistent_issues) == 1
        assert result.new_issues == []
        assert result.fixed_issues == []


# ---------------------------------------------------------------------------
# DiffResult / compute_diff
# ---------------------------------------------------------------------------


class TestDiffResult:
    def test_to_dict(self) -> None:
        result = DiffResult(
            current_score=80,
            previous_score=70,
            score_diff=10,
            new_issues=[_issue("x")],
            fixed_issues=[],
            persistent_issues=[],
            summary="+10",
        )
        d = result.to_dict()
        assert d["current_score"] == 80
        assert d["score_diff"] == 10
        assert d["new_issues"][0]["description"] == "x"
        assert d["summary"] == "+10"

    def test_compute_diff_function(self) -> None:
        prev = _report(score=50)
        cur = _report(score=70)
        result = compute_diff(cur, prev)
        assert result.score_diff == 20
        assert isinstance(result, DiffResult)
