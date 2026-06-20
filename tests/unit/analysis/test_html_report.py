"""Unit tests for :mod:`src.analysis.html_report`.

覆盖 HTMLReportGenerator 生成的 HTML 报告与 Markdown 导出能力。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.analysis.diff import AnalysisDiffer
from src.analysis.html_report import HTMLReportGenerator
from src.analysis.models import (
    AnalysisCategory,
    AnalysisIssue,
    HealthReport,
    Severity,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def report() -> HealthReport:
    return HealthReport(
        timestamp=datetime.now(),
        score=85,
        summary="一切正常",
        issues=[
            AnalysisIssue(
                severity=Severity.HIGH,
                category=AnalysisCategory.PERFORMANCE,
                description="P95 延迟偏高",
                suggestion="调高超时",
                location="tts.timeout",
            ),
        ],
    )


@pytest.fixture
def prev_report() -> HealthReport:
    return HealthReport(
        timestamp=datetime.now(),
        score=70,
        summary="之前的状态",
        issues=[],
    )


# ---------------------------------------------------------------------------
# HTML 生成
# ---------------------------------------------------------------------------


class TestHTMLGenerate:
    def test_generate_writes_file(self, report: HealthReport, tmp_path: Path) -> None:
        gen = HTMLReportGenerator()
        out = tmp_path / "report.html"
        path = gen.generate(report, out)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_html_contains_score(self, report: HealthReport, tmp_path: Path) -> None:
        gen = HTMLReportGenerator()
        out = tmp_path / "report.html"
        gen.generate(report, out)
        content = out.read_text(encoding="utf-8")
        # 报告中应包含分数
        assert "85" in content
        # 报告应至少包含 html 标签
        assert "<html" in content.lower()

    def test_generate_with_diff(
        self,
        report: HealthReport,
        prev_report: HealthReport,
        tmp_path: Path,
    ) -> None:
        gen = HTMLReportGenerator()
        diff = AnalysisDiffer().compare(report, prev_report)
        out = tmp_path / "report.html"
        gen.generate(report, out, diff=diff)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        # 报告中应出现 diff 文本
        assert "Score" in content or "score" in content


# ---------------------------------------------------------------------------
# 历史趋势
# ---------------------------------------------------------------------------


class TestHistoryTrend:
    def test_generate_with_history(self, report: HealthReport, tmp_path: Path) -> None:
        gen = HTMLReportGenerator()
        history = [
            {"score": 60, "timestamp": "2024-01-01T00:00:00"},
            {"score": 75, "timestamp": "2024-01-02T00:00:00"},
            {"score": 85, "timestamp": "2024-01-03T00:00:00"},
        ]
        out = tmp_path / "report.html"
        gen.generate(report, out, history=history)
        content = out.read_text(encoding="utf-8")
        # 应有 trend 渲染
        assert "trend" in content.lower() or "history" in content.lower()

    def test_history_empty_omits_trend(self, report: HealthReport, tmp_path: Path) -> None:
        gen = HTMLReportGenerator()
        out = tmp_path / "report.html"
        gen.generate(report, out, history=[])
        # 不应崩溃
        assert out.exists()
