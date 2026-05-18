"""Analysis engine implementation.

Provides the AnalysisEngine for running multiple analyzers and aggregating
results, the BaseAnalyzer abstract class, and health scoring algorithms.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

from .models import AnalysisCategory, AnalysisIssue, HealthReport, Severity

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerStats:
    """Statistics for a single analyzer run."""

    analyzer_name: str = ""
    duration_seconds: float = 0.0
    issues_found: int = 0
    errors: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analyzer_name": self.analyzer_name,
            "duration_seconds": self.duration_seconds,
            "issues_found": self.issues_found,
            "errors": self.errors,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class BaseAnalyzer(ABC):
    """Abstract base class for analysis plugins.

    Each analyzer inspects a specific aspect of the system and returns
    a list of AnalysisIssue objects.
    """

    def __init__(self, name: str = "") -> None:
        self._name = name or self.__class__.__name__
        self._enabled = True

    @property
    def name(self) -> str:
        """Return the analyzer name."""
        return self._name

    @property
    def enabled(self) -> bool:
        """Return whether the analyzer is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable the analyzer."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the analyzer."""
        self._enabled = False

    @abstractmethod
    async def analyze(self, context: Optional[Dict[str, Any]] = None) -> List[AnalysisIssue]:
        """Run analysis and return a list of issues.

        Args:
            context: Optional context dictionary with analysis parameters.

        Returns:
            List of AnalysisIssue objects.
        """

    def get_categories(self) -> List[AnalysisCategory]:
        """Return the categories this analyzer covers.

        Override to declare categories; empty list means unknown.
        """
        return []


class AnalysisEngine:
    """Deep analysis engine that runs multiple analyzers and aggregates results.

    Supports concurrent analyzer execution, health score computation (0-100),
    and aggregated reporting.
    """

    def __init__(self, max_concurrent: int = 4) -> None:
        self._analyzers: Dict[str, BaseAnalyzer] = {}
        self._max_concurrent = max_concurrent
        self._stats: List[AnalyzerStats] = []
        self._last_report: Optional[HealthReport] = None

    def register(self, analyzer: BaseAnalyzer) -> None:
        """Register an analyzer."""
        if analyzer.name in self._analyzers:
            logger.warning("Analyzer %s already registered, replacing", analyzer.name)
        self._analyzers[analyzer.name] = analyzer

    def unregister(self, name: str) -> Optional[BaseAnalyzer]:
        """Unregister an analyzer by name."""
        return self._analyzers.pop(name, None)

    def get_analyzer(self, name: str) -> Optional[BaseAnalyzer]:
        """Get a registered analyzer by name."""
        return self._analyzers.get(name)

    def list_analyzers(self) -> List[str]:
        """Return a list of registered analyzer names."""
        return list(self._analyzers.keys())

    def enable_analyzer(self, name: str) -> bool:
        """Enable an analyzer by name."""
        analyzer = self._analyzers.get(name)
        if analyzer:
            analyzer.enable()
            return True
        return False

    def disable_analyzer(self, name: str) -> bool:
        """Disable an analyzer by name."""
        analyzer = self._analyzers.get(name)
        if analyzer:
            analyzer.disable()
            return True
        return False

    async def run(
        self,
        context: Optional[Dict[str, Any]] = None,
        analyzer_names: Optional[List[str]] = None,
    ) -> HealthReport:
        """Run selected analyzers and return an aggregated health report.

        Args:
            context: Optional context passed to each analyzer.
            analyzer_names: Optional list of analyzer names to run.
                If None, all enabled analyzers are run.

        Returns:
            Aggregated HealthReport.
        """
        targets = self._select_analyzers(analyzer_names)
        if not targets:
            logger.warning("No analyzers selected for run")
            return HealthReport(
                score=100,
                summary="No analyzers selected",
            )

        self._stats = []
        all_issues: List[AnalysisIssue] = []
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def _run_one(analyzer: BaseAnalyzer) -> List[AnalysisIssue]:
            async with semaphore:
                stats = AnalyzerStats(
                    analyzer_name=analyzer.name,
                    started_at=datetime.now(UTC),
                )
                start = asyncio.get_event_loop().time()
                try:
                    issues = await analyzer.analyze(context)
                    stats.issues_found = len(issues)
                    return issues
                except Exception as exc:
                    logger.exception("Analyzer %s failed", analyzer.name)
                    stats.errors.append(str(exc))
                    return []
                finally:
                    stats.duration_seconds = asyncio.get_event_loop().time() - start
                    stats.completed_at = datetime.now(UTC)
                    self._stats.append(stats)

        results = await asyncio.gather(*[_run_one(a) for a in targets])
        for issues in results:
            all_issues.extend(issues)

        score = self._compute_health_score(all_issues)
        summary = self._generate_summary(all_issues, score)

        report = HealthReport(
            score=score,
            issues=all_issues,
            summary=summary,
            metrics={
                "analyzers_run": len(targets),
                "total_issues": len(all_issues),
                "analyzer_stats": [s.to_dict() for s in self._stats],
            },
        )
        self._last_report = report
        return report

    def _select_analyzers(self, names: Optional[List[str]] = None) -> List[BaseAnalyzer]:
        """Select analyzers to run."""
        if names is not None:
            return [
                self._analyzers[n]
                for n in names
                if n in self._analyzers and self._analyzers[n].enabled
            ]
        return [a for a in self._analyzers.values() if a.enabled]

    def _compute_health_score(self, issues: List[AnalysisIssue]) -> int:
        """Compute a health score from 0 to 100 based on issues.

        Scoring algorithm:
        - Start at 100.
        - Critical: -25 each (max -75)
        - High: -15 each (max -45)
        - Medium: -8 each (max -24)
        - Low: -3 each (max -9)
        - Info: -1 each (max -3)
        - Clamp to [0, 100].
        """
        score = 100
        penalties = {
            Severity.CRITICAL: (25, 3),
            Severity.HIGH: (15, 3),
            Severity.MEDIUM: (8, 3),
            Severity.LOW: (3, 3),
            Severity.INFO: (1, 3),
        }

        counts: Dict[Severity, int] = {}
        for issue in issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1

        for severity, (penalty, max_count) in penalties.items():
            count = counts.get(severity, 0)
            score -= penalty * min(count, max_count)

        return max(0, min(100, score))

    def _generate_summary(self, issues: List[AnalysisIssue], score: int) -> str:
        """Generate a human-readable summary."""
        if not issues:
            return "No issues detected. System health is excellent."

        counts: Dict[Severity, int] = {}
        for issue in issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1

        parts = [f"Health score: {score}/100"]
        for sev in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO):
            count = counts.get(sev, 0)
            if count:
                parts.append(f"{sev.value}: {count}")

        return "; ".join(parts)

    def get_last_report(self) -> Optional[HealthReport]:
        """Return the most recent health report."""
        return self._last_report

    def get_stats(self) -> List[AnalyzerStats]:
        """Return statistics from the last run."""
        return list(self._stats)

    def get_issues_by_category(self, report: Optional[HealthReport] = None) -> Dict[AnalysisCategory, List[AnalysisIssue]]:
        """Group issues by category from the given or last report."""
        target = report or self._last_report
        if target is None:
            return {}
        return target.issues_by_category()

    def get_top_issues(self, limit: int = 10, report: Optional[HealthReport] = None) -> List[AnalysisIssue]:
        """Return the top issues sorted by severity."""
        target = report or self._last_report
        if target is None:
            return []

        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }
        sorted_issues = sorted(target.issues, key=lambda i: severity_order.get(i.severity, 99))
        return sorted_issues[:limit]
