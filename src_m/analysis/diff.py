"""Analysis comparison functionality for PPC9.

Compares two HealthReport objects and produces a DiffResult that describes
what changed between them — new issues, fixed issues, persistent issues,
and the score delta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .models import AnalysisIssue, HealthReport


@dataclass
class DiffResult:
    """Result of comparing two HealthReport snapshots.

    Attributes:
        current_score: Health score of the current report.
        previous_score: Health score of the previous report.
        score_diff: Score difference (current - previous).
        new_issues: Issues present in current but not in previous.
        fixed_issues: Issues present in previous but not in current.
        persistent_issues: Issues present in both reports.
        summary: Human-readable summary of the changes.
    """

    current_score: int = 0
    previous_score: int = 0
    score_diff: int = 0
    new_issues: List[AnalysisIssue] = field(default_factory=list)
    fixed_issues: List[AnalysisIssue] = field(default_factory=list)
    persistent_issues: List[AnalysisIssue] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the diff result to a serializable dictionary."""
        return {
            "current_score": self.current_score,
            "previous_score": self.previous_score,
            "score_diff": self.score_diff,
            "new_issues": [issue.to_dict() for issue in self.new_issues],
            "fixed_issues": [issue.to_dict() for issue in self.fixed_issues],
            "persistent_issues": [issue.to_dict() for issue in self.persistent_issues],
            "summary": self.summary,
        }


class AnalysisDiffer:
    """Compares two HealthReport objects and produces a DiffResult.

    Uses the ``description`` field of each ``AnalysisIssue`` as the unique
    identity key when determining which issues are new, fixed, or persistent.
    """

    def compare(self, current: HealthReport, previous: HealthReport) -> DiffResult:
        """Compare two health reports and return the differences.

        Args:
            current: The newer (current) health report.
            previous: The older (previous) health report.

        Returns:
            A DiffResult describing what changed between the two reports.
        """
        score_diff = current.score - previous.score

        previous_by_desc: Dict[str, AnalysisIssue] = {
            issue.description: issue for issue in previous.issues
        }
        current_by_desc: Dict[str, AnalysisIssue] = {
            issue.description: issue for issue in current.issues
        }

        previous_descriptions = set(previous_by_desc.keys())
        current_descriptions = set(current_by_desc.keys())

        new_descriptions = current_descriptions - previous_descriptions
        fixed_descriptions = previous_descriptions - current_descriptions
        persistent_descriptions = current_descriptions & previous_descriptions

        new_issues = [current_by_desc[desc] for desc in sorted(new_descriptions)]
        fixed_issues = [previous_by_desc[desc] for desc in sorted(fixed_descriptions)]
        persistent_issues = [
            current_by_desc[desc] for desc in sorted(persistent_descriptions)
        ]

        summary = self._build_summary(
            current_score=current.score,
            previous_score=previous.score,
            score_diff=score_diff,
            new_count=len(new_issues),
            fixed_count=len(fixed_issues),
            persistent_count=len(persistent_issues),
            current_summary=current.summary,
            previous_summary=previous.summary,
        )

        return DiffResult(
            current_score=current.score,
            previous_score=previous.score,
            score_diff=score_diff,
            new_issues=new_issues,
            fixed_issues=fixed_issues,
            persistent_issues=persistent_issues,
            summary=summary,
        )

    @staticmethod
    def _build_summary(
        current_score: int,
        previous_score: int,
        score_diff: int,
        new_count: int,
        fixed_count: int,
        persistent_count: int,
        current_summary: str | None,
        previous_summary: str | None,
    ) -> str:
        """Build a human-readable summary string."""
        parts: List[str] = []

        if score_diff > 0:
            parts.append(f"Score improved by +{score_diff} (was {previous_score}, now {current_score})")
        elif score_diff < 0:
            parts.append(f"Score declined by {score_diff} (was {previous_score}, now {current_score})")
        else:
            parts.append(f"Score unchanged at {current_score}")

        if new_count:
            parts.append(f"{new_count} new issue(s) detected")
        if fixed_count:
            parts.append(f"{fixed_count} issue(s) resolved")
        if persistent_count:
            parts.append(f"{persistent_count} issue(s) remain")

        if current_summary and current_summary != previous_summary:
            parts.append(f"Current summary: {current_summary}")

        return "; ".join(parts) + "."


def compute_diff(current: HealthReport, previous: HealthReport) -> DiffResult:
    """Convenience function to diff two health reports.

    Args:
        current: The newer (current) health report.
        previous: The older (previous) health report.

    Returns:
        A DiffResult describing what changed between the two reports.
    """
    return AnalysisDiffer().compare(current, previous)
