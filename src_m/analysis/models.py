"""Analysis module data models.

Defines data models for analysis issues, repair suggestions, health reports,
and related enumerations used by the analysis and repair engines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class AnalysisCategory(str, Enum):
    """Categories of analysis issues."""

    PERFORMANCE = "performance"
    MEMORY = "memory"
    RELIABILITY = "reliability"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    DEPENDENCY = "dependency"
    CODE_QUALITY = "code_quality"
    RESOURCE = "resource"
    NETWORK = "network"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Risk levels for repair suggestions and issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class Severity(str, Enum):
    """Severity levels for analysis issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AnalysisIssue:
    """Represents a single analysis issue found during inspection.

    Attributes:
        severity: The severity level of the issue.
        category: The category the issue belongs to.
        description: Human-readable description of the issue.
        suggestion: Optional suggestion for resolving the issue.
        location: Optional location identifier (e.g., file path, module name).
        details: Additional structured details about the issue.
        timestamp: When the issue was detected.
    """

    severity: Severity = Severity.MEDIUM
    category: AnalysisCategory = AnalysisCategory.UNKNOWN
    description: str = ""
    suggestion: Optional[str] = None
    location: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to a serializable dictionary."""
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "description": self.description,
            "suggestion": self.suggestion,
            "location": self.location,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisIssue":
        """Create an issue from a dictionary."""
        return cls(
            severity=Severity(data.get("severity", "medium")),
            category=AnalysisCategory(data.get("category", "unknown")),
            description=data.get("description", ""),
            suggestion=data.get("suggestion"),
            location=data.get("location"),
            details=data.get("details", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(UTC),
        )


@dataclass
class RepairSuggestion:
    """Represents a suggested repair action.

    Attributes:
        action: Human-readable description of the repair action.
        risk_level: Estimated risk of applying the repair.
        expected_impact: Description of the expected impact after repair.
        strategy_name: Optional name of the repair strategy to use.
        parameters: Optional parameters for the repair strategy.
        auto_applicable: Whether the repair can be applied automatically.
    """

    action: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    expected_impact: str = ""
    strategy_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    auto_applicable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert suggestion to a serializable dictionary."""
        return {
            "action": self.action,
            "risk_level": self.risk_level.value,
            "expected_impact": self.expected_impact,
            "strategy_name": self.strategy_name,
            "parameters": self.parameters,
            "auto_applicable": self.auto_applicable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepairSuggestion":
        """Create a suggestion from a dictionary."""
        return cls(
            action=data.get("action", ""),
            risk_level=RiskLevel(data.get("risk_level", "low")),
            expected_impact=data.get("expected_impact", ""),
            strategy_name=data.get("strategy_name"),
            parameters=data.get("parameters", {}),
            auto_applicable=data.get("auto_applicable", False),
        )


@dataclass
class HealthReport:
    """Aggregated health report for a system or component.

    Attributes:
        score: Health score from 0 to 100.
        issues: List of detected issues.
        timestamp: When the report was generated.
        component: Optional name of the component being reported on.
        summary: Optional human-readable summary.
        metrics: Optional structured metrics used to compute the score.
    """

    score: int = 100
    issues: List[AnalysisIssue] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    component: Optional[str] = None
    summary: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to a serializable dictionary."""
        return {
            "score": self.score,
            "issues": [issue.to_dict() for issue in self.issues],
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "summary": self.summary,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealthReport":
        """Create a report from a dictionary."""
        return cls(
            score=data.get("score", 100),
            issues=[AnalysisIssue.from_dict(i) for i in data.get("issues", [])],
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(UTC),
            component=data.get("component"),
            summary=data.get("summary"),
            metrics=data.get("metrics", {}),
        )

    def critical_count(self) -> int:
        """Return the number of critical issues."""
        return sum(1 for issue in self.issues if issue.severity == Severity.CRITICAL)

    def high_count(self) -> int:
        """Return the number of high severity issues."""
        return sum(1 for issue in self.issues if issue.severity == Severity.HIGH)

    def issues_by_category(self) -> Dict[AnalysisCategory, List[AnalysisIssue]]:
        """Group issues by category."""
        result: Dict[AnalysisCategory, List[AnalysisIssue]] = {}
        for issue in self.issues:
            result.setdefault(issue.category, []).append(issue)
        return result


@dataclass
class RepairResult:
    """Result of a repair operation.

    Attributes:
        success: Whether the repair succeeded.
        message: Human-readable result message.
        backup_path: Optional path to a backup created before repair.
        rolled_back: Whether the repair was rolled back.
        error: Optional error message if the repair failed.
        metrics: Optional metrics about the repair operation.
    """

    success: bool = False
    message: str = ""
    backup_path: Optional[str] = None
    rolled_back: bool = False
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "backup_path": self.backup_path,
            "rolled_back": self.rolled_back,
            "error": self.error,
            "metrics": self.metrics,
        }

    @classmethod
    def success_result(
        cls,
        message: str = "",
        backup_path: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> "RepairResult":
        """Create a successful repair result."""
        return cls(
            success=True,
            message=message,
            backup_path=backup_path,
            metrics=metrics or {},
        )

    @classmethod
    def failure_result(
        cls,
        error: str = "",
        message: str = "",
        backup_path: Optional[str] = None,
        rolled_back: bool = False,
    ) -> "RepairResult":
        """Create a failed repair result."""
        return cls(
            success=False,
            message=message,
            error=error,
            backup_path=backup_path,
            rolled_back=rolled_back,
        )
