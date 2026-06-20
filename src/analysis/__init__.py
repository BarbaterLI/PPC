"""Analysis module for PPC10.

Provides deep analysis, health scoring, and automated repair capabilities.
"""

from .engine import (
    AnalysisEngine,
    AnalyzerStats,
    BaseAnalyzer,
)
from .models import (
    AnalysisCategory,
    AnalysisIssue,
    HealthReport,
    RepairResult,
    RepairSuggestion,
    RiskLevel,
    Severity,
)
from .repair import (
    BackupManager,
    RepairEngine,
    RepairStrategy,
    StrategyInfo,
)

__all__ = [
    # models
    "AnalysisCategory",
    "AnalysisIssue",
    "HealthReport",
    "RepairResult",
    "RepairSuggestion",
    "RiskLevel",
    "Severity",
    # engine
    "AnalysisEngine",
    "AnalyzerStats",
    "BaseAnalyzer",
    # repair
    "BackupManager",
    "RepairEngine",
    "RepairStrategy",
    "StrategyInfo",
]
