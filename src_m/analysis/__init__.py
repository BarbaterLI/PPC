"""Analysis module for PPC10.

Provides deep analysis, health scoring, and automated repair capabilities.
"""

from .models import (
    AnalysisCategory,
    AnalysisIssue,
    HealthReport,
    RepairResult,
    RepairSuggestion,
    RiskLevel,
    Severity,
)
from .engine import (
    AnalysisEngine,
    AnalyzerStats,
    BaseAnalyzer,
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
