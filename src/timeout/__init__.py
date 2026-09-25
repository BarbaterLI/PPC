"""Timeout module

Provides dynamic timeout calculation, timeout history tracking, percentile-based
timeout adjustment, and the synthetic-mode heuristic controller.
"""

from .calculator import (
    TimeoutCalculator,
    TimeoutConfig,
    TimeoutResult,
    calculate_timeout,
    get_calculator,
)
from .heuristics import HeuristicSuggestion, TimeoutHeuristicController
from .history import (
    RequestRecord,
    TimeoutHistory,
    TimeoutStatistics,
)

__all__ = [
    "TimeoutCalculator",
    "TimeoutConfig",
    "TimeoutResult",
    "RequestRecord",
    "TimeoutHistory",
    "TimeoutStatistics",
    "TimeoutHeuristicController",
    "HeuristicSuggestion",
    "calculate_timeout",
    "get_calculator",
]
