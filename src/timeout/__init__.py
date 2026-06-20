"""Timeout module

Provides dynamic timeout calculation, timeout history tracking, and percentile-based timeout adjustment.
"""

from .calculator import (
    TimeoutCalculator,
    TimeoutConfig,
    TimeoutResult,
    calculate_timeout,
    get_calculator,
)
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
    "calculate_timeout",
    "get_calculator",
]
