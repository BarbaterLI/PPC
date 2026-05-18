from __future__ import annotations

import warnings

warnings.warn(
    "src_m.core.models is deprecated; import from src_m.core.result or src_m.reliability instead",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from src_m.core.result import (
        Result,
        ResultState,
        ExecutionMetrics,
    )

    ResultStatus = ResultState

    from src_m.reliability.execution import (
        TaskResult,
        BatchResult,
    )

    ExecutionResult = Result

    __all__ = [
        "Result",
        "ResultState",
        "ResultStatus",
        "ExecutionMetrics",
        "ExecutionResult",
        "TaskResult",
        "BatchResult",
    ]
except ImportError:
    __all__ = []
