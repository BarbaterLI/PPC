from .circuit import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    HalfOpenLimiter,
    create_circuit_breaker,
    get_circuit_breakers,
)
from .errors import (
    CircuitBreakerError,
    DeadlineExceededError,
    MaxRetriesError,
    OperationCancelledError,
    ResourceExhaustedError,
    create_error_from_exception,
    create_exception_chain,
    format_exception_chain,
)
from .execution import (
    ExecutionResult,
    ExecutionMetrics,
    RetryPolicy,
    RetryEvent,
    RetryEventType,
    TaskResult,
    BatchResult,
    create_network_retry_policy,
    create_tts_retry_policy,
    create_tts_circuit_breaker,
    classify_exception,
)
from .result import (
    Err,
    Ok,
    Result,
    ResultState,
    is_err,
    is_ok,
)
from .retry import (
    RetryConfig,
    RetryContext,
    create_default_retry,
    retry,
    retry_async,
)

def __getattr__(name: str):
    if name == "NetworkError":
        from ..core.exceptions import NetworkError
        return NetworkError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitState",
    "DeadlineExceededError",
    "HalfOpenLimiter",
    "MaxRetriesError",
    "OperationCancelledError",
    "ResourceExhaustedError",
    "Result",
    "ResultState",
    "RetryConfig",
    "RetryContext",
    "Ok",
    "Err",
    "ExecutionResult",
    "ExecutionMetrics",
    "RetryPolicy",
    "RetryEvent",
    "RetryEventType",
    "TaskResult",
    "BatchResult",
    "create_circuit_breaker",
    "create_default_retry",
    "create_error_from_exception",
    "create_exception_chain",
    "create_network_retry_policy",
    "create_tts_retry_policy",
    "create_tts_circuit_breaker",
    "classify_exception",
    "format_exception_chain",
    "get_circuit_breakers",
    "is_ok",
    "is_err",
    "retry",
    "retry_async",
    "NetworkError",
]
