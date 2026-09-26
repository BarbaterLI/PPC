from src.core.result import ExecutionMetrics

from .circuit import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    HalfOpenLimiter,
    SimpleCircuitBreaker,
    TripStrategy,
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
    BatchResult,
    ExecutionResult,
    RetryEvent,
    RetryEventType,
    RetryPolicy,
    TaskResult,
    classify_exception,
    create_network_retry_policy,
    create_tts_circuit_breaker,
    create_tts_retry_policy,
)
from .rate_limiter import (
    AsyncSlidingWindow,
    AsyncTokenBucket,
    RateLimitResult,
    RateLimitStrategy,
    RateLimiter,
    SlidingWindowRateLimiter,
    TierRateLimiterConfig,
    TokenBucketRateLimiter,
    TwoTierRateLimiter,
    create_rate_limiter,
    get_rate_limiter,
    set_rate_limiter,
)
from .result import (
    Result,
    ResultState,
)
from .retry import (
    BackoffCurve,
    RetryConfig,
    RetryContext,
    create_default_retry,
    retry,
    retry_async,
    retry_on_failure,
)


def __getattr__(name: str):
    if name == "NetworkError":
        from ..core.exceptions import NetworkError

        return NetworkError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # circuit
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitState",
    "HalfOpenLimiter",
    "TripStrategy",
    "SimpleCircuitBreaker",
    # errors
    "DeadlineExceededError",
    "MaxRetriesError",
    "OperationCancelledError",
    "ResourceExhaustedError",
    # result
    "Result",
    "ResultState",
    # retry
    "RetryConfig",
    "RetryContext",
    "BackoffCurve",
    # execution
    "ExecutionResult",
    "ExecutionMetrics",
    "RetryPolicy",
    "RetryEvent",
    "RetryEventType",
    "TaskResult",
    "BatchResult",
    # rate limiter
    "TokenBucketRateLimiter",
    "SlidingWindowRateLimiter",
    "RateLimiter",
    "AsyncTokenBucket",
    "AsyncSlidingWindow",
    "TwoTierRateLimiter",
    "TierRateLimiterConfig",
    "RateLimitStrategy",
    "RateLimitResult",
    # factories
    "create_circuit_breaker",
    "create_default_retry",
    "create_error_from_exception",
    "create_exception_chain",
    "create_network_retry_policy",
    "create_tts_retry_policy",
    "create_tts_circuit_breaker",
    "create_rate_limiter",
    "classify_exception",
    "format_exception_chain",
    "get_circuit_breakers",
    "get_rate_limiter",
    "set_rate_limiter",
    "retry",
    "retry_async",
    "retry_on_failure",
    "NetworkError",
]
