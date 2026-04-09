"""可靠性层 - 错误处理、重试、熔断
提供统一的Result类型、错误分类、重试策略、熔断器

修复记录 (2026-04-08):
- 添加 TaskInfo 导出，消除 core/models 的重复定义
"""

from .result import (
    ExecutionResult,
    ExecutionMetrics,
    TaskResult,
    BatchResult,
    ResultStatus,
    TaskInfo,
)

from .errors import (
    PPC7Error,
    PPC6Error,
    NetworkError,
    NetworkTimeoutError,
    IOError,
    FileNotFoundError,
    ConfigError,
    ValidationError,
    RuntimeError,
    MemoryError,
    RateLimitError,
    TaskTimeoutError,
    classify_error,
    ErrorCategory,
    ErrorCode,
)

from .retry import (
    RetryConfig,
    RetryPolicy,
    RetryStats,
    ErrorTypeStats,
    RetryEvent,
    RetryEventType,
    RetryableError,
    NonRetryableError,
    NetworkError as RetryNetworkError,
    RateLimitError as RetryRateLimitError,
    TimeoutError as RetryTimeoutError,
    ServiceUnavailableError,
    AuthenticationError,
    ContentError,
    ErrorSpecificRetryConfig,
    create_network_retry_policy,
    create_aggressive_retry_policy,
    create_conservative_retry_policy,
    create_tts_retry_policy,
    create_error_specific_retry_policy,
    classify_exception,
    is_edge_tts_parameter_error,
)

from .circuit import (
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitState,
    CircuitConfig,
    CircuitStats,
    CircuitOpenError,
    create_tts_circuit_breaker,
    create_network_circuit_breaker,
)

__all__ = [
    # Result types
    "ExecutionResult",
    "ExecutionMetrics",
    "TaskResult",
    "BatchResult",
    "ResultStatus",
    "TaskInfo",
    # Error types
    "PPC7Error",
    "PPC6Error",
    "NetworkError",
    "NetworkTimeoutError",
    "IOError",
    "FileNotFoundError",
    "ConfigError",
    "ValidationError",
    "RuntimeError",
    "MemoryError",
    "RateLimitError",
    "TaskTimeoutError",
    "classify_error",
    "ErrorCategory",
    "ErrorCode",
    # Retry types
    "RetryConfig",
    "RetryPolicy",
    "RetryStats",
    "ErrorTypeStats",
    "RetryEvent",
    "RetryEventType",
    "RetryableError",
    "NonRetryableError",
    "RetryNetworkError",
    "RetryRateLimitError",
    "RetryTimeoutError",
    "ServiceUnavailableError",
    "AuthenticationError",
    "ContentError",
    "ErrorSpecificRetryConfig",
    # Retry factories
    "create_network_retry_policy",
    "create_aggressive_retry_policy",
    "create_conservative_retry_policy",
    "create_tts_retry_policy",
    "create_error_specific_retry_policy",
    "classify_exception",
    "is_edge_tts_parameter_error",
    # Circuit breaker types
    "CircuitBreaker",
    "CircuitBreakerManager",
    "CircuitState",
    "CircuitConfig",
    "CircuitStats",
    "CircuitOpenError",
    "create_tts_circuit_breaker",
    "create_network_circuit_breaker",
]
