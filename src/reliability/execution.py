import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from src.core.result import Result

from .retry import RetryConfig, _calculate_delay

if TYPE_CHECKING:
    from .circuit import CircuitBreaker

T = TypeVar("T")

ExecutionResult = Result


@dataclass
class TaskResult:
    task_id: str = ""
    success: bool = False
    output_path: Path | None = None
    duration: float = 0.0
    output_size: int = 0
    attempts: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "output_path": str(self.output_path) if self.output_path else None,
            "duration": self.duration,
            "output_size": self.output_size,
            "attempts": self.attempts,
            "error": self.error,
        }


@dataclass
class BatchResult:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    duration: float = 0.0
    results: list[TaskResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "duration": self.duration,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class RetryPolicy(RetryConfig):
    """Retry policy for executors.

    Subclass of :class:`~src.reliability.retry.RetryConfig` to avoid
    duplicating retry configuration fields. ``get_delay`` delegates to
    :func:`~src.reliability.retry._calculate_delay` so there is a single
    delay-calculation implementation.
    """

    # Override jitter to preserve float semantics: callers (config schema,
    # factory functions) pass float values such as 0.1 / 0.05.
    jitter: float = 0.1

    def get_delay(self, attempt: int) -> float:
        return _calculate_delay(self, attempt)

    def should_retry(self, attempt: int, error: Exception) -> bool:
        if attempt >= self.max_retries:
            return False
        return any(isinstance(error, exc_type) for exc_type in self.retryable_exceptions)

    def to_dict(self) -> dict:
        return {
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "exponential_base": self.exponential_base,
            "jitter": self.jitter,
        }


class RetryEventType(str, Enum):
    RETRY = "retry"
    GIVE_UP = "give_up"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker"
    RECOVERY = "recovery"


@dataclass
class RetryEvent:
    event_type: RetryEventType = RetryEventType.RETRY
    attempt: int = 0
    max_retries: int = 3
    error: str | None = None
    delay: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value
            if isinstance(self.event_type, RetryEventType)
            else str(self.event_type),
            "attempt": self.attempt,
            "max_retries": self.max_retries,
            "error": self.error,
            "delay": self.delay,
            "timestamp": self.timestamp,
        }


def create_network_retry_policy(
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
) -> RetryPolicy:
    return RetryPolicy(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
    )


def create_tts_retry_policy(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
) -> RetryPolicy:
    return RetryPolicy(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
    )


def create_tts_circuit_breaker(
    name: str = "tts",
    failure_threshold: int = 5,
    timeout_seconds: float = 60.0,
    success_threshold: int = 3,
    half_open_max_calls: int = 3,
    window_seconds: float = 60.0,
    trip_strategy: "TripStrategy | str" = "combined",
    failure_rate_threshold: float = 0.5,
    minimum_calls: int = 10,
    slow_call_duration_threshold: float = 30.0,
    slow_call_rate_threshold: float = 0.8,
    consecutive_failure_threshold: int = 5,
    half_open_canary_ratio: float = 0.05,
) -> "CircuitBreaker":
    from .circuit import (
        CircuitBreakerConfig,
        TripStrategy,
        get_or_create_circuit_breaker,
    )

    if isinstance(trip_strategy, str):
        trip_strategy = TripStrategy(trip_strategy)

    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout=timeout_seconds,
        success_threshold=success_threshold,
        half_open_max_calls=half_open_max_calls,
        failure_rate_threshold=failure_rate_threshold,
        minimum_calls=minimum_calls,
        slow_call_duration_threshold=slow_call_duration_threshold,
        slow_call_rate_threshold=slow_call_rate_threshold,
        consecutive_failure_threshold=consecutive_failure_threshold,
        trip_strategy=trip_strategy,
        half_open_canary_ratio=half_open_canary_ratio,
        # Real time-based sliding window (previously the legacy count-decay
        # counter was silently used because window_seconds was fed into
        # ``sliding_window_size``).
        sliding_window_seconds=window_seconds,
    )
    # Shared registry instance: repeated engine constructions with the same
    # name return the SAME breaker instead of clobbering the registry entry
    # and isolating state. The caller (TTSEngine) explicitly resets state at
    # construction so request-history does not bleed across engines.
    return get_or_create_circuit_breaker(name, config)


def classify_exception(error: Exception) -> str:
    error_msg = str(error).lower()

    if "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
        return "network"
    if "auth" in error_msg or "permission" in error_msg:
        return "auth"
    if "rate" in error_msg or "limit" in error_msg:
        return "rate_limit"
    if "quota" in error_msg:
        return "quota"
    if "parse" in error_msg or "json" in error_msg:
        return "parse"
    return "unknown"
