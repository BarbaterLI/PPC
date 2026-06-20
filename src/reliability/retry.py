"""Retry mechanism

Provides configurable retry strategies including:
- Exponential / linear / fixed backoff curves
- Exception type -> backoff curve mapping
- Deadline to prevent retry storms
- Jitter and async/sync wrappers
"""

import asyncio
import logging
import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class BackoffCurve(str, Enum):
    """Backoff curve type"""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


class RetryContext:
    """Retry context"""

    def __init__(
        self,
        operation: str,
        max_retries: int,
        current_attempt: int,
        delay: float,
    ):
        self.operation = operation
        self.max_retries = max_retries
        self.current_attempt = current_attempt
        self.delay = delay
        self.errors: list[Exception] = []
        self.start_time = time.time()

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def has_more_retries(self) -> bool:
        return self.current_attempt < self.max_retries

    def add_error(self, error: Exception) -> None:
        self.errors.append(error)

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "max_retries": self.max_retries,
            "current_attempt": self.current_attempt,
            "delay": self.delay,
            "elapsed_time": self.elapsed_time,
            "errors_count": len(self.errors),
        }


@dataclass
class RetryConfig:
    """Retry configuration

    Backward compatible: existing fields keep their semantics.
    New fields:
        - ``backoff_curve`` / ``exception_backoff_map`` for curve selection
        - ``deadline`` for retry-storm prevention
        - ``retry_after_extractor`` to honor Retry-After headers
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Sequence[type[Exception]] = (Exception,)
    non_retryable_exceptions: Sequence[type[Exception]] = ()
    retry_condition: Callable[[Exception], bool] | None = None
    before_retry: Callable[["RetryContext"], None] | None = None
    after_retry: Callable[["RetryContext", Exception | None], None] | None = None
    timeout: float | None = None

    # Phase 2 additions -----------------------------------------------------
    backoff_curve: BackoffCurve = BackoffCurve.EXPONENTIAL
    # Mapping from exception type to its own backoff curve parameters
    exception_backoff_map: dict[type[Exception], dict[str, Any]] = field(default_factory=dict)
    # Absolute deadline from the start of the operation; once exceeded, retries stop.
    deadline: float | None = None
    # Optional callable to extract a "retry after" hint (seconds) from an exception
    retry_after_extractor: Callable[[Exception], float | None] | None = None


def _calculate_delay(config: RetryConfig, attempt: int, error: Exception | None = None) -> float:
    """Calculate retry delay according to the configured curve.

    ``error`` is consulted for per-exception overrides and ``retry_after``
    hints.
    """
    base_delay = config.base_delay
    max_delay = config.max_delay
    curve = config.backoff_curve
    exponential_base = config.exponential_base

    if error is not None:
        for exc_type, params in config.exception_backoff_map.items():
            if isinstance(error, exc_type):
                if "base_delay" in params:
                    base_delay = params["base_delay"]
                if "max_delay" in params:
                    max_delay = params["max_delay"]
                if "exponential_base" in params:
                    exponential_base = params["exponential_base"]
                if "curve" in params:
                    curve = BackoffCurve(params["curve"])
                break

    if curve == BackoffCurve.LINEAR:
        delay = base_delay * (attempt + 1)
    elif curve == BackoffCurve.FIXED:
        delay = base_delay
    else:  # EXPONENTIAL
        delay = base_delay * (exponential_base**attempt)

    delay = min(delay, max_delay)

    # Honor Retry-After if present
    if error is not None and config.retry_after_extractor is not None:
        try:
            retry_after = config.retry_after_extractor(error)
            if retry_after is not None and retry_after > delay:
                delay = min(retry_after, max_delay)
        except Exception as ex:  # extractor must not break retries
            logger.debug("retry_after_extractor failed: %s", ex)

    if config.jitter:
        # decorrelated jitter between base_delay and computed delay
        delay = random.uniform(base_delay, max(delay, base_delay))

    return delay


def _should_retry(config: RetryConfig, error: Exception) -> bool:
    """Determine if should retry"""
    if type(error) in config.non_retryable_exceptions:
        return False

    if config.retry_condition:
        return config.retry_condition(error)

    return any(isinstance(error, exc_type) for exc_type in config.retryable_exceptions)


def _deadline_remaining(config: RetryConfig, start_time: float) -> float | None:
    if config.deadline is None:
        return None
    return config.deadline - (time.time() - start_time)


def _is_deadline_exceeded(config: RetryConfig, start_time: float, next_delay: float) -> bool:
    """Return True if waiting ``next_delay`` would breach the deadline."""
    if config.deadline is None:
        return False
    remaining = _deadline_remaining(config, start_time)
    if remaining is None:
        return False
    return remaining <= 0 or remaining < next_delay


def retry(
    func: Callable[..., Any],
    *args: Any,
    config: RetryConfig | None = None,
    operation: str | None = None,
    **kwargs: Any,
) -> Any:
    """Synchronous retry wrapper

    Args:
        func: Function to retry
        *args: Function arguments
        config: Retry configuration
        operation: Operation name
        **kwargs: Function keyword arguments

    Returns:
        Function result

    Raises:
        Exception: Last error after exhausting retries or deadline
    """
    config = config or RetryConfig()
    operation_name = operation or func.__name__
    last_error: Exception | None = None
    start_time = time.time()

    for attempt in range(config.max_retries + 1):
        try:
            if config.timeout:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    try:
                        result = future.result(timeout=config.timeout)
                        return result
                    except FuturesTimeoutError:
                        raise TimeoutError(f"Operation {operation_name} timed out after {config.timeout}s") from None
            else:
                return func(*args, **kwargs)
        except Exception as e:
            last_error = e

            if attempt == config.max_retries or not _should_retry(config, e):
                raise

            delay = _calculate_delay(config, attempt, e)
            if _is_deadline_exceeded(config, start_time, delay):
                logger.warning(
                    "Retry aborted: deadline reached for %s (attempt %d/%d)",
                    operation_name,
                    attempt + 1,
                    config.max_retries,
                )
                raise

            context = RetryContext(
                operation=operation_name,
                max_retries=config.max_retries,
                current_attempt=attempt + 1,
                delay=delay,
            )
            context.add_error(e)

            if config.before_retry:
                config.before_retry(context)

            logger.warning(
                f"Retry {operation_name} attempt {attempt + 1}/{config.max_retries}, waiting {delay:.2f}s, error: {e}"
            )

            time.sleep(delay)

            if config.after_retry:
                config.after_retry(context, e)

    assert last_error is not None  # pragma: no cover - loop always produces an error here
    raise last_error


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    config: RetryConfig | None = None,
    operation: str | None = None,
    **kwargs: Any,
) -> Any:
    """Asynchronous retry wrapper

    Args:
        func: Async function to retry
        *args: Function arguments
        config: Retry configuration
        operation: Operation name
        **kwargs: Function keyword arguments

    Returns:
        Function result

    Raises:
        Exception: Last error after exhausting retries or deadline
    """
    config = config or RetryConfig()
    operation_name = operation or func.__name__
    last_error: Exception | None = None
    start_time = time.time()

    for attempt in range(config.max_retries + 1):
        try:
            if config.timeout:
                try:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=config.timeout,
                    )
                    return result
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Operation {operation_name} timed out after {config.timeout}s") from None
            else:
                return await func(*args, **kwargs)
        except Exception as e:
            last_error = e

            if attempt == config.max_retries or not _should_retry(config, e):
                raise

            delay = _calculate_delay(config, attempt, e)
            if _is_deadline_exceeded(config, start_time, delay):
                logger.warning(
                    "Retry aborted: deadline reached for %s (attempt %d/%d)",
                    operation_name,
                    attempt + 1,
                    config.max_retries,
                )
                raise

            context = RetryContext(
                operation=operation_name,
                max_retries=config.max_retries,
                current_attempt=attempt + 1,
                delay=delay,
            )
            context.add_error(e)

            if config.before_retry:
                if asyncio.iscoroutinefunction(config.before_retry):
                    await config.before_retry(context)
                else:
                    config.before_retry(context)

            logger.warning(
                f"Retry {operation_name} attempt {attempt + 1}/{config.max_retries}, waiting {delay:.2f}s, error: {e}"
            )

            await asyncio.sleep(delay)

            if config.after_retry:
                if asyncio.iscoroutinefunction(config.after_retry):
                    await config.after_retry(context, e)
                else:
                    config.after_retry(context, e)

    assert last_error is not None  # pragma: no cover - loop always produces an error here
    raise last_error


def create_default_retry(max_retries: int = 3, base_delay: float = 1.0) -> RetryConfig:
    """Create default retry configuration

    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds

    Returns:
        RetryConfig instance
    """
    return RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=True,
    )
