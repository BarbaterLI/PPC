"""Retry mechanism

Provides configurable retry strategies including exponential backoff, jitter, and circuit breaker integration.
"""

import asyncio
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Type, Union

logger = logging.getLogger(__name__)


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
        self.errors: List[Exception] = []
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
    """Retry configuration"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Sequence[Type[Exception]] = (Exception,)
    non_retryable_exceptions: Sequence[Type[Exception]] = ()
    retry_condition: Optional[Callable[[Exception], bool]] = None
    before_retry: Optional[Callable[["RetryContext"], None]] = None
    after_retry: Optional[Callable[["RetryContext", Optional[Exception]], None]] = None
    timeout: Optional[float] = None


def _calculate_delay(config: RetryConfig, attempt: int) -> float:
    """Calculate retry delay with exponential backoff and jitter"""
    delay = config.base_delay * (config.exponential_base ** attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        delay = random.uniform(config.base_delay, min(delay, config.max_delay))

    return delay


def _should_retry(config: RetryConfig, error: Exception) -> bool:
    """Determine if should retry"""
    if type(error) in config.non_retryable_exceptions:
        return False

    if config.retry_condition:
        return config.retry_condition(error)

    for exc_type in config.retryable_exceptions:
        if isinstance(error, exc_type):
            return True

    return False


def retry(
    func: Callable[..., Any],
    *args: Any,
    config: Optional[RetryConfig] = None,
    operation: Optional[str] = None,
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
        Exception: Last error after exhausting retries
    """
    config = config or RetryConfig()
    operation_name = operation or func.__name__
    last_error: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            if config.timeout:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    try:
                        result = future.result(timeout=config.timeout)
                        return result
                    except FuturesTimeoutError:
                        raise TimeoutError(f"Operation {operation_name} timed out after {config.timeout}s")
            else:
                return func(*args, **kwargs)
        except Exception as e:
            last_error = e

            if attempt == config.max_retries or not _should_retry(config, e):
                raise

            delay = _calculate_delay(config, attempt)

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
                f"Retry {operation_name} attempt {attempt + 1}/{config.max_retries}, "
                f"waiting {delay:.2f}s, error: {e}"
            )

            time.sleep(delay)

            if config.after_retry:
                config.after_retry(context, e)

    raise last_error


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    config: Optional[RetryConfig] = None,
    operation: Optional[str] = None,
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
        Exception: Last error after exhausting retries
    """
    config = config or RetryConfig()
    operation_name = operation or func.__name__
    last_error: Optional[Exception] = None

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
                    raise TimeoutError(
                        f"Operation {operation_name} timed out after {config.timeout}s"
                    )
            else:
                return await func(*args, **kwargs)
        except Exception as e:
            last_error = e

            if attempt == config.max_retries or not _should_retry(config, e):
                raise

            delay = _calculate_delay(config, attempt)

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
                f"Retry {operation_name} attempt {attempt + 1}/{config.max_retries}, "
                f"waiting {delay:.2f}s, error: {e}"
            )

            await asyncio.sleep(delay)

            if config.after_retry:
                if asyncio.iscoroutinefunction(config.after_retry):
                    await config.after_retry(context, e)
                else:
                    config.after_retry(context, e)

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
