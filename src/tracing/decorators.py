"""Tracing decorators and context managers (@traced, trace_span, trace_span_sync)"""

import asyncio
import functools
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from typing import Any, TypeVar, cast

from .core import Tracer, get_tracer

F = TypeVar("F", bound=Callable[..., Any])


def traced(name: str | None = None, tracer: Tracer | None = None) -> Callable[[F], F]:
    """Tracing decorator

    Args:
        name: Span name, defaults to function name
        tracer: Tracer instance, defaults to global tracer

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            target_tracer = tracer or get_tracer()
            with target_tracer.span(span_name):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            target_tracer = tracer or get_tracer()
            async with target_tracer.async_span(span_name):
                return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, wrapper)

    return decorator


@asynccontextmanager
async def trace_span(name: str, tracer: Tracer | None = None):
    """Async tracing context manager

    Args:
        name: Span name
        tracer: Tracer instance, defaults to global tracer

    Yields:
        Span instance
    """
    target_tracer = tracer or get_tracer()
    async with target_tracer.async_span(name) as span:
        yield span


@contextmanager
def trace_span_sync(name: str, tracer: Tracer | None = None):
    """Sync tracing context manager

    Args:
        name: Span name
        tracer: Tracer instance, defaults to global tracer

    Yields:
        Span instance
    """
    target_tracer = tracer or get_tracer()
    with target_tracer.span(name) as span:
        yield span
