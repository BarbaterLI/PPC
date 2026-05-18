"""Core abstract base classes.

Defines unified engine base class interface
with consistent lifecycle management, statistics tracking,
and async context manager support.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


@dataclass
class EngineStats:
    """Engine and executor statistics."""

    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    items_processed: int = 0
    items_failed: int = 0
    items_cancelled: int = 0
    bytes_processed: int = 0
    peak_memory_mb: float = 0.0
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to a serializable dictionary."""
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "items_processed": self.items_processed,
            "items_failed": self.items_failed,
            "items_cancelled": self.items_cancelled,
            "bytes_processed": self.bytes_processed,
            "peak_memory_mb": self.peak_memory_mb,
            "custom_metrics": self.custom_metrics,
        }


class BaseEngine(ABC, Generic[InputType, OutputType]):
    """Unified engine abstract base class.

    Provides lifecycle management (initialize/cleanup), statistics
    tracking, and async context manager support.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._stats = EngineStats()
        self._start_time: float | None = None

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the engine."""
        self._initialized = True
        self._stats.started_at = datetime.now(UTC)
        self._start_time = time.time()

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up engine resources."""
        if self._start_time is not None:
            self._stats.duration_seconds = time.time() - self._start_time
        self._stats.completed_at = datetime.now(UTC)
        self._initialized = False

    @abstractmethod
    async def process(self, input_data: InputType, /, **kwargs: Any) -> OutputType:
        """Process input data and return the result."""

    def get_stats(self) -> EngineStats:
        """Return current engine statistics."""
        if self._start_time is not None and self._stats.completed_at is None:
            self._stats.duration_seconds = time.time() - self._start_time
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics while preserving initialization state."""
        self._stats = EngineStats()
        self._start_time = time.time() if self._initialized else None
        if self._initialized:
            self._stats.started_at = datetime.now(UTC)

    def is_initialized(self) -> bool:
        """Check whether the engine has been initialized."""
        return self._initialized

    def _check_initialized(self) -> None:
        """Raise RuntimeError if the engine is not initialized."""
        if not self._initialized:
            raise RuntimeError(f"Engine {self.__class__.__name__} is not initialized")

    async def __aenter__(self) -> BaseEngine[InputType, OutputType]:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()
