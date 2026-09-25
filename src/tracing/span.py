"""Span tracking unit"""

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """Span tracking unit

    Records start, end, tags, and logs for individual operations.
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)
    status: str = "OK"

    def set_tag(self, key: str, value: Any) -> None:
        """Set tag

        Args:
            key: Tag key
            value: Tag value
        """
        self.tags[key] = value

    def log(self, message: str, **kwargs: Any) -> None:
        """Record log entry

        Args:
            message: Log message
            **kwargs: Additional log fields
        """
        log_entry: dict[str, Any] = {
            "timestamp": time.time(),
            "message": message,
        }
        log_entry.update(kwargs)
        self.logs.append(log_entry)

    def set_error(self, error: Exception) -> None:
        """Set error status

        Args:
            error: Exception instance
        """
        self.status = "ERROR"
        self.set_tag("error", True)
        self.set_tag("error.type", type(error).__name__)
        self.set_tag("error.message", str(error))

    def finish(self) -> None:
        """End Span and calculate duration"""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration_ms = (self.end_time - self.start_time) * 1000

    def is_finished(self) -> bool:
        """Check if Span has ended

        Returns:
            Whether ended
        """
        return self.end_time is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary

        Returns:
            Dictionary containing Span information
        """
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status,
        }

    def to_json(self) -> str:
        """Convert to JSON string

        Returns:
            JSON formatted string
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)
