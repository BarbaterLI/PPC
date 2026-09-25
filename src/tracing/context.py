"""Trace context propagation (IDs, baggage, header serialization)"""

import uuid
from dataclasses import dataclass, field
from typing import Any


def generate_trace_id() -> str:
    """Generate trace ID

    Returns:
        32-character hexadecimal string
    """
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate Span ID

    Returns:
        16-character hexadecimal string
    """
    return uuid.uuid4().hex[:16]


@dataclass
class TraceContext:
    """Trace context for propagating trace information across request chains"""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    baggage: dict[str, str] = field(default_factory=dict)
    sampled: bool = True

    def new_child_span(self) -> "TraceContext":
        """Create child Span context

        Returns:
            New TraceContext with current span_id as parent_span_id
        """
        return TraceContext(
            trace_id=self.trace_id,
            span_id=generate_span_id(),
            parent_span_id=self.span_id,
            baggage=self.baggage.copy(),
            sampled=self.sampled,
        )

    def to_headers(self) -> dict[str, str]:
        """Convert to HTTP headers format

        Returns:
            HTTP headers dictionary containing trace information
        """
        headers = {
            "x-trace-id": self.trace_id,
            "x-span-id": self.span_id,
            "x-sampled": "1" if self.sampled else "0",
        }
        if self.parent_span_id:
            headers["x-parent-span-id"] = self.parent_span_id
        for key, value in self.baggage.items():
            headers[f"x-baggage-{key}"] = value
        return headers

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> "TraceContext":
        """Create context from HTTP headers

        Args:
            headers: HTTP headers dictionary

        Returns:
            TraceContext instance
        """
        trace_id = headers.get("x-trace-id") or headers.get("X-Trace-Id", "")
        span_id = headers.get("x-span-id") or headers.get("X-Span-Id", "")

        if not trace_id:
            trace_id = generate_trace_id()
        if not span_id:
            span_id = generate_span_id()

        parent_span_id = headers.get("x-parent-span-id") or headers.get("X-Parent-Span-Id")
        sampled = (headers.get("x-sampled") or headers.get("X-Sampled", "1")) == "1"

        baggage: dict[str, str] = {}
        for key, value in headers.items():
            key_lower = key.lower()
            if key_lower.startswith("x-baggage-"):
                baggage_key = key_lower[10:]
                baggage[baggage_key] = value

        return cls(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage,
            sampled=sampled,
        )

    @classmethod
    def new_root(cls, sampled: bool = True) -> "TraceContext":
        """Create root context

        Args:
            sampled: Whether to sample

        Returns:
            New root TraceContext instance
        """
        return cls(
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
            sampled=sampled,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary

        Returns:
            Dictionary containing context information
        """
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage,
            "sampled": self.sampled,
        }
