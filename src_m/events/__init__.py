"""Event system module

Provides event bus, event type definitions, and event handler interfaces.
"""

from .event_bus import (
    EventBus,
    EventBusStats,
    Event,
    EventPriority,
    EventHandler,
    EventHandlerInfo,
    TaskStartedEvent,
    TaskProgressEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    RetryEvent,
    CircuitBreakerEvent,
    FormatConversionEvent,
    AudioProcessedEvent,
    ScheduledTaskEvent,
    WebhookEvent,
    get_event_bus,
    reset_event_bus,
)

__all__ = [
    "EventBus",
    "EventBusStats",
    "Event",
    "EventPriority",
    "EventHandler",
    "EventHandlerInfo",
    "TaskStartedEvent",
    "TaskProgressEvent",
    "TaskCompletedEvent",
    "TaskFailedEvent",
    "RetryEvent",
    "CircuitBreakerEvent",
    "FormatConversionEvent",
    "AudioProcessedEvent",
    "ScheduledTaskEvent",
    "WebhookEvent",
    "get_event_bus",
    "reset_event_bus",
]
