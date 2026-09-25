"""Event bus system

Supports synchronous and asynchronous event distribution, subscription/publishing mechanism,
event filters, and handler priority.
"""

import asyncio
import atexit
import logging
import threading
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    TypeVar,
)

logger = logging.getLogger(__name__)

E = TypeVar("E", bound="Event")


class EventPriority(Enum):
    """Event handler priority"""

    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


@dataclass
class Event:
    """Base event class"""

    timestamp: datetime = field(default_factory=datetime.now)
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def event_type(self) -> str:
        """Get event type name"""
        return self.__class__.__name__

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        data: dict[str, Any] = {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
        }
        for f in fields(type(self)):
            if f.name in ("timestamp", "source", "metadata"):
                continue
            value = getattr(self, f.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            data[f.name] = value
        return data


@dataclass
class TaskStartedEvent(Event):
    """Task started event"""

    task_id: str = ""
    task_name: str = ""
    task_type: str = ""
    input_path: str | None = None
    output_path: str | None = None


@dataclass
class TaskProgressEvent(Event):
    """Task progress event"""

    task_id: str = ""
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    message: str = ""
    bytes_processed: int = 0
    total_bytes: int = 0


@dataclass
class TaskCompletedEvent(Event):
    """Task completed event"""

    task_id: str = ""
    task_name: str = ""
    success: bool = True
    output_path: str | None = None
    duration_seconds: float = 0.0
    output_size: int = 0


@dataclass
class TaskFailedEvent(Event):
    """Task failed event"""

    task_id: str = ""
    task_name: str = ""
    error: str = ""
    error_code: str | None = None
    attempt: int = 0
    max_attempts: int = 0
    recoverable: bool = False


@dataclass
class RetryEvent(Event):
    """Retry event"""

    operation: str = ""
    attempt: int = 0
    max_attempts: int = 0
    delay_seconds: float = 0.0
    error: str = ""
    will_retry: bool = True


@dataclass
class CircuitBreakerEvent(Event):
    """Circuit breaker event"""

    breaker_name: str = ""
    old_state: str = ""
    new_state: str = ""
    failure_count: int = 0
    success_count: int = 0
    reason: str = ""


@dataclass
class FormatConversionEvent(Event):
    """Format conversion event"""

    source_format: str = ""
    target_format: str = ""
    file_path: str | None = None
    success: bool = True


@dataclass
class AudioProcessedEvent(Event):
    """Audio post-processing event"""

    file_path: str | None = None
    effect_name: str = ""
    duration: float = 0.0
    input_size: int = 0
    output_size: int = 0


@dataclass
class ScheduledTaskEvent(Event):
    """Scheduled task event"""

    task_id: str = ""
    task_type: str = ""
    scheduled_time: datetime | None = None
    action: str = ""


@dataclass
class WebhookEvent(Event):
    """Webhook callback event"""

    url: str = ""
    event_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    response_code: int | None = None
    response_body: str | None = None
    error: str | None = None


@dataclass
class ExtensionEvent(Event):
    """扩展事件"""

    extension_name: str = ""
    action: str = ""
    data: dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[E], None] | Callable[[E], Coroutine[Any, Any, None]]


@dataclass(eq=False)
class EventHandlerInfo:
    """Event handler information"""

    handler: Callable[..., Any]
    event_type: type[Event]
    priority: EventPriority = EventPriority.NORMAL
    filter_func: Callable[[Any], bool] | None = None
    is_async: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_async = asyncio.iscoroutinefunction(self.handler)

    def should_handle(self, event: Event) -> bool:
        """Check if this handler should process the event"""
        if not isinstance(event, self.event_type):
            return False
        return not (self.filter_func and not self.filter_func(event))


@dataclass
class EventBusStats:
    """Event bus statistics"""

    events_published: int = 0
    events_handled: int = 0
    handler_errors: int = 0
    handler_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "events_published": self.events_published,
            "events_handled": self.events_handled,
            "handler_errors": self.handler_errors,
            "handler_count": self.handler_count,
        }


class EventBus:
    """Event bus
    Supports synchronous and asynchronous event distribution, event filtering,
    handler priority, and thread safety.
    """

    def __init__(self):
        self._handlers: dict[type[Event], list[EventHandlerInfo]] = {}
        self._global_handlers: list[EventHandlerInfo] = []
        self._lock = threading.RLock()
        self._event_history: list[Event] = []
        self._max_history_size: int = 1000
        self._stats: EventBusStats = EventBusStats()
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_loop_thread: threading.Thread | None = None
        self._async_loop_ready = threading.Event()
        self._atexit_registered: bool = False

    def start(self) -> "EventBus":
        """Explicitly start the event bus lifecycle.

        Starts the background async loop (if not already running) and registers
        an atexit hook so the loop is shut down cleanly at interpreter exit.
        Idempotent and thread-safe.

        Returns:
            self, for chaining.
        """
        self._ensure_async_loop()
        self._ensure_atexit_registered()
        return self

    def _ensure_atexit_registered(self) -> None:
        """Register the atexit shutdown hook exactly once."""
        if self._atexit_registered:
            return
        with self._lock:
            if not self._atexit_registered:
                atexit.register(self.shutdown)
                self._atexit_registered = True

    def _ensure_async_loop(self) -> None:
        """Lazily start the background async loop on first use.

        Called on the first ``publish_async`` (or when a synchronous publish
        needs to invoke an async handler), so constructing the bus has no
        thread-creation side effects.
        """
        if self._async_loop_thread is not None and self._async_loop_thread.is_alive():
            return
        with self._lock:
            if self._async_loop_thread is not None and self._async_loop_thread.is_alive():
                return
            self._start_async_loop()

    def _start_async_loop(self):
        self._async_loop = asyncio.new_event_loop()
        self._async_loop_ready.clear()
        self._async_loop_thread = threading.Thread(target=self._run_async_loop, daemon=True, name="EventBusAsyncLoop")
        self._async_loop_thread.start()
        if not self._async_loop_ready.wait(timeout=5):
            logger.warning("EventBus async loop failed to start")

    def _run_async_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop_ready.set()
        self._async_loop.run_forever()

    def shutdown(self):
        """Shut down the background async loop and unregister the atexit hook.

        Safe to call multiple times.
        """
        with self._lock:
            if self._atexit_registered:
                atexit.unregister(self.shutdown)
                self._atexit_registered = False
        if self._async_loop and not self._async_loop.is_closed():
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        if self._async_loop_thread and self._async_loop_thread.is_alive():
            self._async_loop_thread.join(timeout=5)
        if self._async_loop and not self._async_loop.is_closed():
            self._async_loop.close()

    def subscribe(
        self,
        event_type: type[E],
        handler: EventHandler[E],
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Callable[[E], bool] | None = None,
    ) -> Callable[[], None]:
        """Subscribe to an event

        Args:
            event_type: Event type
            handler: Event handler (sync or async)
            priority: Handler priority
            filter_func: Event filter function

        Returns:
            Unsubscribe function
        """
        handler_info = EventHandlerInfo(
            handler=handler,
            event_type=event_type,
            priority=priority,
            filter_func=filter_func,
        )

        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler_info)
            self._handlers[event_type].sort(key=lambda h: h.priority.value)
            self._stats.handler_count += 1

        logger.debug(
            f"Subscribed to event {event_type.__name__}, handler: {handler.__name__}, priority: {priority.name}"
        )

        def unsubscribe():
            self.unsubscribe(event_type, handler)

        return unsubscribe

    def subscribe_global(
        self,
        handler: EventHandler[Event],
        priority: EventPriority = EventPriority.LOWEST,
        filter_func: Callable[[Event], bool] | None = None,
    ) -> Callable[[], None]:
        """Subscribe to all events

        Args:
            handler: Event handler
            priority: Handler priority
            filter_func: Event filter function

        Returns:
            Unsubscribe function
        """
        handler_info = EventHandlerInfo(
            handler=handler,
            event_type=Event,
            priority=priority,
            filter_func=filter_func,
        )

        with self._lock:
            self._global_handlers.append(handler_info)
            self._global_handlers.sort(key=lambda h: h.priority.value)
            self._stats.handler_count += 1

        logger.debug(f"Subscribed to global events, handler: {handler.__name__}")

        def unsubscribe():
            with self._lock:
                if handler_info in self._global_handlers:
                    self._global_handlers.remove(handler_info)
                    self._stats.handler_count -= 1

        return unsubscribe

    def unsubscribe(
        self,
        event_type: type[E],
        handler: EventHandler[E],
    ) -> bool:
        """Unsubscribe from an event

        Args:
            event_type: Event type
            handler: Event handler

        Returns:
            True if successfully unsubscribed
        """
        with self._lock:
            if event_type not in self._handlers:
                return False

            for info in self._handlers[event_type]:
                if info.handler == handler:
                    self._handlers[event_type].remove(info)
                    self._stats.handler_count -= 1
                    logger.debug(f"Unsubscribed from event {event_type.__name__}, handler: {handler.__name__}")
                    return True

        return False

    def _prepare_dispatch(self, event: Event) -> list[EventHandlerInfo]:
        """Record the event, update stats, and collect matching handlers"""
        self._record_event(event)
        with self._lock:
            self._stats.events_published += 1

        handlers_to_call: list[EventHandlerInfo] = []

        with self._lock:
            event_type = type(event)
            if event_type in self._handlers:
                handlers_to_call.extend(h for h in self._handlers[event_type] if h.should_handle(event))
            handlers_to_call.extend(h for h in self._global_handlers if h.should_handle(event))

        return handlers_to_call

    def publish(self, event: Event, fire_and_forget: bool = False) -> None:
        """Publish event (synchronous)

        Args:
            event: Event instance
            fire_and_forget: If True, async handlers are scheduled on the background
                loop without waiting for completion. This prevents slow async handlers
                from blocking the publishing thread, at the cost of unobservable
                handler errors. Defaults to False (blocking, backward compatible).
        """
        for handler_info in self._prepare_dispatch(event):
            self._invoke_handler_sync(handler_info, event, fire_and_forget=fire_and_forget)

    async def publish_async(self, event: Event) -> None:
        """Publish event (asynchronous)

        Lazily starts the background async loop on first use (thread-safe,
        idempotent) so constructing the bus has no thread side effects.

        Args:
            event: Event instance
        """
        self._ensure_async_loop()
        for handler_info in self._prepare_dispatch(event):
            await self._invoke_handler_async(handler_info, event)

    def _invoke_handler_sync(
        self, handler_info: EventHandlerInfo, event: Event, fire_and_forget: bool = False
    ) -> None:
        """Invoke handler synchronously

        Args:
            handler_info: Handler metadata
            event: Event instance
            fire_and_forget: If True, schedule async handlers without waiting for
                completion. Has no effect on synchronous handlers.
        """
        try:
            if handler_info.is_async:
                self._ensure_async_loop()
                if self._async_loop is not None and not self._async_loop.is_closed():
                    future = asyncio.run_coroutine_threadsafe(handler_info.handler(event), self._async_loop)
                    if not fire_and_forget:
                        future.result(timeout=60)
                else:
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(handler_info.handler(event))
                    finally:
                        loop.close()
            else:
                handler_info.handler(event)
            with self._lock:
                self._stats.events_handled += 1
        except Exception as e:
            with self._lock:
                self._stats.handler_errors += 1
            logger.warning(f"Event handler failed: {handler_info.handler.__name__}, error: {e}")

    async def _invoke_handler_async(self, handler_info: EventHandlerInfo, event: Event) -> None:
        """Invoke handler asynchronously"""
        try:
            if handler_info.is_async:
                await handler_info.handler(event)
            else:
                handler_info.handler(event)
            with self._lock:
                self._stats.events_handled += 1
        except Exception as e:
            with self._lock:
                self._stats.handler_errors += 1
            logger.warning(f"Event handler failed: {handler_info.handler.__name__}, error: {e}")

    def _record_event(self, event: Event) -> None:
        """Record event to history"""
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history_size:
                self._event_history = self._event_history[-self._max_history_size :]

    def get_history(
        self,
        event_type: type[Event] | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """Get event history

        Args:
            event_type: Filter by event type
            limit: Maximum number of events to return

        Returns:
            List of events
        """
        with self._lock:
            events = self._event_history
            if event_type:
                events = [e for e in events if isinstance(e, event_type)]
            return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history"""
        with self._lock:
            self._event_history.clear()

    def get_stats(self) -> EventBusStats:
        """Get statistics"""
        return self._stats

    def clear_all_handlers(self) -> None:
        """Clear all handlers"""
        with self._lock:
            self._handlers.clear()
            self._global_handlers.clear()
            self._stats.handler_count = 0
            logger.info("All event handlers cleared")


_event_bus_instance: EventBus | None = None
_event_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get global event bus singleton (lazily created, thread-safe).

    Creating the singleton starts no background thread and registers no
    atexit hook. The background async loop is started lazily on the first
    ``publish_async`` (or when a synchronous publish needs to invoke an
    async handler); for explicit lifecycle management call
    ``EventBus.start()`` / ``EventBus.shutdown()``.

    Returns:
        EventBus instance
    """
    global _event_bus_instance

    if _event_bus_instance is None:
        with _event_bus_lock:
            if _event_bus_instance is None:
                _event_bus_instance = EventBus()
                logger.info("Created global event bus instance")

    return _event_bus_instance


def reset_event_bus() -> None:
    """Reset global event bus (mainly for testing)"""
    global _event_bus_instance

    with _event_bus_lock:
        if _event_bus_instance is not None:
            _event_bus_instance.shutdown()
            _event_bus_instance.clear_all_handlers()
            _event_bus_instance.clear_history()
        _event_bus_instance = None
