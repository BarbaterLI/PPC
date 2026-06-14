"""Event bus system

Supports synchronous and asynchronous event distribution, subscription/publishing mechanism,
event filters, and handler priority.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    Coroutine,
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
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_type(self) -> str:
        """Get event type name"""
        return self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class TaskStartedEvent(Event):
    """Task started event"""
    task_id: str = ""
    task_name: str = ""
    task_type: str = ""
    input_path: Optional[str] = None
    output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_type": self.task_type,
            "input_path": self.input_path,
            "output_path": self.output_path,
        })
        return data


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

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "task_id": self.task_id,
            "progress": self.progress,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "message": self.message,
            "bytes_processed": self.bytes_processed,
            "total_bytes": self.total_bytes,
        })
        return data


@dataclass
class TaskCompletedEvent(Event):
    """Task completed event"""
    task_id: str = ""
    task_name: str = ""
    success: bool = True
    output_path: Optional[str] = None
    duration_seconds: float = 0.0
    output_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "task_id": self.task_id,
            "task_name": self.task_name,
            "success": self.success,
            "output_path": self.output_path,
            "duration_seconds": self.duration_seconds,
            "output_size": self.output_size,
        })
        return data


@dataclass
class TaskFailedEvent(Event):
    """Task failed event"""
    task_id: str = ""
    task_name: str = ""
    error: str = ""
    error_code: Optional[str] = None
    attempt: int = 0
    max_attempts: int = 0
    recoverable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "task_id": self.task_id,
            "task_name": self.task_name,
            "error": self.error,
            "error_code": self.error_code,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "recoverable": self.recoverable,
        })
        return data


@dataclass
class RetryEvent(Event):
    """Retry event"""
    operation: str = ""
    attempt: int = 0
    max_attempts: int = 0
    delay_seconds: float = 0.0
    error: str = ""
    will_retry: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "operation": self.operation,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "delay_seconds": self.delay_seconds,
            "error": self.error,
            "will_retry": self.will_retry,
        })
        return data


@dataclass
class CircuitBreakerEvent(Event):
    """Circuit breaker event"""
    breaker_name: str = ""
    old_state: str = ""
    new_state: str = ""
    failure_count: int = 0
    success_count: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "breaker_name": self.breaker_name,
            "old_state": self.old_state,
            "new_state": self.new_state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "reason": self.reason,
        })
        return data


@dataclass
class FormatConversionEvent(Event):
    """Format conversion event"""
    source_format: str = ""
    target_format: str = ""
    file_path: Optional[str] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "source_format": self.source_format,
            "target_format": self.target_format,
            "file_path": self.file_path,
            "success": self.success,
        })
        return data


@dataclass
class AudioProcessedEvent(Event):
    """Audio post-processing event"""
    file_path: Optional[str] = None
    effect_name: str = ""
    duration: float = 0.0
    input_size: int = 0
    output_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "file_path": self.file_path,
            "effect_name": self.effect_name,
            "duration": self.duration,
            "input_size": self.input_size,
            "output_size": self.output_size,
        })
        return data


@dataclass
class ScheduledTaskEvent(Event):
    """Scheduled task event"""
    task_id: str = ""
    task_type: str = ""
    scheduled_time: Optional[datetime] = None
    action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "task_id": self.task_id,
            "task_type": self.task_type,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "action": self.action,
        })
        return data


@dataclass
class WebhookEvent(Event):
    """Webhook callback event"""
    url: str = ""
    event_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "url": self.url,
            "event_type": self.event_type,
            "payload": self.payload,
            "status": self.status,
            "response_code": self.response_code,
            "response_body": self.response_body,
            "error": self.error,
        })
        return data


@dataclass
class ExtensionEvent(Event):
    """扩展事件"""
    extension_name: str = ""
    action: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "extension_name": self.extension_name,
            "action": self.action,
            "data": self.data,
        })
        return data


@dataclass
class PipelineStartedEvent(Event):
    """Pipeline started event"""
    pipeline_name: str = ""
    total_steps: int = 0
    variables: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "pipeline_name": self.pipeline_name,
            "total_steps": self.total_steps,
            "variables": self.variables,
        })
        return data


@dataclass
class PipelineStepStartedEvent(Event):
    """Pipeline step started event"""
    pipeline_name: str = ""
    step_name: str = ""
    step_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "pipeline_name": self.pipeline_name,
            "step_name": self.step_name,
            "step_type": self.step_type,
        })
        return data


@dataclass
class PipelineStepCompletedEvent(Event):
    """Pipeline step completed event"""
    pipeline_name: str = ""
    step_name: str = ""
    duration_seconds: float = 0.0
    output_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "pipeline_name": self.pipeline_name,
            "step_name": self.step_name,
            "duration_seconds": self.duration_seconds,
            "output_data": self.output_data,
        })
        return data


@dataclass
class PipelineStepFailedEvent(Event):
    """Pipeline step failed event"""
    pipeline_name: str = ""
    step_name: str = ""
    error: str = ""
    attempt: int = 0
    will_retry: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "pipeline_name": self.pipeline_name,
            "step_name": self.step_name,
            "error": self.error,
            "attempt": self.attempt,
            "will_retry": self.will_retry,
        })
        return data


@dataclass
class PipelineCompletedEvent(Event):
    """Pipeline completed event"""
    pipeline_name: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "pipeline_name": self.pipeline_name,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "duration_seconds": self.duration_seconds,
        })
        return data


@dataclass
class PipelineFailedEvent(Event):
    """Pipeline failed event"""
    pipeline_name: str = ""
    failed_step: str = ""
    error: str = ""
    completed_steps: int = 0
    total_steps: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "pipeline_name": self.pipeline_name,
            "failed_step": self.failed_step,
            "error": self.error,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "duration_seconds": self.duration_seconds,
        })
        return data


EventHandler = Union[
    Callable[[E], None],
    Callable[[E], Coroutine[Any, Any, None]],
]


class EventHandlerInfo:
    """Event handler information"""

    def __init__(
        self,
        handler: EventHandler,
        event_type: Type[Event],
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Optional[Callable[[Event], bool]] = None,
    ):
        self.handler = handler
        self.event_type = event_type
        self.priority = priority
        self.filter_func = filter_func
        self.is_async = asyncio.iscoroutinefunction(handler)

    def should_handle(self, event: Event) -> bool:
        """Check if this handler should process the event"""
        if not isinstance(event, self.event_type):
            return False
        if self.filter_func and not self.filter_func(event):
            return False
        return True


@dataclass
class EventBusStats:
    """Event bus statistics"""
    events_published: int = 0
    events_handled: int = 0
    handler_errors: int = 0
    handler_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
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
        self._handlers: Dict[Type[Event], List[EventHandlerInfo]] = {}
        self._global_handlers: List[EventHandlerInfo] = []
        self._lock = threading.RLock()
        self._event_history: List[Event] = []
        self._max_history_size: int = 1000
        self._stats = EventBusStats()
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_loop_thread: Optional[threading.Thread] = None
        self._async_loop_ready = threading.Event()
        self._start_async_loop()

    def _start_async_loop(self):
        self._async_loop = asyncio.new_event_loop()
        self._async_loop_ready.clear()
        self._async_loop_thread = threading.Thread(
            target=self._run_async_loop, daemon=True, name="EventBusAsyncLoop"
        )
        self._async_loop_thread.start()
        if not self._async_loop_ready.wait(timeout=5):
            logger.warning("EventBus async loop failed to start")

    def _run_async_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop_ready.set()
        self._async_loop.run_forever()

    def shutdown(self):
        if self._async_loop and not self._async_loop.is_closed():
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        if self._async_loop_thread and self._async_loop_thread.is_alive():
            self._async_loop_thread.join(timeout=5)
        if self._async_loop and not self._async_loop.is_closed():
            self._async_loop.close()

    def subscribe(
        self,
        event_type: Type[E],
        handler: EventHandler[E],
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Optional[Callable[[E], bool]] = None,
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
            f"Subscribed to event {event_type.__name__}, "
            f"handler: {handler.__name__}, priority: {priority.name}"
        )

        def unsubscribe():
            self.unsubscribe(event_type, handler)

        return unsubscribe

    def subscribe_global(
        self,
        handler: EventHandler[Event],
        priority: EventPriority = EventPriority.LOWEST,
        filter_func: Optional[Callable[[Event], bool]] = None,
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
        event_type: Type[E],
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
                    logger.debug(
                        f"Unsubscribed from event {event_type.__name__}, "
                        f"handler: {handler.__name__}"
                    )
                    return True

        return False

    def publish(self, event: Event) -> None:
        """Publish event (synchronous)

        Args:
            event: Event instance
        """
        self._record_event(event)
        with self._lock:
            self._stats.events_published += 1

        handlers_to_call: List[EventHandlerInfo] = []

        with self._lock:
            event_type = type(event)
            if event_type in self._handlers:
                handlers_to_call.extend(
                    h for h in self._handlers[event_type] if h.should_handle(event)
                )
            handlers_to_call.extend(
                h for h in self._global_handlers if h.should_handle(event)
            )

        for handler_info in handlers_to_call:
            self._invoke_handler_sync(handler_info, event)

    async def publish_async(self, event: Event) -> None:
        """Publish event (asynchronous)

        Args:
            event: Event instance
        """
        self._record_event(event)
        with self._lock:
            self._stats.events_published += 1

        handlers_to_call: List[EventHandlerInfo] = []

        with self._lock:
            event_type = type(event)
            if event_type in self._handlers:
                handlers_to_call.extend(
                    h for h in self._handlers[event_type] if h.should_handle(event)
                )
            handlers_to_call.extend(
                h for h in self._global_handlers if h.should_handle(event)
            )

        for handler_info in handlers_to_call:
            await self._invoke_handler_async(handler_info, event)

    def _invoke_handler_sync(self, handler_info: EventHandlerInfo, event: Event) -> None:
        """Invoke handler synchronously"""
        try:
            if handler_info.is_async:
                if self._async_loop is not None and not self._async_loop.is_closed():
                    future = asyncio.run_coroutine_threadsafe(
                        handler_info.handler(event), self._async_loop
                    )
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
            logger.warning(
                f"Event handler failed: {handler_info.handler.__name__}, error: {e}"
            )

    async def _invoke_handler_async(
        self, handler_info: EventHandlerInfo, event: Event
    ) -> None:
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
            logger.warning(
                f"Event handler failed: {handler_info.handler.__name__}, error: {e}"
            )

    def _record_event(self, event: Event) -> None:
        """Record event to history"""
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history_size:
                self._event_history = self._event_history[-self._max_history_size:]

    def get_history(
        self,
        event_type: Optional[Type[Event]] = None,
        limit: int = 100,
    ) -> List[Event]:
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


_event_bus_instance: Optional[EventBus] = None
_event_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get global event bus singleton

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
