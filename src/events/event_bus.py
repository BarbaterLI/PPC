"""事件总线系统
支持同步和异步事件分发、订阅/发布机制、事件过滤器和处理器优先级
"""

import asyncio
import logging
import threading
from abc import ABC, abstractmethod
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
    """事件处理器优先级"""
    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


@dataclass
class Event:
    """事件基类"""
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_type(self) -> str:
        """获取事件类型名称"""
        return self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class TaskStartedEvent(Event):
    """任务开始事件"""
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
    """任务进度事件"""
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
    """任务完成事件"""
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
    """任务失败事件"""
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
    """重试事件"""
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
    """熔断器事件"""
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


EventHandler = Union[
    Callable[[E], None],
    Callable[[E], Coroutine[Any, Any, None]],
]


class EventHandlerInfo:
    """事件处理器信息"""

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
        """检查是否应该处理该事件"""
        if not isinstance(event, self.event_type):
            return False
        if self.filter_func and not self.filter_func(event):
            return False
        return True


class EventBus:
    """事件总线
    支持同步和异步事件分发、事件过滤、处理器优先级、线程安全
    """

    def __init__(self):
        self._handlers: Dict[Type[Event], List[EventHandlerInfo]] = {}
        self._global_handlers: List[EventHandlerInfo] = []
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        self._event_history: List[Event] = []
        self._max_history_size: int = 1000
        self._stats = EventBusStats()

    def subscribe(
        self,
        event_type: Type[E],
        handler: EventHandler[E],
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Optional[Callable[[E], bool]] = None,
    ) -> Callable[[], None]:
        """订阅事件

        Args:
            event_type: 事件类型
            handler: 事件处理器（同步或异步）
            priority: 处理器优先级
            filter_func: 事件过滤器函数

        Returns:
            取消订阅的函数
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
            f"已订阅事件 {event_type.__name__}, 处理器: {handler.__name__}, 优先级: {priority.name}"
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
        """订阅所有事件

        Args:
            handler: 事件处理器
            priority: 处理器优先级
            filter_func: 事件过滤器函数

        Returns:
            取消订阅的函数
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
        logger.debug(f"已订阅全局事件, 处理器: {handler.__name__}")

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
        """取消订阅事件

        Args:
            event_type: 事件类型
            handler: 事件处理器

        Returns:
            是否成功取消订阅
        """
        with self._lock:
            if event_type not in self._handlers:
                return False

            for info in self._handlers[event_type]:
                if info.handler == handler:
                    self._handlers[event_type].remove(info)
                    self._stats.handler_count -= 1
                    logger.debug(
                        f"已取消订阅事件 {event_type.__name__}, 处理器: {handler.__name__}"
                    )
                    return True

        return False

    def publish(self, event: Event) -> None:
        """发布事件（同步）

        Args:
            event: 事件实例
        """
        self._record_event(event)
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
        """发布事件（异步）

        Args:
            event: 事件实例
        """
        self._record_event(event)
        self._stats.events_published += 1

        handlers_to_call: List[EventHandlerInfo] = []

        async with self._async_lock:
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
        """同步调用处理器"""
        try:
            if handler_info.is_async:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(handler_info.handler(event))
                else:
                    loop.run_until_complete(handler_info.handler(event))
            else:
                handler_info.handler(event)
            self._stats.events_handled += 1
        except Exception as e:
            self._stats.handler_errors += 1
            logger.warning(
                f"事件处理器执行失败: {handler_info.handler.__name__}, 错误: {e}"
            )

    async def _invoke_handler_async(
        self, handler_info: EventHandlerInfo, event: Event
    ) -> None:
        """异步调用处理器"""
        try:
            if handler_info.is_async:
                await handler_info.handler(event)
            else:
                handler_info.handler(event)
            self._stats.events_handled += 1
        except Exception as e:
            self._stats.handler_errors += 1
            logger.warning(
                f"事件处理器执行失败: {handler_info.handler.__name__}, 错误: {e}"
            )

    def _record_event(self, event: Event) -> None:
        """记录事件到历史"""
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history_size:
                self._event_history = self._event_history[-self._max_history_size :]

    def get_history(
        self,
        event_type: Optional[Type[Event]] = None,
        limit: int = 100,
    ) -> List[Event]:
        """获取事件历史

        Args:
            event_type: 过滤事件类型
            limit: 返回数量限制

        Returns:
            事件列表
        """
        with self._lock:
            events = self._event_history
            if event_type:
                events = [e for e in events if isinstance(e, event_type)]
            return events[-limit:]

    def clear_history(self) -> None:
        """清除事件历史"""
        with self._lock:
            self._event_history.clear()

    def get_stats(self) -> "EventBusStats":
        """获取统计信息"""
        return self._stats

    def clear_all_handlers(self) -> None:
        """清除所有处理器"""
        with self._lock:
            self._handlers.clear()
            self._global_handlers.clear()
            self._stats.handler_count = 0
            logger.info("已清除所有事件处理器")


@dataclass
class EventBusStats:
    """事件总线统计"""
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


_event_bus_instance: Optional[EventBus] = None
_event_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """获取全局事件总线单例

    Returns:
        EventBus 实例
    """
    global _event_bus_instance

    if _event_bus_instance is None:
        with _event_bus_lock:
            if _event_bus_instance is None:
                _event_bus_instance = EventBus()
                logger.info("已创建全局事件总线实例")

    return _event_bus_instance


def reset_event_bus() -> None:
    """重置全局事件总线（主要用于测试）"""
    global _event_bus_instance

    with _event_bus_lock:
        if _event_bus_instance is not None:
            _event_bus_instance.clear_all_handlers()
            _event_bus_instance.clear_history()
        _event_bus_instance = None
