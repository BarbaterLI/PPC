import json
import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Full, Queue
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    task_id: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: float = 0.0
    progress_message: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _subscribers: list[Queue] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "task_id": self.task_id,
            "type": self.task_type,
            "task_type": self.task_type,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "params": self.params,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "progress_message": self.progress_message,
        }


class TaskHandler(ABC):
    """任务处理器显式基类：子类声明 ``task_type`` 并实现 :meth:`run`。"""

    task_type: ClassVar[str]

    @abstractmethod
    def run(self, task_id: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """执行任务；返回结果 dict（可为空/None）。"""
        pass


class TaskHandlerRegistry:
    """显式 handler 注册表：字符串 task_type → :class:`TaskHandler` 实例。"""

    def __init__(self) -> None:
        self._handlers: dict[str, TaskHandler] = {}

    def register(self, handler: TaskHandler) -> None:
        self._handlers[handler.task_type] = handler

    def get(self, task_type: str) -> TaskHandler | None:
        return self._handlers.get(task_type)

    def registered_types(self) -> list[str]:
        return list(self._handlers.keys())


_Notify = tuple[list[Queue], str, dict[str, Any]]


class TaskStore:
    """任务状态存储：任务表 / 锁 / 终态历史清理。

    所有状态转换方法在锁内同步快照订阅者与通知 payload，
    返回 ``(subscribers, event, data)`` 供 :class:`EventBroker` 在锁外推送。
    """

    MAX_HISTORY = 100

    def __init__(self) -> None:
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.Lock()
        self._completed_order: deque[str] = deque()

    def add(self, task: TaskInfo) -> None:
        with self._lock:
            self._tasks[task.task_id] = task
            self._cleanup_oldest()

    def get(self, task_id: str) -> TaskInfo | None:
        with self._lock:
            return self._tasks.get(task_id)

    def all(self) -> list[TaskInfo]:
        with self._lock:
            return list(self._tasks.values())

    def _cleanup_oldest(self):
        """Evict oldest completed tasks beyond MAX_HISTORY.

        Must be called under ``self._lock``. Uses ``_completed_order`` deque
        to avoid O(N) full-table scans on every call.
        """
        while len(self._completed_order) > self.MAX_HISTORY:
            old_id = self._completed_order.popleft()
            old_task = self._tasks.get(old_id)
            if old_task is not None and old_task.status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ):
                del self._tasks[old_id]

    def mark_running(self, task_id: str) -> bool:
        """将任务标记为 RUNNING；已处于终态（如已取消）的任务拒绝并返回 False。

        修复：排队中（PENDING）被 cancel 的任务，其工作线程可能稍后才启动，
        旧实现无条件把状态覆盖回 RUNNING 并返回 True → ``_run_task`` 照常执行
        ``handler.run``，cancel 形同虚设（白跑整个 handler、状态来回跳）。
        ``_run_task`` 在 ``if not mark_running: return`` 处即跳过。
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status in (TaskStatus.CANCELLED, TaskStatus.COMPLETED, TaskStatus.FAILED):
                return False
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            return True

    def set_progress(self, task_id: str, progress: float, message: str = "") -> _Notify | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.progress = progress
            task.progress_message = message
            # Build snapshot under lock: subscribers list + payload data.
            # Avoids passing the live TaskInfo to _notify_subscribers,
            # which could be mutated between lock release and notify.
            subscribers = list(task._subscribers)
            snapshot = {
                "task_id": task_id,
                "progress": progress,
                "message": message,
            }
            return _Notify((subscribers, "progress", snapshot))

    def complete(self, task_id: str, result: dict[str, Any]) -> _Notify | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = TaskStatus.COMPLETED
            task.progress = 100.0
            task.result = result
            task.completed_at = time.time()
            self._completed_order.append(task_id)
            self._cleanup_oldest()
            # Snapshot subscribers under lock (terminal COMPLETED status
            # makes the race benign, but snapshot for consistency).
            subscribers = list(task._subscribers)
            return _Notify((subscribers, "complete", result))

    def fail(self, task_id: str, error: str) -> _Notify | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = time.time()
            self._completed_order.append(task_id)
            self._cleanup_oldest()
            # Snapshot subscribers + payload under lock.
            subscribers = list(task._subscribers)
            data = {"message": error}
            return _Notify((subscribers, "error", data))

    def cancel(self, task_id: str) -> _Notify | bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
                return False
            task._cancel_event.set()
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            self._completed_order.append(task_id)
            self._cleanup_oldest()
            # Snapshot subscribers + payload under lock (terminal CANCELLED
            # status makes the live-object race benign, but snapshot for
            # consistency with other notify paths).
            subscribers = list(task._subscribers)
            data = {"message": "Task cancelled"}
            return _Notify((subscribers, "error", data))

    def mark_cancelled(self, task_id: str) -> _Notify | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            # Snapshot subscribers + payload under lock.
            subscribers = list(task._subscribers)
            data = {"message": "Task cancelled"}
            return _Notify((subscribers, "error", data))


class EventBroker:
    """进度订阅 / 事件推送：管理任务订阅者队列与 SSE 事件流。"""

    # 订阅队列上限：慢消费者（前端断开前堆积）不会无限占用内存。
    SUBSCRIBER_QUEUE_MAXSIZE = 1000

    def __init__(self, store: TaskStore) -> None:
        self._store = store

    def subscribe(self, task_id: str) -> Queue | None:
        with self._store._lock:
            task = self._store._tasks.get(task_id)
            if task is None:
                return None
            q: Queue = Queue(maxsize=self.SUBSCRIBER_QUEUE_MAXSIZE)
            task._subscribers.append(q)
            return q

    def unsubscribe(self, task_id: str, q: Queue | None) -> None:
        """注销订阅者（幂等）。

        C2（P1）：触发条件——SSE 流结束（任务终态）或客户端断连后 EventSource
        自动重连。修复前 ``subscribe()`` 只追加、全项目没有注销路径，Queue 与
        积压消息随重连无限累积（内存泄漏），``publish`` 还要遍历所有死队列。
        用 ``list.remove`` 定位（O(订阅者数) 的一次性操作），避免每次 publish
        全表扫描；任务已被历史清理或重复注销都静默返回。
        """
        if q is None:
            return
        with self._store._lock:
            task = self._store._tasks.get(task_id)
            if task is None:
                return
            try:
                task._subscribers.remove(q)
            except ValueError:
                return

    def _enqueue(self, q: Queue, item: Any) -> None:
        """有界入队：满则丢弃最旧一条，保证最新状态与终止信号不被饿死。

        进度消息具有「最新优先」语义（旧进度无意义），故丢弃最旧；
        终止信号 ``None`` 与终态事件同理强制入队，避免 SSE 流永不结束。
        """
        try:
            q.put_nowait(item)
        except Full:
            try:
                q.get_nowait()
            except Empty:
                pass
            try:
                q.put_nowait(item)
            except Full:
                pass

    def publish(self, subscribers: list[Queue], event: str, data: dict[str, Any]) -> None:
        message = json.dumps({"event": event, "data": data})
        # subscribers list is already snapshotted by the caller under the
        # manager lock; iterate the snapshot directly to avoid holding the
        # lock during subscriber callbacks.
        for q in subscribers:
            self._enqueue(q, message)
        if event in ("complete", "error"):
            for q in subscribers:
                self._enqueue(q, None)


class TaskRunner:
    """执行与线程管理：为每个任务启动后台线程并驱动 handler 跑完。

    用有界并发信号量钳制同时运行的任务数：超限的任务直接失败（而非无限
    堆积线程/内存），同时保留 daemon 线程语义（进程可正常退出）。
    并发上限由环境变量 ``PPC_WEB_MAX_TASKS`` 控制，默认 8。
    """

    def __init__(
        self,
        store: TaskStore,
        broker: EventBroker,
        registry: TaskHandlerRegistry,
        max_concurrent_tasks: int | None = None,
    ) -> None:
        self._store = store
        self._broker = broker
        self._registry = registry
        self._max_concurrent_tasks = max_concurrent_tasks or _env_int("PPC_WEB_MAX_TASKS", 8)
        self._slots = threading.BoundedSemaphore(self._max_concurrent_tasks)

    def create_task(self, task_type: str, params: dict[str, Any] | None = None) -> str:
        task_id = uuid.uuid4().hex
        self._store.add(TaskInfo(task_id=task_id, task_type=task_type, params=params or {}))

        thread = threading.Thread(
            target=self._run_task_guarded,
            args=(task_id, task_type, params or {}),
            daemon=True,
        )
        thread.start()

        return task_id

    def _run_task_guarded(self, task_id: str, task_type: str, params: dict[str, Any]):
        """占用并发槽位执行任务；无空闲槽位时立即失败，避免线程无限堆积。"""
        if not self._slots.acquire(blocking=False):
            self._fail(task_id, "任务队列已满，请稍后重试 / task queue is full")
            return
        try:
            self._run_task(task_id, task_type, params)
        finally:
            self._slots.release()

    def _run_task(self, task_id: str, task_type: str, params: dict[str, Any]):
        handler = self._registry.get(task_type)
        if handler is None:
            self._fail(task_id, f"No handler registered for task type: {task_type}")
            return

        if not self._store.mark_running(task_id):
            return

        try:
            result = handler.run(task_id, params)
            task = self._store.get(task_id)
            if task is None:
                return
            if task._cancel_event.is_set():
                note = self._store.mark_cancelled(task_id)
                if note is not None:
                    self._broker.publish(*note)
            else:
                self._complete(task_id, result or {})
        except Exception as e:
            logger.exception("Task %s failed", task_id)
            self._fail(task_id, str(e))

    def _complete(self, task_id: str, result: dict[str, Any]) -> None:
        note = self._store.complete(task_id, result)
        if note is not None:
            self._broker.publish(*note)

    def _fail(self, task_id: str, error: str) -> None:
        note = self._store.fail(task_id, error)
        if note is not None:
            self._broker.publish(*note)


class TaskManager:
    """TaskStore / TaskRunner / EventBroker 的组合门面。

    由 web 应用工厂显式构造并注入各蓝图 / handler（非全局单例），
    对外保持原有方法名，内部全部委托给三个组件。
    """

    def __init__(
        self,
        store: TaskStore | None = None,
        broker: EventBroker | None = None,
        registry: TaskHandlerRegistry | None = None,
        runner: TaskRunner | None = None,
    ) -> None:
        self.store = store or TaskStore()
        self.broker = broker or EventBroker(self.store)
        self.registry = registry or TaskHandlerRegistry()
        self.runner = runner or TaskRunner(self.store, self.broker, self.registry)

    def register_handler(self, handler: TaskHandler) -> None:
        self.registry.register(handler)

    def create_task(self, task_type: str, params: dict[str, Any] | None = None) -> str:
        return self.runner.create_task(task_type, params)

    def get_task(self, task_id: str) -> TaskInfo | None:
        return self.store.get(task_id)

    def get_all_tasks(self) -> list[TaskInfo]:
        return self.store.all()

    def subscribe(self, task_id: str) -> Queue | None:
        return self.broker.subscribe(task_id)

    def unsubscribe(self, task_id: str, q: Queue | None) -> None:
        """注销 SSE 订阅者；SSE 生成器的 ``finally`` 中调用，避免重连累积。"""
        self.broker.unsubscribe(task_id, q)

    def is_cancelled(self, task_id: str) -> bool:
        task = self.store.get(task_id)
        if task is None:
            return True
        return task._cancel_event.is_set()

    def update_progress(self, task_id: str, progress: float, message: str = "") -> None:
        note = self.store.set_progress(task_id, progress, message)
        if note is not None:
            self.broker.publish(*note)

    def complete_task(self, task_id: str, result: dict[str, Any]) -> None:
        note = self.store.complete(task_id, result)
        if note is not None:
            self.broker.publish(*note)

    def fail_task(self, task_id: str, error: str) -> None:
        note = self.store.fail(task_id, error)
        if note is not None:
            self.broker.publish(*note)

    def cancel_task(self, task_id: str) -> bool:
        note = self.store.cancel(task_id)
        if note is False:
            return False
        if note is not None:
            self.broker.publish(*note)
        return True


__all__ = [
    "TaskStatus",
    "TaskInfo",
    "TaskHandler",
    "TaskHandlerRegistry",
    "TaskStore",
    "EventBroker",
    "TaskRunner",
    "TaskManager",
]
