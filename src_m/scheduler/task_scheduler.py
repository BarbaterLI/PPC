"""Task scheduler implementation

Provides priority scheduling, execution time tracking, and task lifecycle management.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """Task state enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority enumeration"""
    HIGHEST = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    LOWEST = 4


@dataclass
class TaskTimeoutError(Exception):
    """Task timeout error"""
    task_id: str = ""
    timeout: float = 0.0
    elapsed: float = 0.0

    def __str__(self) -> str:
        return f"Task {self.task_id} timed out after {self.elapsed:.2f}s (limit: {self.timeout:.2f}s)"


@dataclass
class CancelledTaskError(Exception):
    """Cancelled task error"""
    task_id: str = ""
    reason: str = ""

    def __str__(self) -> str:
        msg = f"Task {self.task_id} cancelled"
        return f"{msg} (reason: {self.reason})" if self.reason else msg


@dataclass
class ScheduledTask:
    """Scheduled task"""
    task_id: str
    name: str
    priority: TaskPriority = TaskPriority.MEDIUM
    state: TaskState = TaskState.PENDING
    timeout: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    execution_time: float = 0.0
    error: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_pending(self) -> bool:
        return self.state == TaskState.PENDING

    @property
    def is_running(self) -> bool:
        return self.state == TaskState.RUNNING

    @property
    def is_completed(self) -> bool:
        return self.state == TaskState.COMPLETED

    @property
    def is_failed(self) -> bool:
        return self.state == TaskState.FAILED

    @property
    def is_cancelled(self) -> bool:
        return self.state == TaskState.CANCELLED

    @property
    def wait_time(self) -> float:
        """Time waiting for execution"""
        if self.started_at:
            return self.started_at - self.created_at
        return time.time() - self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "priority": self.priority.name,
            "state": self.state.value,
            "timeout": self.timeout,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_time": self.execution_time,
            "error": self.error,
            "wait_time": self.wait_time,
            "tags": self.tags,
        }


class PriorityScheduler:
    """Priority task scheduler

    Manages task scheduling with priorities, timeouts, and lifecycle tracking.
    """

    def __init__(self, max_concurrent: int = 10):
        self._max_concurrent = max_concurrent
        self._tasks: Dict[str, ScheduledTask] = {}
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._task_counter = 0
        self._lock = asyncio.Lock()
        self._completion_events: Dict[str, asyncio.Event] = {}
        self._results: Dict[str, Any] = {}

    async def submit(
        self,
        name: str,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        priority: TaskPriority = TaskPriority.MEDIUM,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Submit a task for execution

        Args:
            name: Task name
            func: Async function to execute
            *args: Function arguments
            priority: Task priority
            timeout: Task timeout in seconds
            **kwargs: Function keyword arguments

        Returns:
            Task ID
        """
        async with self._lock:
            self._task_counter += 1
            task_id = f"task_{self._task_counter}_{int(time.time())}"

        task = ScheduledTask(
            task_id=task_id,
            name=name,
            priority=priority,
            timeout=timeout,
        )

        async with self._lock:
            self._tasks[task_id] = task
            self._completion_events[task_id] = asyncio.Event()

        await self._queue.put((priority.value, task_id, func, args, kwargs))

        if not self._running:
            await self.start()

        logger.debug(f"Task submitted: {task_id} ({name}), priority: {priority.name}")
        return task_id

    async def start(self) -> None:
        """Start the scheduler"""
        if self._running:
            return

        self._running = True
        for i in range(self._max_concurrent):
            task = asyncio.create_task(self._worker(i))
            self._worker_tasks.append(task)

        logger.info(f"Scheduler started with {self._max_concurrent} workers")

    async def stop(self, wait: bool = True) -> None:
        """Stop the scheduler

        Args:
            wait: Whether to wait for running tasks to complete
        """
        self._running = False

        if not wait:
            for task in self._worker_tasks:
                task.cancel()

        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        self._worker_tasks.clear()
        logger.info("Scheduler stopped")

    async def wait_for(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for a task to complete

        Args:
            task_id: Task ID
            timeout: Wait timeout

        Returns:
            Task result
        """
        event = self._completion_events.get(task_id)
        if not event:
            raise ValueError(f"Task not found: {task_id}")

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Wait for task {task_id} timed out")

        task = self._tasks.get(task_id)
        if task and task.is_failed:
            raise RuntimeError(f"Task {task_id} failed: {task.error}")

        if task and task.is_cancelled:
            raise CancelledTaskError(task_id=task_id)

        return self._results.get(task_id)

    async def cancel(self, task_id: str, reason: str = "") -> bool:
        """Cancel a task (including RUNNING tasks)

        Args:
            task_id: Task ID
            reason: Cancellation reason

        Returns:
            True if cancelled successfully
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            if task.is_pending or task.is_running:
                task.state = TaskState.CANCELLED
                task.completed_at = time.time()
                event = self._completion_events.get(task_id)
                if event:
                    event.set()
                logger.info(f"Task cancelled: {task_id}")
                return True

        return False

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get task by ID"""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[ScheduledTask]:
        """Get all tasks"""
        return list(self._tasks.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        stats = {
            "total_tasks": len(self._tasks),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "max_concurrent": self._max_concurrent,
            "queue_size": self._queue.qsize(),
            "is_running": self._running,
        }

        for task in self._tasks.values():
            if task.is_pending:
                stats["pending"] += 1
            elif task.is_running:
                stats["running"] += 1
            elif task.is_completed:
                stats["completed"] += 1
            elif task.is_failed:
                stats["failed"] += 1
            elif task.is_cancelled:
                stats["cancelled"] += 1

        return stats

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine"""
        while self._running:
            try:
                priority, task_id, func, args, kwargs = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            await self._execute_task(task_id, func, args, kwargs)
            self._queue.task_done()

    async def _execute_task(
        self,
        task_id: str,
        func: Callable[..., Awaitable[Any]],
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Execute a task"""
        task = self._tasks.get(task_id)
        if not task:
            return

        if task.is_cancelled:
            return

        async with self._lock:
            task.state = TaskState.RUNNING
            task.started_at = time.time()

        try:
            if task.timeout:
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=task.timeout)
            else:
                result = await func(*args, **kwargs)

            async with self._lock:
                task.state = TaskState.COMPLETED
                task.completed_at = time.time()
                task.execution_time = task.completed_at - task.started_at
                self._results[task_id] = result

            logger.debug(
                f"Task completed: {task_id} ({task.name}), "
                f"execution time: {task.execution_time:.2f}s"
            )
        except asyncio.TimeoutError:
            async with self._lock:
                task.state = TaskState.FAILED
                task.completed_at = time.time()
                task.execution_time = task.completed_at - task.started_at
                task.error = f"Timeout after {task.timeout}s"
            logger.warning(f"Task timed out: {task_id} ({task.name})")
        except asyncio.CancelledError:
            async with self._lock:
                task.state = TaskState.CANCELLED
                task.completed_at = time.time()
                task.execution_time = task.completed_at - task.started_at
            logger.info(f"Task cancelled: {task_id} ({task.name})")
        except Exception as e:
            async with self._lock:
                task.state = TaskState.FAILED
                task.completed_at = time.time()
                task.execution_time = task.completed_at - task.started_at
                task.error = str(e)
            logger.error(f"Task failed: {task_id} ({task.name}), error: {e}")
        finally:
            event = self._completion_events.get(task_id)
            if event:
                event.set()

    async def cleanup_completed(self, max_age: float = 3600.0) -> int:
        """Clean up completed tasks

        Args:
            max_age: Maximum age in seconds

        Returns:
            Number of cleaned tasks
        """
        async with self._lock:
            to_remove = []
            current_time = time.time()

            for task_id, task in self._tasks.items():
                if task.completed_at and (current_time - task.completed_at) > max_age:
                    to_remove.append(task_id)

            for task_id in to_remove:
                del self._tasks[task_id]
                self._completion_events.pop(task_id, None)
                self._results.pop(task_id, None)

            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove)} completed tasks")

            return len(to_remove)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()
