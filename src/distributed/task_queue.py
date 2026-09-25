"""调度器任务队列组件（自 DistributedScheduler 拆分）。

承载任务注册表 / asyncio 队列 / 计数器；执行逻辑在
:class:`src.distributed.worker_pool.WorkerPool` 与
:class:`src.distributed.scheduler.DistributedScheduler`。
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class TaskAssignment:
    """Task assignment information.

    A single task corresponds to one text->audio synthesis. The
    scheduler forwards a single :class:`ConvertRequest` to a worker per
    task, so this dataclass still mirrors the per-text shape that legacy
    callers expect.
    """

    task_id: str
    text: str
    voice: str
    rate: str
    output_path: Path
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: str | None = None
    attempts: int = 0
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "task_id": self.task_id,
            "text": self._truncate_text(self.text),
            "voice": self.voice,
            "rate": self.rate,
            "output_path": str(self.output_path),
            "status": self.status.value,
            "assigned_node": self.assigned_node,
            "attempts": self.attempts,
            "error": self.error,
            "duration": self.duration,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @staticmethod
    def _truncate_text(text: str, max_length: int = 50) -> str:
        """Truncate text for display"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."


class TaskQueue:
    """任务注册表 + 待执行队列 + 计数器。"""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskAssignment] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._counter_lock = asyncio.Lock()

        self._total_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0

    async def submit(self, task: TaskAssignment) -> str:
        async with self._lock:
            self._tasks[task.task_id] = task
            self._total_tasks += 1

        await self._queue.put(task.task_id)
        logger.debug("Task submitted: %s", task.task_id)

        return task.task_id

    async def submit_batch(self, tasks: list[TaskAssignment]) -> list[str]:
        submitted_task_ids: list[str] = []
        try:
            for task in tasks:
                task_id = await self.submit(task)
                submitted_task_ids.append(task_id)
        except Exception as e:
            for submitted_id in submitted_task_ids:
                del self._tasks[submitted_id]
                self._total_tasks -= 1
            logger.warning("Batch submission failed, rolled back %d tasks: %s", len(submitted_task_ids), e)
            raise
        return submitted_task_ids

    def get(self, task_id: str) -> TaskAssignment | None:
        return self._tasks.get(task_id)

    def all(self) -> dict[str, TaskAssignment]:
        return self._tasks.copy()

    def pending(self) -> int:
        return self._queue.qsize()

    async def queue_get(self) -> str:
        return await self._queue.get()

    async def enqueue(self, task_id: str) -> None:
        await self._queue.put(task_id)

    def task_done(self) -> None:
        self._queue.task_done()

    async def drain(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def mark_completed(self) -> None:
        async with self._counter_lock:
            self._completed_tasks += 1

    async def mark_failed(self) -> None:
        async with self._counter_lock:
            self._failed_tasks += 1

    def stats(self) -> dict[str, Any]:
        return {
            "total_tasks": self._total_tasks,
            "completed_tasks": self._completed_tasks,
            "failed_tasks": self._failed_tasks,
        }
