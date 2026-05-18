import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


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
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0
    progress_message: str = ""
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _subscribers: List[Queue] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "progress_message": self.progress_message,
        }


class TaskManager:
    MAX_HISTORY = 100

    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._handlers: Dict[str, Callable] = {}
        self._lock = threading.Lock()

    def register_handler(self, task_type: str, handler_fn: Callable):
        self._handlers[task_type] = handler_fn

    def create_task(self, task_type: str, params: Optional[Dict[str, Any]] = None) -> str:
        task_id = uuid.uuid4().hex
        task_info = TaskInfo(task_id=task_id, task_type=task_type)

        with self._lock:
            self._tasks[task_id] = task_info
            self._cleanup_oldest()

        thread = threading.Thread(
            target=self._run_task,
            args=(task_id, task_type, params or {}),
            daemon=True,
        )
        thread.start()

        return task_id

    def _cleanup_oldest(self):
        completed = [
            t for t in self._tasks.values()
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
        ]
        if len(completed) > self.MAX_HISTORY:
            completed.sort(key=lambda t: t.completed_at or 0)
            for t in completed[:len(completed) - self.MAX_HISTORY]:
                del self._tasks[t.task_id]

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        with self._lock:
            return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[TaskInfo]:
        with self._lock:
            return list(self._tasks.values())

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)

        if task is None:
            return False
        if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
            return False

        task._cancel_event.set()
        task.status = TaskStatus.CANCELLED
        task.completed_at = time.time()
        self._notify_subscribers(task, "error", {"message": "Task cancelled"})
        return True

    def is_cancelled(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        if task is None:
            return True
        return task._cancel_event.is_set()

    def update_progress(self, task_id: str, progress: float, message: str = ""):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.progress = progress
                task.progress_message = message

        if task:
            self._notify_subscribers(task, "progress", {
                "progress": progress,
                "message": message,
            })

    def complete_task(self, task_id: str, result: Dict[str, Any]):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.COMPLETED
                task.progress = 100.0
                task.result = result
                task.completed_at = time.time()

        if task:
            self._notify_subscribers(task, "complete", result)

    def fail_task(self, task_id: str, error: str):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.FAILED
                task.error = error
                task.completed_at = time.time()

        if task:
            self._notify_subscribers(task, "error", {"message": error})

    def subscribe(self, task_id: str) -> Optional[Queue]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            q: Queue = Queue()
            task._subscribers.append(q)
            return q

    def _notify_subscribers(self, task: TaskInfo, event: str, data: Dict[str, Any]):
        message = json.dumps({"event": event, "data": data})
        for q in task._subscribers:
            q.put(message)
        if event in ("complete", "error"):
            for q in task._subscribers:
                q.put(None)

    def _run_task(self, task_id: str, task_type: str, params: Dict[str, Any]):
        handler = self._handlers.get(task_type)
        if handler is None:
            self.fail_task(task_id, f"No handler registered for task type: {task_type}")
            return

        task = self.get_task(task_id)
        if task is None:
            return

        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        try:
            result = handler(task_id, params)
            if task._cancel_event.is_set():
                task.status = TaskStatus.CANCELLED
                task.completed_at = time.time()
                self._notify_subscribers(task, "error", {"message": "Task cancelled"})
            else:
                self.complete_task(task_id, result or {})
        except Exception as e:
            logger.exception("Task %s failed", task_id)
            self.fail_task(task_id, str(e))


_task_manager = TaskManager()


def get_task_manager() -> TaskManager:
    return _task_manager