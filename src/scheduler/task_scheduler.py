"""高性能异步任务调度器
支持优先级队列、工作窃取算法、任务超时和取消、指标采集和监控
"""

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Coroutine

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """任务优先级枚举"""
    CRITICAL = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    BACKGROUND = 100


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ScheduleStrategy(Enum):
    """调度策略枚举"""
    FIFO = "fifo"
    PRIORITY = "priority"
    FAIR = "fair"


@dataclass(order=True)
class PrioritizedTask:
    """优先级任务数据类"""
    priority: int
    sequence: int
    task_id: str = field(compare=False)
    coroutine: Callable = field(compare=False)
    created_at: float = field(default_factory=time.time, compare=False)

    def __post_init__(self):
        if not isinstance(self.coroutine, Coroutine):
            if asyncio.iscoroutinefunction(self.coroutine):
                self._coro_func = self.coroutine
            else:
                self._coro_func = None
        else:
            self._coro_func = None


@dataclass
class SchedulerStats:
    """调度器统计指标"""
    total_tasks_submitted: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    total_tasks_cancelled: int = 0
    total_tasks_timeout: int = 0
    avg_wait_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    current_queue_size: int = 0
    active_workers: int = 0
    total_wait_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0
    tasks_by_priority: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_tasks_submitted": self.total_tasks_submitted,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "total_tasks_cancelled": self.total_tasks_cancelled,
            "total_tasks_timeout": self.total_tasks_timeout,
            "avg_wait_time_ms": self.avg_wait_time_ms,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "current_queue_size": self.current_queue_size,
            "active_workers": self.active_workers,
            "tasks_by_priority": self.tasks_by_priority,
            "created_at": self.created_at.isoformat(),
        }

    def update_avg_wait_time(self, wait_time_ms: float) -> None:
        """更新平均等待时间"""
        self.total_wait_time_ms += wait_time_ms
        if self.total_tasks_completed > 0:
            self.avg_wait_time_ms = self.total_wait_time_ms / self.total_tasks_completed

    def update_avg_execution_time(self, execution_time_ms: float) -> None:
        """更新平均执行时间"""
        self.total_execution_time_ms += execution_time_ms
        if self.total_tasks_completed > 0:
            self.avg_execution_time_ms = self.total_execution_time_ms / self.total_tasks_completed


class WorkStealingQueue:
    """工作窃取队列
    支持本地任务缓存和跨队列窃取
    """

    def __init__(self, worker_id: int = 0):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._local_tasks: List[PrioritizedTask] = []
        self._stealable: bool = True
        self._lock = asyncio.Lock()
        self._worker_id = worker_id
        self._size = 0

    async def put(self, task: PrioritizedTask) -> None:
        """添加任务到队列"""
        async with self._lock:
            await self._queue.put(task)
            self._size += 1

    def put_nowait(self, task: PrioritizedTask) -> None:
        """非阻塞添加任务"""
        self._queue.put_nowait(task)
        self._size += 1

    async def get(self, timeout: Optional[float] = None) -> PrioritizedTask:
        """从队列获取任务"""
        try:
            if timeout is not None:
                task = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            else:
                task = await self._queue.get()
            async with self._lock:
                self._size -= 1
            return task
        except asyncio.TimeoutError:
            raise

    def get_nowait(self) -> Optional[PrioritizedTask]:
        """非阻塞获取任务"""
        try:
            task = self._queue.get_nowait()
            self._size -= 1
            return task
        except asyncio.QueueEmpty:
            return None

    async def steal(self, count: int = 1) -> List[PrioritizedTask]:
        """从队列窃取任务
        Args:
            count: 要窃取的任务数量
        Returns:
            被窃取的任务列表
        """
        stolen_tasks = []
        async with self._lock:
            if not self._stealable or self._size == 0:
                return stolen_tasks

            steal_count = min(count, max(1, self._size // 2))
            for _ in range(steal_count):
                try:
                    task = self._queue.get_nowait()
                    self._size -= 1
                    stolen_tasks.append(task)
                except asyncio.QueueEmpty:
                    break

        if stolen_tasks:
            logger.debug(
                f"Worker {self._worker_id} 被窃取了 {len(stolen_tasks)} 个任务"
            )
        return stolen_tasks

    def size(self) -> int:
        """获取队列大小"""
        return self._size

    def is_empty(self) -> bool:
        """队列是否为空"""
        return self._size == 0

    def set_stealable(self, stealable: bool) -> None:
        """设置是否可被窃取"""
        self._stealable = stealable

    @property
    def stealable(self) -> bool:
        """是否可被窃取"""
        return self._stealable


class TaskHandle:
    """任务句柄
    用于等待、取消和查询任务状态
    """

    def __init__(self, task_id: str, scheduler: "TaskScheduler"):
        self.task_id = task_id
        self._scheduler = scheduler
        self._result: Any = None
        self._exception: Optional[Exception] = None
        self._status = TaskStatus.PENDING
        self._event = asyncio.Event()
        self._asyncio_task: Optional[asyncio.Task] = None

    def _set_result(self, result: Any) -> None:
        """设置任务结果"""
        self._result = result
        self._status = TaskStatus.COMPLETED
        self._event.set()

    def _set_exception(self, exception: Exception) -> None:
        """设置任务异常"""
        self._exception = exception
        if isinstance(exception, asyncio.TimeoutError):
            self._status = TaskStatus.TIMEOUT
        elif isinstance(exception, asyncio.CancelledError):
            self._status = TaskStatus.CANCELLED
        else:
            self._status = TaskStatus.FAILED
        self._event.set()

    def _set_running(self) -> None:
        """设置任务运行中"""
        self._status = TaskStatus.RUNNING

    def _set_cancelled(self) -> None:
        """设置任务已取消"""
        self._status = TaskStatus.CANCELLED
        self._event.set()

    def _set_asyncio_task(self, task: asyncio.Task) -> None:
        """设置关联的 asyncio.Task"""
        self._asyncio_task = task

    async def wait(self, timeout: Optional[float] = None) -> Any:
        """等待任务完成
        Args:
            timeout: 超时时间（秒）
        Returns:
            任务结果
        Raises:
            TimeoutError: 等待超时
            Exception: 任务执行异常
        """
        try:
            if timeout is not None:
                await asyncio.wait_for(self._event.wait(), timeout=timeout)
            else:
                await self._event.wait()
        except asyncio.TimeoutError:
            raise TimeoutError(f"等待任务 {self.task_id} 超时")

        if self._exception is not None:
            raise self._exception
        return self._result

    async def cancel(self) -> bool:
        """取消任务
        Returns:
            是否成功取消
        """
        if self._status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return False

        if self._asyncio_task is not None:
            self._asyncio_task.cancel()
            try:
                await self._asyncio_task
            except asyncio.CancelledError:
                pass

        self._set_cancelled()
        await self._scheduler._on_task_cancelled(self.task_id)
        return True

    def status(self) -> TaskStatus:
        """获取任务状态"""
        return self._status

    def result(self) -> Any:
        """获取任务结果（如果已完成）"""
        if self._status != TaskStatus.COMPLETED:
            raise RuntimeError(f"任务未完成，当前状态: {self._status.value}")
        return self._result

    def exception(self) -> Optional[Exception]:
        """获取任务异常（如果已失败）"""
        if self._status not in (TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED):
            return None
        return self._exception

    def done(self) -> bool:
        """任务是否已完成"""
        return self._status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        )


class Worker:
    """工作线程
    从队列获取任务并执行，支持工作窃取
    """

    def __init__(
        self,
        worker_id: int,
        queue: WorkStealingQueue,
        scheduler: "TaskScheduler",
        steal_enabled: bool = True,
    ):
        self.worker_id = worker_id
        self._queue = queue
        self._scheduler = scheduler
        self._steal_enabled = steal_enabled
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._current_task_id: Optional[str] = None
        self._tasks_completed = 0

    async def start(self) -> None:
        """启动工作线程"""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.debug(f"Worker {self.worker_id} 已启动")

    async def stop(self) -> None:
        """停止工作线程"""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.debug(f"Worker {self.worker_id} 已停止")

    async def _run_loop(self) -> None:
        """工作线程主循环"""
        while self._running:
            try:
                task = await self._get_task()
                if task is not None:
                    await self._execute_task(task)
                else:
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {self.worker_id} 执行出错: {e}")

    async def _get_task(self) -> Optional[PrioritizedTask]:
        """获取任务（从本地队列或窃取）"""
        try:
            task = await self._queue.get(timeout=0.1)
            return task
        except asyncio.TimeoutError:
            if self._steal_enabled:
                return await self._try_steal()
            return None

    async def _try_steal(self) -> Optional[PrioritizedTask]:
        """尝试从其他队列窃取任务"""
        for other_queue in self._scheduler._queues:
            if other_queue is not self._queue and other_queue.stealable:
                stolen = await other_queue.steal(1)
                if stolen:
                    return stolen[0]
        return None

    async def _execute_task(self, prioritized_task: PrioritizedTask) -> None:
        """执行任务"""
        task_id = prioritized_task.task_id
        self._current_task_id = task_id
        start_time = time.time()

        handle = self._scheduler._handles.get(task_id)
        if handle is None:
            logger.warning(f"找不到任务句柄: {task_id}")
            return

        handle._set_running()

        try:
            coro = prioritized_task.coroutine
            if asyncio.iscoroutinefunction(coro):
                result = await coro()
            elif isinstance(coro, Coroutine):
                result = await coro
            else:
                result = coro

            handle._set_result(result)
            self._tasks_completed += 1

            execution_time_ms = (time.time() - start_time) * 1000
            await self._scheduler._on_task_completed(task_id, execution_time_ms)

        except asyncio.CancelledError:
            handle._set_exception(asyncio.CancelledError())
            await self._scheduler._on_task_cancelled(task_id)
            raise

        except asyncio.TimeoutError as e:
            handle._set_exception(e)
            await self._scheduler._on_task_timeout(task_id)

        except Exception as e:
            handle._set_exception(e)
            await self._scheduler._on_task_failed(task_id, e)

        finally:
            self._current_task_id = None

    @property
    def is_idle(self) -> bool:
        """是否空闲"""
        return self._current_task_id is None

    @property
    def tasks_completed(self) -> int:
        """已完成的任务数"""
        return self._tasks_completed


class TaskScheduler:
    """高性能异步任务调度器
    支持优先级队列、工作窃取、任务超时和取消、指标监控
    """

    def __init__(
        self,
        worker_count: int = 4,
        strategy: ScheduleStrategy = ScheduleStrategy.PRIORITY,
        steal_enabled: bool = True,
        default_timeout: Optional[float] = None,
    ):
        self._worker_count = worker_count
        self._strategy = strategy
        self._steal_enabled = steal_enabled
        self._default_timeout = default_timeout
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._running = False
        self._started = False

        self._workers: List[Worker] = []
        self._queues: List[WorkStealingQueue] = []
        self._handles: Dict[str, TaskHandle] = {}
        self._handles_lock = threading.Lock()

        self._stats = SchedulerStats()
        self._stats_lock = threading.Lock()
        self._main_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._submit_times: Dict[str, float] = {}

    async def start(self) -> None:
        """启动调度器"""
        if self._started:
            logger.warning("调度器已经启动")
            return

        self._running = True
        self._started = True

        for i in range(self._worker_count):
            queue = WorkStealingQueue(worker_id=i)
            self._queues.append(queue)
            worker = Worker(
                worker_id=i,
                queue=queue,
                scheduler=self,
                steal_enabled=self._steal_enabled,
            )
            self._workers.append(worker)
            await worker.start()

        logger.info(f"任务调度器已启动，工作线程数: {self._worker_count}")

    async def stop(self) -> None:
        """停止调度器"""
        if not self._started:
            return

        self._running = False

        for worker in self._workers:
            await worker.stop()

        self._workers.clear()
        self._queues.clear()
        self._started = False

        logger.info("任务调度器已停止")

    async def submit(
        self,
        coro: Callable,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
    ) -> str:
        """提交任务
        Args:
            coro: 协程函数或协程对象
            priority: 任务优先级
            timeout: 任务超时时间
        Returns:
            任务ID
        """
        if not self._running:
            raise RuntimeError("调度器未运行")

        with self._sequence_lock:
            self._sequence += 1
            sequence = self._sequence

        task_id = str(uuid.uuid4())

        prioritized_task = PrioritizedTask(
            priority=priority.value,
            sequence=sequence,
            task_id=task_id,
            coroutine=coro,
        )

        handle = TaskHandle(task_id, self)
        if timeout is None:
            timeout = self._default_timeout

        with self._handles_lock:
            self._handles[task_id] = handle

        self._submit_times[task_id] = time.time()

        with self._stats_lock:
            self._stats.total_tasks_submitted += 1
            priority_name = priority.name
            self._stats.tasks_by_priority[priority_name] = (
                self._stats.tasks_by_priority.get(priority_name, 0) + 1
            )

        queue_index = self._select_queue()
        await self._queues[queue_index].put(prioritized_task)

        logger.debug(f"任务 {task_id} 已提交，优先级: {priority.name}")
        return task_id

    def submit_sync(
        self,
        coro: Callable,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
    ) -> str:
        """同步提交任务（需要在事件循环中运行）"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.ensure_future(
                    self.submit(coro, priority, timeout)
                )
                task_id = loop.run_until_complete(future)
            else:
                task_id = loop.run_until_complete(
                    self.submit(coro, priority, timeout)
                )
        except RuntimeError:
            task_id = asyncio.run(self.submit(coro, priority, timeout))
        return task_id

    async def submit_batch(
        self,
        tasks: List[Tuple[Callable, TaskPriority]],
    ) -> List[str]:
        """批量提交任务
        Args:
            tasks: 任务列表，每个元素为 (协程, 优先级) 元组
        Returns:
            任务ID列表
        """
        task_ids = []
        for coro, priority in tasks:
            task_id = await self.submit(coro, priority)
            task_ids.append(task_id)
        return task_ids

    def _select_queue(self) -> int:
        """选择队列（根据调度策略）"""
        if self._strategy == ScheduleStrategy.FAIR:
            min_size = float('inf')
            selected = 0
            for i, queue in enumerate(self._queues):
                if queue.size() < min_size:
                    min_size = queue.size()
                    selected = i
            return selected
        else:
            import random
            return random.randint(0, len(self._queues) - 1)

    def get_handle(self, task_id: str) -> Optional[TaskHandle]:
        """获取任务句柄"""
        with self._handles_lock:
            return self._handles.get(task_id)

    async def wait(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """等待任务完成"""
        handle = self.get_handle(task_id)
        if handle is None:
            raise ValueError(f"任务不存在: {task_id}")
        return await handle.wait(timeout)

    async def cancel(self, task_id: str) -> bool:
        """取消任务"""
        handle = self.get_handle(task_id)
        if handle is None:
            return False
        return await handle.cancel()

    def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        with self._stats_lock:
            total_size = sum(q.size() for q in self._queues)
            self._stats.current_queue_size = total_size
            self._stats.active_workers = sum(
                1 for w in self._workers if not w.is_idle
            )
            return self._stats.to_dict()

    def get_queue_sizes(self) -> Dict[int, int]:
        """获取各队列大小"""
        return {i: q.size() for i, q in enumerate(self._queues)}

    async def _on_task_completed(self, task_id: str, execution_time_ms: float) -> None:
        """任务完成回调"""
        with self._stats_lock:
            self._stats.total_tasks_completed += 1
            self._stats.update_avg_execution_time(execution_time_ms)

            submit_time = self._submit_times.pop(task_id, None)
            if submit_time:
                wait_time_ms = (time.time() - submit_time) * 1000 - execution_time_ms
                self._stats.update_avg_wait_time(max(0, wait_time_ms))

        logger.debug(f"任务 {task_id} 已完成，执行时间: {execution_time_ms:.2f}ms")

    async def _on_task_failed(self, task_id: str, error: Exception) -> None:
        """任务失败回调"""
        with self._stats_lock:
            self._stats.total_tasks_failed += 1
            self._submit_times.pop(task_id, None)

        logger.error(f"任务 {task_id} 执行失败: {error}")

    async def _on_task_cancelled(self, task_id: str) -> None:
        """任务取消回调"""
        with self._stats_lock:
            self._stats.total_tasks_cancelled += 1
            self._submit_times.pop(task_id, None)

        logger.debug(f"任务 {task_id} 已取消")

    async def _on_task_timeout(self, task_id: str) -> None:
        """任务超时回调"""
        with self._stats_lock:
            self._stats.total_tasks_timeout += 1
            self._submit_times.pop(task_id, None)

        logger.warning(f"任务 {task_id} 执行超时")

    @property
    def is_running(self) -> bool:
        """调度器是否运行中"""
        return self._running

    @property
    def worker_count(self) -> int:
        """工作线程数"""
        return self._worker_count

    async def __aenter__(self) -> "TaskScheduler":
        """异步上下文管理器入口"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器出口"""
        await self.stop()


def create_default_scheduler(
    worker_count: int = 4,
    strategy: ScheduleStrategy = ScheduleStrategy.PRIORITY,
) -> TaskScheduler:
    """创建默认调度器"""
    return TaskScheduler(
        worker_count=worker_count,
        strategy=strategy,
        steal_enabled=True,
    )
