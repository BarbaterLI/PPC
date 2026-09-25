"""Worker 协程池组件（自 DistributedScheduler 拆分）。

管理 N 个 worker 协程：从 :class:`TaskQueue` 取 task_id，
交给注入的执行回调（scheduler 门面）处理。
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from src.core.exceptions import PPCTimeoutError
from src.distributed.task_queue import TaskAssignment, TaskQueue

logger = logging.getLogger(__name__)


class WorkerPool:
    """worker 协程池：启动 / 停止 / 计数。"""

    def __init__(self, num_workers: int, queue: TaskQueue) -> None:
        self._num_workers = num_workers
        self._queue = queue
        self._workers: list[asyncio.Task] = []

    async def start(self, execute_task: Callable[[TaskAssignment], Awaitable[None]]) -> None:
        for i in range(self._num_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}", execute_task))
            self._workers.append(worker)

    async def stop(self) -> None:
        for worker in self._workers:
            worker.cancel()

        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    def active_count(self) -> int:
        return len([w for w in self._workers if not w.done()])

    async def _worker_loop(
        self,
        worker_id: str,
        execute_task: Callable[[TaskAssignment], Awaitable[None]],
    ):
        """Worker main loop"""
        try:
            while True:
                task_id = await self._queue.queue_get()

                try:
                    task = self._queue.get(task_id)
                    if task is not None:
                        await execute_task(task)
                except asyncio.CancelledError:
                    # 由 finally 统一调用一次 task_done()，避免双重记账导致
                    # queue.join() 在任务实际完成前就返回（不变量：每个出队
                    # 任务恰好一次 task_done()）。
                    logger.info("Worker %s: task %s cancelled", worker_id, task_id)
                    raise
                except Exception as e:
                    error_type = type(e).__name__
                    is_retryable = isinstance(e, (ConnectionError, PPCTimeoutError, OSError))
                    logger.error(
                        "Worker %s failed to process task %s [%s] (retryable=%s): %s",
                        worker_id,
                        task_id,
                        error_type,
                        is_retryable,
                        e,
                        exc_info=True,
                    )
                    if task is not None:
                        task.error = f"{error_type}: {e}"
                finally:
                    self._queue.task_done()

        except asyncio.CancelledError:
            logger.info("Worker %s cancelled", worker_id)
