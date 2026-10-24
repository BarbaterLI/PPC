"""TTS Worker Pool - Worker orchestration for TTSExecutor.

Owns the priority task queue, worker coroutines, concurrency semaphore and
the ramp-up (concurrency warm-up) adjuster. ``TTSExecutor`` passes a bound
processor callback, so the pool never touches executor internals.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_SENTINEL = object()


class RampUpController:
    """并发渐进预热控制器

    从 1 并发逐步增加到目标并发数，规避风控。
    """

    def __init__(self, target_concurrency: int, duration: float = 30.0):
        self._target = target_concurrency
        self._duration = duration
        self._start_time: float | None = None
        self._current = 1

    def start(self):
        self._start_time = time.time()
        self._current = 1

    def get_current_concurrency(self) -> int:
        if self._start_time is None:
            return 1
        elapsed = time.time() - self._start_time
        if elapsed >= self._duration:
            return self._target
        ratio = elapsed / self._duration
        self._current = max(1, int(self._target * ratio))
        return self._current

    @property
    def is_complete(self) -> bool:
        if self._start_time is None:
            return False
        return (time.time() - self._start_time) >= self._duration

    @property
    def target(self) -> int:
        return self._target

    @property
    def duration(self) -> float:
        return self._duration


class WorkerPool:
    """TTS 任务工作池

    持有优先级任务队列、信号量与工作协程；通过 ``processor`` 回调把单个
    任务的处理委托回执行器，池自身不读写执行器内部状态。
    """

    def __init__(
        self,
        concurrency: int,
        processor: Callable[..., Any],
        ramp_up_controller: RampUpController | None = None,
    ):
        self.concurrency = concurrency
        self.processor = processor
        self.ramp_up_controller = ramp_up_controller
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(1 if ramp_up_controller is not None else concurrency)
        self.workers: list[asyncio.Task[Any]] = []
        self._ramp_up_task: asyncio.Task[Any] | None = None
        self._ramp_up_released = 0
        self._progress_handler: Any | None = None
        # shutdown 后队列中可能残留未被消费的 sentinel，见 _drain_sentinels
        self._shutdown = False

    @property
    def ramp_up_released(self) -> int:
        return self._ramp_up_released

    async def enqueue(self, task: Any):
        """入队一个任务（按优先级排队）。"""
        await self.queue.put((task.priority, time.time(), task.id, task))

    async def start(self, progress_handler: Any | None = None):
        """启动并发预热（如启用）与全部工作协程。"""
        # 允许同一个池服务多个批次：上一批 shutdown 已结束旧 worker，
        # 这里重新开始即可（残留 sentinel 已在 shutdown 内清空）。
        self._shutdown = False
        self._progress_handler = progress_handler
        self.workers = []
        if self.ramp_up_controller is not None:
            self.ramp_up_controller.start()
            ramp_up_task = asyncio.create_task(self._ramp_up_adjuster())
            self._ramp_up_task = ramp_up_task
            self.workers.append(ramp_up_task)
            logger.info(
                "并发预热已启动: %ds 内从 1 逐步增加到 %d 并发",
                self.ramp_up_controller.duration,
                self.ramp_up_controller.target,
            )
        for i in range(self.concurrency):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.workers.append(worker)
        logger.info("已启动 %d 个工作协程", len(self.workers))

    async def wait_all(self):
        """等待队列中所有任务处理完成。"""
        await self.queue.join()
        logger.info("所有任务已从队列完成，正在通知 Workers 退出...")

    async def shutdown(self, timeout: float = 5.0):
        """向每个工作协程发送退出信号并等待其安全退出。"""
        # 先取消并摘除 ramp-up 调整器：它不消费 sentinel，只靠 is_complete
        # （ramp 完成）或被取消退出。若留在 gather 里，ramp 未完成时会干等
        # 满 timeout 才强制取消（实测 ~5s 关停延迟）。
        if self._ramp_up_task is not None:
            self._ramp_up_task.cancel()
            try:
                await self._ramp_up_task
            except asyncio.CancelledError:
                pass
            if self._ramp_up_task in self.workers:
                self.workers.remove(self._ramp_up_task)
            self._ramp_up_task = None

        for i in range(len(self.workers)):
            await self.queue.put((0, time.time(), f"sentinel-{i}", _SENTINEL))

        if self.workers:
            try:
                await asyncio.wait_for(asyncio.gather(*self.workers, return_exceptions=True), timeout=timeout)
                logger.info("所有工作协程已安全退出")
            except asyncio.TimeoutError:
                logger.warning("等待工作协程退出超时，强制取消")
                for worker in self.workers:
                    if not worker.done():
                        worker.cancel()

        self._shutdown = True
        # 触发条件：sentinel 数量多于实际消费它的 worker（worker 提前退出/
        # 被取消），或某 worker 退出前已取走多个。后果：这些 sentinel 从未
        # task_done，queue._unfinished_tasks 永远 > 0，下一批次的
        # wait_all()（queue.join）会永久挂起；若被新 worker 优先取到，还会
        # 立即退出导致真实任务无人消费。故 shutdown 末尾清空残留 sentinel。
        self._drain_sentinels()

    def _drain_sentinels(self) -> None:
        """丢弃队列中残留的 sentinel，真实任务原样重新入队。"""
        leftover: list[tuple[Any, ...]] = []
        while True:
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item[3] is _SENTINEL:
                self.queue.task_done()  # 抵消 shutdown 时 put 的未完成计数
            else:
                leftover.append(item)
        for item in leftover:
            # 真实任务不能丢：task_done 抵消 get，put 重新计入未完成数
            self.queue.task_done()
            self.queue.put_nowait(item)

    def clear_workers(self):
        """清空工作协程引用（执行器 cleanup 时调用）。"""
        self.workers = []

    async def _worker_loop(self, worker_id: str):
        try:
            while True:
                # 12.x W2-E：限流门移到取任务**之前**——旧实现先出队再抢
                # 信号量，被阻塞的 worker 会持有任务空等（预热期任务滞留、
                # 关停时滞留加剧）。现在无许可的 worker 挂在 semaphore 上，
                # 队列把任务交给当下空闲的 worker。
                gated = self.ramp_up_controller is not None
                if gated:
                    await self.semaphore.acquire()
                task = await self._get_next_task()
                if task is None:
                    if gated:
                        self.semaphore.release()
                    logger.info("工作协程 %s 正常退出", worker_id)
                    break
                await self._process_single_task(task, worker_id, gated)
        except asyncio.CancelledError:
            logger.info("工作协程 %s 被取消", worker_id)
        except Exception as e:
            logger.error("工作协程 %s 异常：%s", worker_id, e, exc_info=True)

    async def _get_next_task(self) -> Any | None:
        try:
            priority, timestamp, task_id, task = await self.queue.get()

            if task is _SENTINEL:
                # 触发条件：worker 取到 shutdown 投递的退出信号。sentinel 也是
                # 一次 put，必须在这里 task_done，否则 unfinished 计数永远
                # 无法归零，后续批次的 queue.join() 会永久阻塞。
                logger.debug("Worker 收到退出信号")
                self.queue.task_done()
                return None

            return task
        except Exception as e:
            logger.warning("获取任务时发生异常: %s", e)
            return None

    async def _process_single_task(self, task: Any, worker_id: str, gated: bool = True):
        try:
            await self.processor(task, worker_id, self._progress_handler)
        except Exception as e:
            # processor 意外逃逸异常：显式记录（不静默丢失），后续由
            # 执行器侧任务状态兜底；task_done() 仍由 finally 保证恰好一次。
            logger.error("Worker %s 处理任务 %s 时异常：%s", worker_id, getattr(task, "id", task), e, exc_info=True)
        finally:
            if gated:
                self.semaphore.release()
            self.queue.task_done()

    async def _ramp_up_adjuster(self):
        """后台调整任务：每秒检查并发预热进度，逐步释放信号量许可。"""
        controller = self.ramp_up_controller
        try:
            while not controller.is_complete:
                await asyncio.sleep(1)
                current = controller.get_current_concurrency()
                while self._ramp_up_released < current - 1:
                    self.semaphore.release()
                    self._ramp_up_released += 1
            target = controller.target
            while self._ramp_up_released < target - 1:
                self.semaphore.release()
                self._ramp_up_released += 1
            logger.info("并发预热完成，已达到 %d 并发", target)
        except asyncio.CancelledError:
            target = controller.target
            while self._ramp_up_released < target - 1:
                self.semaphore.release()
                self._ramp_up_released += 1
            raise
