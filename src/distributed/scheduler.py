"""Unified distributed scheduler for PPC_NEXT TTS cluster.

Merges MasterScheduler and NodePool functionality into a single
coordinated scheduler that manages:
- Node pool management (add/remove/health check)
- Task scheduling and assignment
- Load balancing (pluggable strategies)
- Local fallback execution (only when no workers are available and
  ``local_fallback`` is enabled)

Design note: the scheduler is now a thin coordinator on top of the
:mod:`src.distributed.processing_unit` module. When a task is
submitted the scheduler forwards it to a worker's ``/api/v1/convert``
endpoint; the worker is the only place that actually runs TTS. This
means the master and worker share the same TTS execution code as
``ppc convert``.

Wave 2: the god object is decomposed into four components
(:class:`TaskQueue` / :class:`WorkerPool` / :class:`HealthChecker` /
:class:`LoadBalancer`); this class remains the composition facade with
an unchanged public API.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout

from src.config import PPCConfig
from src.distributed.load_balancer import LoadBalancer
from src.distributed.node_pool import NodeInfo, NodePool
from src.distributed.processing_unit import (
    ConvertRequest,
    ConvertResult,
    UnitRole,
    WorkerUnit,
    make_processing_unit,
)
from src.distributed.task_queue import TaskAssignment, TaskQueue, TaskStatus
from src.distributed.worker_pool import WorkerPool
from src.extensions.base import LoadBalanceStrategy
from src.web.auth import auth_headers

logger = logging.getLogger(__name__)


class RemoteExecutionError(Exception):
    """远程节点执行失败（可重试）。

    触发条件：连接失败、请求超时、节点返回非 200 等“远端确定没有产出
    音频”的失败。
    后果：由 :meth:`DistributedScheduler._execute_task` 的重试循环处理
    （旧实现把这类失败吞成 ``False``，``max_retries`` 完全失效）。
    """


class RemoteResponseError(RemoteExecutionError):
    """节点已执行但响应读取/落盘失败（不可重试、不可本地重跑）。

    触发条件：HTTP 200 之后读取响应体、写输出文件或解析响应头失败。
    后果：远端可能已经合成并落盘，重试或本地兜底会让同一任务合成两次，
    因此调用方必须直接判失败（见 :meth:`_execute_task`）。
    """


class DistributedScheduler:
    """Unified distributed scheduler.

    The scheduler keeps the original public API
    (``start``/``stop``/``submit_task``/``submit_batch``/``get_stats``/
    ``on_task_complete``/``on_task_failed``/``set_load_balance_strategy``/
    ``add_node``/``remove_node``) so existing callers keep working. The
    key change is that TTS work is now done by the workers themselves
    via the :class:`ProcessingUnit` abstraction. The scheduler only
    chooses which worker should handle a task and forwards the
    request.

    If ``local_fallback`` is ``True`` and the node pool is empty, the
    scheduler instantiates a local :class:`WorkerUnit` and runs the task
    locally. Default is ``False`` so master nodes do not perform TTS
    unless explicitly configured.
    """

    def __init__(
        self,
        config: PPCConfig,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        task_timeout: float = 300.0,
        load_balance_strategy: str = "round_robin",
        local_execution: bool = True,
        health_check_interval: float = 30.0,
        health_check_timeout: float = 5.0,
        unhealthy_threshold: int = 3,
        custom_lb_strategy: LoadBalanceStrategy | None = None,
        shutdown_timeout: float = 30.0,
        local_fallback: bool | None = None,
        worker_unit: WorkerUnit | None = None,
    ):
        # ``local_fallback`` is the new name for the old ``local_execution``
        # behaviour, but we keep both for backwards compatibility. If the
        # caller passes ``local_fallback`` we honour it; otherwise we fall
        # back to ``local_execution`` (default True).
        if local_fallback is None:
            local_fallback = local_execution
        self.config = config
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.task_timeout = task_timeout
        self.load_balance_strategy_name = load_balance_strategy
        self.local_execution = local_execution
        self.local_fallback = local_fallback
        self.shutdown_timeout = shutdown_timeout

        self._node_pool = NodePool(
            health_check_interval=health_check_interval,
            health_check_timeout=health_check_timeout,
            unhealthy_threshold=unhealthy_threshold,
        )
        self._load_balancer = LoadBalancer(
            default_strategy=load_balance_strategy,
            custom_strategy=custom_lb_strategy,
        )
        self._task_queue = TaskQueue()
        self._http_session: ClientSession | None = None
        self._num_workers = config.tts.concurrency
        self._worker_pool = WorkerPool(self._num_workers, self._task_queue)

        self._on_task_complete: list[Callable] = []
        self._on_task_failed: list[Callable] = []
        self._lock = asyncio.Lock()
        self._custom_lb_strategy = custom_lb_strategy
        # 生命周期幂等标记：重复 start() 会覆盖 ClientSession / 健康检查任务、
        # 追加 N 个 worker 协程（旧协程永不退出）。
        self._started = False

        # Optional injected local worker used for fallback. When the
        # caller does not provide one we lazily create it in :meth:`start`.
        self._local_worker_unit: WorkerUnit | None = worker_unit

        logger.info(
            "DistributedScheduler initialized: workers=%s, strategy=%s, local_execution=%s, local_fallback=%s",
            self._num_workers,
            load_balance_strategy,
            local_execution,
            self.local_fallback,
        )

    @property
    def node_pool(self) -> NodePool:
        """Access to the underlying node pool for node management."""
        return self._node_pool

    async def start(self):
        """Start the scheduler（幂等：重复调用直接返回）。

        触发条件：CLI/Web/测试重复调用 ``start()``。
        后果：旧实现每次都会新建 ``ClientSession``（旧会话未 close）、
        追加 N 个 worker 协程、覆盖 ``_health_check_task`` 引用（旧任务变孤儿）。
        """
        if self._started:
            logger.debug("DistributedScheduler 已在运行，忽略重复 start()")
            return

        await self._node_pool.start()

        self._http_session = ClientSession(
            timeout=ClientTimeout(total=self.task_timeout),
            headers=auth_headers(),
        )

        await self._worker_pool.start(self._execute_task)

        # Lazily create the local worker used for fallback. This only
        # spins up a TTSExecutor when ``local_fallback`` is enabled.
        if self.local_fallback and self._local_worker_unit is None:
            self._local_worker_unit = make_processing_unit(
                role=UnitRole.WORKER,
                host="local",
                port=0,
                config=self.config,
                max_concurrency=self.config.tts.concurrency,
                node_id=f"{id(self)}-local",
            )
            try:
                await self._local_worker_unit.start()
            except Exception as e:  # noqa: BLE001
                logger.debug("local worker start: %s", e)

        self._started = True
        logger.info("DistributedScheduler started with %d workers", self._num_workers)

    async def stop(self):
        """Stop the scheduler（关闭 session、取消健康检查、停止 worker）。"""
        if not self._started:
            # 未启动时仍执行一次幂等清理（Session 可能由外部注入），
            # 但不重复 drain 队列。
            await self._node_pool.stop()
            if self._http_session is not None:
                await self._http_session.close()
                self._http_session = None
            return

        if not await self._task_queue.drain(self.shutdown_timeout):
            logger.warning(
                "Queue drain timed out after %ss, cancelling pending tasks",
                self.shutdown_timeout,
            )

        await self._worker_pool.stop()

        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        if self._local_worker_unit is not None:
            try:
                await self._local_worker_unit.stop()
            except Exception as e:  # noqa: BLE001
                logger.debug("local worker stop: %s", e)
            self._local_worker_unit = None

        await self._node_pool.stop()

        self._started = False
        logger.info("DistributedScheduler stopped")

    async def submit_task(self, task: TaskAssignment) -> str:
        """Submit a single task"""
        return await self._task_queue.submit(task)

    async def submit_batch(self, tasks: list[TaskAssignment]) -> list[str]:
        """Submit multiple tasks in batch"""
        return await self._task_queue.submit_batch(tasks)

    async def get_task_status(self, task_id: str) -> TaskAssignment | None:
        """Get task status by ID"""
        return self._task_queue.get(task_id)

    def get_all_tasks(self) -> dict[str, TaskAssignment]:
        """Get all tasks"""
        return self._task_queue.all()

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics"""
        stats = self._task_queue.stats()
        return {
            "total_tasks": stats["total_tasks"],
            "completed_tasks": stats["completed_tasks"],
            "failed_tasks": stats["failed_tasks"],
            "pending_tasks": self._task_queue.pending(),
            "active_workers": self._worker_pool.active_count(),
            "node_pool": self._node_pool.get_stats(),
        }

    def on_task_complete(self, callback: Callable):
        """Register task complete callback"""
        self._on_task_complete.append(callback)

    def on_task_failed(self, callback: Callable):
        """Register task failed callback"""
        self._on_task_failed.append(callback)

    def set_load_balance_strategy(self, strategy: LoadBalanceStrategy):
        """Set a custom load balance strategy.

        Args:
            strategy: Custom load balance strategy implementation
        """
        self._custom_lb_strategy = strategy
        self._load_balancer.set_custom_strategy(strategy)
        logger.info(f"Custom load balance strategy set: {strategy.get_name()}")

    async def add_node(
        self,
        host: str,
        port: int,
        node_id: str | None = None,
        max_concurrency: int = 4,
    ) -> NodeInfo:
        """Add a node to the pool.

        Args:
            host: Node host address
            port: Node port
            node_id: Optional node identifier
            max_concurrency: Maximum concurrent tasks for this node

        Returns:
            NodeInfo for the added node
        """
        return await self._node_pool.add_node(host, port, node_id, max_concurrency)

    async def remove_node(self, node_id: str) -> bool:
        """Remove a node from the pool.

        Args:
            node_id: Node identifier to remove

        Returns:
            True if node was removed, False if not found
        """
        return await self._node_pool.remove_node(node_id)

    async def forward_convert(self, node: NodeInfo, request: ConvertRequest) -> ConvertResult:
        """Forward a convert request to a specific node.

        Public helper used by callers that need to push a convert
        request through the HTTP transport without going through the
        task queue.
        """
        if self._http_session is None:
            raise RuntimeError("Scheduler not started")
        url = f"{node.base_url}/api/v1/convert"
        async with self._http_session.post(url, json=request.to_dict()) as resp:
            data = await resp.json(content_type=None)
            if resp.status >= 400:
                raise RuntimeError(str(data.get("error") or data))
            return ConvertResult.from_dict(data)

    async def _execute_task(self, task: TaskAssignment):
        """Execute a single task by forwarding it to a worker (or running locally)."""
        task.status = TaskStatus.RUNNING
        task.attempts += 1
        task.started_at = datetime.now(timezone.utc)

        # 远程失败标记：只有“远端确定未产出”的失败（RemoteExecutionError）
        # 才允许在重试耗尽后走本地兜底，响应解析失败不允许（避免重复合成）。
        remote_failure: RemoteExecutionError | None = None

        try:
            if self._should_use_remote_node():
                selected_node = await self._select_best_node()
                if selected_node is not None:
                    task.assigned_node = selected_node.node_id

                try:
                    success = await self._execute_on_remote_node(task, selected_node)
                except RemoteResponseError as e:
                    # 触发条件：节点返回 200 但读取响应/落盘失败。
                    # 后果：远端可能已完成合成，重试或本地重跑会导致同一
                    # 任务被合成两次 → 直接判失败，不重试、不兜底。
                    await self._record_node_failure(selected_node)
                    task.error = str(e)
                    logger.error(
                        "Task failed after remote response error %s: %s",
                        task.task_id,
                        e,
                    )
                    task.status = TaskStatus.FAILED
                    await self._task_queue.mark_failed()
                    await self._notify_task_failed(task)
                    return
                except RemoteExecutionError as e:
                    # 触发条件：连接失败/超时/节点非 200 等可重试失败。
                    # 后果：交给下方重试逻辑；旧实现直接返回 False 并立刻
                    # 本地兜底，max_retries 对远程失败完全失效。
                    await self._record_node_failure(selected_node)
                    remote_failure = e
                    raise

                if success:
                    return
                # 无可用节点或远程返回 False（远端未产出）：记录失败后走
                # 兜底/重试逻辑。
                await self._record_node_failure(selected_node)

            if self.local_fallback:
                await self._execute_on_local_node(task)
                return

            task.status = TaskStatus.FAILED
            task.error = "No available nodes and local fallback disabled"
            await self._task_queue.mark_failed()
            await self._notify_task_failed(task)

        except Exception as e:
            task.error = str(e)
            logger.error("Task execution failed %s: %s", task.task_id, e)

            if task.attempts < self.max_retries:
                task.status = TaskStatus.RETRYING
                logger.info(
                    "Task will retry in %ss (%s/%s): %s",
                    self.retry_delay,
                    task.attempts,
                    self.max_retries,
                    task.task_id,
                )
                await asyncio.sleep(self.retry_delay)
                await self._task_queue.enqueue(task.task_id)
            elif self.local_fallback and remote_failure is not None:
                # 远程重试耗尽且远端未产出音频 → 本地兜底
                logger.warning(
                    "Remote retries exhausted for %s, falling back to local: %s",
                    task.task_id,
                    remote_failure,
                )
                try:
                    await self._execute_on_local_node(task)
                except Exception as local_error:
                    task.error = f"{e}; local fallback failed: {local_error}"
                    logger.error(
                        "Local fallback failed %s: %s", task.task_id, local_error
                    )
                    task.status = TaskStatus.FAILED
                    await self._task_queue.mark_failed()
                    await self._notify_task_failed(task)
            else:
                task.status = TaskStatus.FAILED
                await self._task_queue.mark_failed()
                await self._notify_task_failed(task)

    async def _record_node_failure(self, node: NodeInfo | None) -> None:
        """记录一次远程节点失败（增量计数，保证统计线性增长）。"""
        if node is None:
            return
        latest_node = self._node_pool.get_node(node.node_id)
        if latest_node is None:
            return
        await self._node_pool.update_node_stats(
            latest_node.node_id,
            {
                "total_requests": 1,
                "failed_requests": 1,
                "current_concurrency": latest_node.current_concurrency,
            },
        )

    async def _execute_on_remote_node(self, task: TaskAssignment, node: NodeInfo | None = None) -> bool:
        """Forward the task to a remote node and stream the audio back.

        For backwards compatibility with the legacy single-text API
        (which used ``/api/v1/synthesize``) the request is sent as a
        single-text payload to that endpoint. The full convert flow
        (multiple files / directory handling) should use
        :meth:`forward_convert` directly.

        Returns ``True`` on success and ``False`` when no node is
        available (nothing was attempted). Remote failures are raised as
        :class:`RemoteExecutionError` / :class:`RemoteResponseError` so
        the caller can apply the retry policy — the previous
        ``return False`` swallowed them and disabled ``max_retries``.
        """
        try:
            if node is None:
                node = await self._select_best_node()
            if node is None:
                logger.warning("No available nodes, falling back to local execution")
                return False

            task.assigned_node = node.node_id

            if self._http_session is None:
                raise RuntimeError("Scheduler not started")

            url = f"{node.base_url}/api/v1/synthesize"
            payload = {
                "text": task.text,
                "voice": task.voice,
                "rate": task.rate,
            }

            async with self._http_session.post(url, json=payload) as response:
                if response.status != 200:
                    try:
                        error_data = await response.json()
                    except Exception:  # noqa: BLE001 - 错误体可能不是 JSON
                        error_data = {}
                    error_text = (
                        error_data.get("error", "Unknown error")
                        if isinstance(error_data, dict)
                        else str(error_data)
                    )
                    # 节点明确返回错误：远端未产出音频，重试是安全的
                    raise RemoteExecutionError(f"Node error: {error_text}")

                # 200 之后节点已完成合成：任何读取/落盘失败都不能再重试，
                # 否则远端与本地会各合成一次（同一任务产出两份音频）。
                try:
                    audio_data = await response.read()

                    task.output_path.parent.mkdir(parents=True, exist_ok=True)
                    task.output_path.write_bytes(audio_data)

                    duration_header = response.headers.get("X-Duration", "0")
                    task.duration = float(duration_header)
                    task.completed_at = datetime.now(timezone.utc)
                    task.status = TaskStatus.COMPLETED
                    await self._task_queue.mark_completed()

                    # P2-17: 使用 worker 回传的实时并发，而不是本地旧值。
                    # 触发条件：worker 在响应头 X-Current-Concurrency 中上报
                    # 当前处理中的请求数；后果：旧实现回传 node.current_concurrency
                    # （永远是 0），LeastConnections/BestResponseTime 与 drain
                    # 判定全部失效。
                    reported = response.headers.get("X-Current-Concurrency")
                    try:
                        current_concurrency = (
                            int(reported) if reported is not None else node.current_concurrency
                        )
                    except (TypeError, ValueError):
                        current_concurrency = node.current_concurrency

                    await self._node_pool.update_node_stats(
                        node.node_id,
                        {
                            "total_requests": 1,
                            "successful_requests": 1,
                            "current_concurrency": current_concurrency,
                        },
                    )
                except Exception as e:  # noqa: BLE001
                    raise RemoteResponseError(
                        f"Remote node executed but response handling failed: {e}"
                    ) from e

                logger.info("Task completed %s (node: %s)", task.task_id, node.node_id)
                await self._notify_task_complete(task)
                return True

        except RemoteExecutionError:
            # 已分类的失败：保持类型，交由调用方决定重试/兜底
            raise
        except Exception as e:
            logger.warning("Remote node execution failed %s: %s", task.task_id, e)
            raise RemoteExecutionError(f"Remote node execution failed: {e}") from e

    async def _execute_on_local_node(self, task: TaskAssignment):
        """Execute task on the local fallback worker.

        This is the only place where a master can run TTS itself, and
        it only happens when ``local_fallback`` is enabled. The work is
        delegated to the shared :class:`WorkerUnit`, which uses the
        same :class:`TTSExecutor` that ``ppc convert`` uses. Since
        per-text execution isn't a directory-style convert, we materialise
        the text into a temporary file and call the worker's convert
        handler. If the worker isn't ready we fall back to a simple
        failure notification.
        """
        if self._local_worker_unit is None:
            raise RuntimeError("Local worker not initialised; local_fallback disabled?")

        task.assigned_node = "local"
        start_time = time.time()

        try:
            import tempfile

            with tempfile.TemporaryDirectory(prefix="ppc-sched-") as tmpdir:
                in_dir = Path(tmpdir) / "in"
                out_dir = Path(tmpdir) / "out"
                in_dir.mkdir(parents=True, exist_ok=True)
                out_dir.mkdir(parents=True, exist_ok=True)
                input_file = in_dir / f"{task.task_id}.txt"
                input_file.write_text(task.text, encoding="utf-8")

                request = ConvertRequest(
                    input_dir=in_dir,
                    output_dir=out_dir,
                    voice=task.voice or None,
                    rate=task.rate or None,
                    recursive=False,
                )
                result = await self._local_worker_unit.handle_convert_request(request)

                task.duration = time.time() - start_time
                task.completed_at = datetime.now(timezone.utc)

                if not result.success:
                    raise RuntimeError(result.error or "Local execution failed")

                # Copy the synthesised file back to the task's output path.
                expected = out_dir / input_file.with_suffix(".mp3").name
                if expected.exists():
                    task.output_path.parent.mkdir(parents=True, exist_ok=True)
                    task.output_path.write_bytes(expected.read_bytes())

                task.status = TaskStatus.COMPLETED
                await self._task_queue.mark_completed()
                logger.info("Task completed %s (local execution)", task.task_id)
                await self._notify_task_complete(task)
        except Exception as e:
            logger.error("Local execution failed %s: %s", task.task_id, e)
            raise

    async def _select_best_node(self) -> NodeInfo | None:
        """Select the best node using the configured or custom strategy.

        This is the strategy injection point that allows custom load
        balancing strategies to be used.
        """
        available = self._node_pool.get_available_nodes()
        if not available:
            return None

        task_context = {
            "strategy": self.load_balance_strategy_name,
        }
        return await self._load_balancer.select_node(available, task_context=task_context)

    def _should_use_remote_node(self) -> bool:
        """Determine whether to use remote nodes"""
        return len(self._node_pool.get_available_nodes()) > 0

    async def _notify_task_complete(self, task: TaskAssignment):
        """Notify task completion"""
        await self._emit_callbacks(self._on_task_complete, task)

    async def _notify_task_failed(self, task: TaskAssignment):
        """Notify task failure"""
        await self._emit_callbacks(self._on_task_failed, task)

    @staticmethod
    async def _emit_callbacks(callbacks: list[Callable], *args):
        """Execute callback list, supporting both sync and async callbacks"""
        for callback in callbacks:
            try:
                result = callback(*args)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning("Callback execution failed: %s", e)


__all__ = [
    "DistributedScheduler",
    "RemoteExecutionError",
    "RemoteResponseError",
    "TaskStatus",
    "TaskAssignment",
    "TaskQueue",
    "WorkerPool",
    "LoadBalancer",
]
