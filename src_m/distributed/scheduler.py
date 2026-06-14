"""Unified distributed scheduler for PPC10 TTS cluster.

Merges MasterScheduler and NodePool functionality into a single
coordinated scheduler that manages:
- Node pool management (add/remove/health check)
- Task scheduling and assignment
- Load balancing (pluggable strategies)
- Local fallback execution (only when no workers are available and
  ``local_fallback`` is enabled)

Design note: the scheduler is now a thin coordinator on top of the
:mod:`src_m.infrastructure.processing_unit` module. When a task is
submitted the scheduler forwards it to a worker's ``/api/v1/convert``
endpoint; the worker is the only place that actually runs TTS. This
means the master and worker share the same TTS execution code as
``ppc10 convert``.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from aiohttp import ClientSession, ClientTimeout

from src_m.config import PPC10Config
from src_m.distributed.node_pool import NodeInfo, NodePool, NodeStatus
from src_m.extensions.base import LoadBalanceStrategy
from src_m.infrastructure.processing_unit import (
    ConvertRequest,
    ConvertResult,
    MasterUnit,
    UnitRole,
    WorkerUnit,
    make_processing_unit,
)

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
    assigned_node: Optional[str] = None
    attempts: int = 0
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
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
        config: PPC10Config,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        task_timeout: float = 300.0,
        load_balance_strategy: str = "round_robin",
        local_execution: bool = True,
        health_check_interval: float = 30.0,
        health_check_timeout: float = 5.0,
        unhealthy_threshold: int = 3,
        custom_lb_strategy: Optional[LoadBalanceStrategy] = None,
        shutdown_timeout: float = 30.0,
        local_fallback: Optional[bool] = None,
        worker_unit: Optional[WorkerUnit] = None,
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

        self._tasks: Dict[str, TaskAssignment] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._http_session: Optional[ClientSession] = None
        self._workers: List[asyncio.Task] = []
        self._num_workers = config.tts.concurrency

        self._total_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0

        self._on_task_complete: List[Callable] = []
        self._on_task_failed: List[Callable] = []
        self._lock = asyncio.Lock()
        self._counter_lock = asyncio.Lock()

        self._custom_lb_strategy = custom_lb_strategy

        # Optional injected local worker used for fallback. When the
        # caller does not provide one we lazily create it in :meth:`start`.
        self._local_worker_unit: Optional[WorkerUnit] = worker_unit

        logger.info(
            "DistributedScheduler initialized: workers=%s, strategy=%s, local_execution=%s, local_fallback=%s",
            self._num_workers, load_balance_strategy, local_execution, self.local_fallback,
        )

    @property
    def node_pool(self) -> NodePool:
        """Access to the underlying node pool for node management."""
        return self._node_pool

    async def start(self):
        """Start the scheduler"""
        await self._node_pool.start()

        self._http_session = ClientSession(
            timeout=ClientTimeout(total=self.task_timeout)
        )

        for i in range(self._num_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._workers.append(worker)

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

        logger.info("DistributedScheduler started with %d workers", self._num_workers)

    async def stop(self):
        """Stop the scheduler"""
        try:
            await asyncio.wait_for(self._task_queue.join(), timeout=self.shutdown_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Queue drain timed out after %ss, cancelling pending tasks",
                self.shutdown_timeout,
            )

        for worker in self._workers:
            worker.cancel()

        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

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

        logger.info("DistributedScheduler stopped")

    async def submit_task(self, task: TaskAssignment) -> str:
        """Submit a single task"""
        async with self._lock:
            self._tasks[task.task_id] = task
            self._total_tasks += 1

        await self._task_queue.put(task.task_id)
        logger.debug("Task submitted: %s", task.task_id)

        return task.task_id

    async def submit_batch(self, tasks: List[TaskAssignment]) -> List[str]:
        """Submit multiple tasks in batch"""
        submitted_task_ids: List[str] = []
        try:
            for task in tasks:
                task_id = await self.submit_task(task)
                submitted_task_ids.append(task_id)
        except Exception as e:
            for submitted_id in submitted_task_ids:
                del self._tasks[submitted_id]
                self._total_tasks -= 1
            logger.warning("Batch submission failed, rolled back %d tasks: %s", len(submitted_task_ids), e)
            raise
        return submitted_task_ids

    async def get_task_status(self, task_id: str) -> Optional[TaskAssignment]:
        """Get task status by ID"""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, TaskAssignment]:
        """Get all tasks"""
        return self._tasks.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            "total_tasks": self._total_tasks,
            "completed_tasks": self._completed_tasks,
            "failed_tasks": self._failed_tasks,
            "pending_tasks": self._task_queue.qsize(),
            "active_workers": len([w for w in self._workers if not w.done()]),
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
        logger.info(f"Custom load balance strategy set: {strategy.get_name()}")

    async def add_node(
        self,
        host: str,
        port: int,
        node_id: Optional[str] = None,
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

    # ----------------------------------------------------------- workers

    async def _worker_loop(self, worker_id: str):
        """Worker main loop"""
        try:
            while True:
                task_id = await self._task_queue.get()

                try:
                    task = self._tasks.get(task_id)
                    if task is not None:
                        await self._execute_task(task)
                except asyncio.CancelledError:
                    logger.info("Worker %s: task %s cancelled", worker_id, task_id)
                    self._task_queue.task_done()
                    raise
                except Exception as e:
                    error_type = type(e).__name__
                    is_retryable = isinstance(e, (ConnectionError, TimeoutError, OSError))
                    logger.error(
                        "Worker %s failed to process task %s [%s] (retryable=%s): %s",
                        worker_id, task_id, error_type, is_retryable, e,
                        exc_info=True,
                    )
                    if task is not None:
                        task.error = f"{error_type}: {e}"
                finally:
                    self._task_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Worker %s cancelled", worker_id)

    async def _execute_task(self, task: TaskAssignment):
        """Execute a single task by forwarding it to a worker (or running locally)."""
        task.status = TaskStatus.RUNNING
        task.attempts += 1
        task.started_at = datetime.now(timezone.utc)

        try:
            if self._should_use_remote_node():
                selected_node = await self._select_best_node()
                if selected_node is not None:
                    task.assigned_node = selected_node.node_id
                success = await self._execute_on_remote_node(task, selected_node)
                if success:
                    return
                if selected_node is not None:
                    latest_node = self._node_pool.get_node(selected_node.node_id)
                    if latest_node is not None:
                        await self._node_pool.update_node_stats(
                            latest_node.node_id,
                            {
                                "total_requests": latest_node.total_requests + 1,
                                "successful_requests": latest_node.successful_requests,
                                "failed_requests": latest_node.failed_requests + 1,
                                "current_concurrency": latest_node.current_concurrency,
                            },
                        )

            if self.local_fallback:
                await self._execute_on_local_node(task)
                return

            task.status = TaskStatus.FAILED
            task.error = "No available nodes and local fallback disabled"
            async with self._counter_lock:
                self._failed_tasks += 1
            await self._notify_task_failed(task)

        except Exception as e:
            task.error = str(e)
            logger.error("Task execution failed %s: %s", task.task_id, e)

            if task.attempts < self.max_retries:
                task.status = TaskStatus.RETRYING
                logger.info(
                    "Task will retry in %ss (%s/%s): %s",
                    self.retry_delay, task.attempts, self.max_retries, task.task_id,
                )
                await asyncio.sleep(self.retry_delay)
                await self._task_queue.put(task.task_id)
            else:
                task.status = TaskStatus.FAILED
                async with self._counter_lock:
                    self._failed_tasks += 1
                await self._notify_task_failed(task)

    async def _execute_on_remote_node(self, task: TaskAssignment, node: Optional[NodeInfo] = None) -> bool:
        """Forward the task to a remote node and stream the audio back.

        For backwards compatibility with the legacy single-text API
        (which used ``/api/v1/synthesize``) the request is sent as a
        single-text payload to that endpoint. The full convert flow
        (multiple files / directory handling) should use
        :meth:`forward_convert` directly.
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
                    error_data = await response.json()
                    raise RuntimeError(f"Node error: {error_data.get('error', 'Unknown error')}")

                audio_data = await response.read()

                task.output_path.parent.mkdir(parents=True, exist_ok=True)
                task.output_path.write_bytes(audio_data)

                duration_header = response.headers.get("X-Duration", "0")
                task.duration = float(duration_header)
                task.completed_at = datetime.now(timezone.utc)
                task.status = TaskStatus.COMPLETED
                async with self._counter_lock:
                    self._completed_tasks += 1

                await self._node_pool.update_node_stats(
                    node.node_id,
                    {
                        "total_requests": 1,
                        "successful_requests": 1,
                        "current_concurrency": node.current_concurrency,
                    },
                )

                logger.info("Task completed %s (node: %s)", task.task_id, node.node_id)
                await self._notify_task_complete(task)
                return True

        except Exception as e:
            logger.warning("Remote node execution failed %s: %s", task.task_id, e)
            return False

    async def _execute_on_local_node(self, task: TaskAssignment):
        """Execute task on the local fallback worker.

        This is the only place where a master can run TTS itself, and
        it only happens when ``local_fallback`` is enabled. The work is
        delegated to the shared :class:`WorkerUnit`, which uses the
        same :class:`TTSExecutor` that ``ppc10 convert`` uses. Since
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

            with tempfile.TemporaryDirectory(prefix="ppc10-sched-") as tmpdir:
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
                async with self._counter_lock:
                    self._completed_tasks += 1
                logger.info("Task completed %s (local execution)", task.task_id)
                await self._notify_task_complete(task)
        except Exception as e:
            logger.error("Local execution failed %s: %s", task.task_id, e)
            raise

    async def _select_best_node(self) -> Optional[NodeInfo]:
        """Select the best node using the configured or custom strategy.

        This is the strategy injection point that allows custom load
        balancing strategies to be used.
        """
        if self._custom_lb_strategy is not None:
            available = self._node_pool.get_available_nodes()
            if not available:
                return None

            task_context = {
                "strategy": self.load_balance_strategy_name,
            }
            return await self._custom_lb_strategy.select_node(available, task_context)

        return await self._node_pool.get_best_node(self.load_balance_strategy_name)

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
    async def _emit_callbacks(callbacks: List[Callable], *args):
        """Execute callback list, supporting both sync and async callbacks"""
        for callback in callbacks:
            try:
                result = callback(*args)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning("Callback execution failed: %s", e)


__all__ = ["DistributedScheduler", "TaskStatus", "TaskAssignment"]
