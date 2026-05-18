"""Unified distributed scheduler for PPC9 TTS cluster.

Merges MasterScheduler and NodePool functionality into a single
coordinated scheduler that manages:
- Node pool management (add/remove/health check)
- Task scheduling and assignment
- Load balancing (pluggable strategies)
- Local and remote execution
- Strategy injection points for custom extensions
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

from src_m.config import PPC9Config
from src_m.distributed.node_pool import NodeInfo, NodePool, NodeStatus
from src_m.extensions.base import LoadBalanceStrategy

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
    """Task assignment information"""
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
    
    Combines task scheduling and node pool management into a single
    coordinator that supports pluggable strategies for load balancing
    and health checking.
    """

    def __init__(
        self,
        config: PPC9Config,
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
    ):
        self.config = config
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.task_timeout = task_timeout
        self.load_balance_strategy_name = load_balance_strategy
        self.local_execution = local_execution
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
        self._local_tts_engine = None

        self._total_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0

        self._on_task_complete: List[Callable] = []
        self._on_task_failed: List[Callable] = []
        self._lock = asyncio.Lock()
        self._counter_lock = asyncio.Lock()

        self._custom_lb_strategy = custom_lb_strategy

        logger.info(
            "DistributedScheduler initialized: workers=%s, strategy=%s, local_execution=%s",
            self._num_workers, load_balance_strategy, local_execution,
        )

    @property
    def node_pool(self) -> NodePool:
        """Access to the underlying node pool for node management."""
        return self._node_pool

    async def start(self):
        """Start the scheduler"""
        await self._node_pool.start()

        if self.local_execution:
            from src_m.engines.tts_engine import TTSEngine
            self._local_tts_engine = TTSEngine(self.config)
            await self._local_tts_engine.initialize()

        self._http_session = ClientSession(
            timeout=ClientTimeout(total=self.task_timeout)
        )

        for i in range(self._num_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._workers.append(worker)

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

        if self._local_tts_engine:
            await self._local_tts_engine.cleanup()
            self._local_tts_engine = None

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
        """Execute a single task"""
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

            if self.local_execution:
                await self._execute_on_local_node(task)
                return

            task.status = TaskStatus.FAILED
            task.error = "No available nodes and local execution disabled"
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
        """Execute task on a remote node"""
        try:
            if node is None:
                node = await self._select_best_node()
            if node is None:
                logger.warning("No available nodes, falling back to local execution")
                return False

            task.assigned_node = node.node_id

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
                        "total_requests": node.total_requests + 1,
                        "successful_requests": node.successful_requests + 1,
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
        """Execute task on local node"""
        if self._local_tts_engine is None:
            raise RuntimeError("Local TTS engine not initialized")

        task.assigned_node = "local"
        start_time = time.time()

        try:
            should_segment = (
                self.config.tts.enable_segmentation
                and len(task.text) > self.config.tts.max_segment_length
            )

            if should_segment:
                result = await self._local_tts_engine.synthesize_segmented(
                    task.text, task.output_path
                )
            else:
                result = await self._local_tts_engine.synthesize(task.text, task.output_path)

            task.duration = time.time() - start_time
            task.completed_at = datetime.now(timezone.utc)

            if result.success:
                task.status = TaskStatus.COMPLETED
                async with self._counter_lock:
                    self._completed_tasks += 1
                logger.info("Task completed %s (local execution)", task.task_id)
                await self._notify_task_complete(task)
            else:
                raise RuntimeError(result.error or "Local synthesis failed")

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
