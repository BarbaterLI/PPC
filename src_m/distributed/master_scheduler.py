"""主控端任务调度器
负责任务分配、负载均衡、故障转移
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
from src_m.distributed.node_pool import NodeInfo, NodePool

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class TaskAssignment:
    """任务分配信息"""
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
        """转换为字典格式"""
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
        """截断文本用于展示"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."


class MasterScheduler:
    """主控端任务调度器
    负责任务的提交、分配、执行监控和故障转移
    """

    def __init__(
        self,
        config: PPC9Config,
        node_pool: NodePool,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        task_timeout: float = 300.0,
        load_balance_strategy: str = "round_robin",
        local_execution: bool = True,
        owns_node_pool: bool = False,
    ):
        self.config = config
        self.node_pool = node_pool
        self.owns_node_pool = owns_node_pool
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.task_timeout = task_timeout
        self.load_balance_strategy = load_balance_strategy
        self.local_execution = local_execution

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

        logger.info(
            "主控端调度器初始化: workers=%s, strategy=%s, local_execution=%s",
            self._num_workers, load_balance_strategy, local_execution,
        )

    async def start(self):
        """启动调度器"""
        await self.node_pool.start()

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

        logger.info("主控端调度器已启动，Worker 数量: %s", self._num_workers)

    async def stop(self):
        """停止调度器"""
        try:
            await asyncio.wait_for(self._task_queue.join(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("等待任务队列完成超时")

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

        if self.owns_node_pool:
            await self.node_pool.stop()

        logger.info("主控端调度器已停止")

    async def submit_task(self, task: TaskAssignment) -> str:
        """提交单个任务"""
        async with self._lock:
            self._tasks[task.task_id] = task
            self._total_tasks += 1

        await self._task_queue.put(task.task_id)
        logger.debug("任务已提交: %s", task.task_id)

        return task.task_id

    async def submit_batch(self, tasks: List[TaskAssignment]) -> List[str]:
        """批量提交任务"""
        return [await self.submit_task(task) for task in tasks]

    async def get_task_status(self, task_id: str) -> Optional[TaskAssignment]:
        """获取任务状态"""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, TaskAssignment]:
        """获取所有任务"""
        return self._tasks.copy()

    def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        return {
            "total_tasks": self._total_tasks,
            "completed_tasks": self._completed_tasks,
            "failed_tasks": self._failed_tasks,
            "pending_tasks": self._task_queue.qsize(),
            "active_workers": len([w for w in self._workers if not w.done()]),
            "node_pool": self.node_pool.get_stats(),
        }

    def on_task_complete(self, callback: Callable):
        """注册任务完成回调"""
        self._on_task_complete.append(callback)

    def on_task_failed(self, callback: Callable):
        """注册任务失败回调"""
        self._on_task_failed.append(callback)

    async def _worker_loop(self, worker_id: str):
        """Worker 主循环"""
        try:
            while True:
                task_id = await self._task_queue.get()

                try:
                    task = self._tasks.get(task_id)
                    if task is not None:
                        await self._execute_task(task)
                except Exception as e:
                    logger.error("Worker %s 处理任务失败: %s", worker_id, e)
                finally:
                    self._task_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Worker %s 被取消", worker_id)

    async def _execute_task(self, task: TaskAssignment):
        """执行单个任务"""
        task.status = TaskStatus.RUNNING
        task.attempts += 1
        task.started_at = datetime.now(timezone.utc)

        try:
            if self._should_use_remote_node():
                success = await self._execute_on_remote_node(task)
                if success:
                    return

            if self.local_execution:
                await self._execute_on_local_node(task)
                return

            task.status = TaskStatus.FAILED
            task.error = "无可用节点且未启用本地执行"
            self._failed_tasks += 1
            await self._notify_task_failed(task)

        except Exception as e:
            task.error = str(e)
            logger.error("任务执行失败 %s: %s", task.task_id, e)

            if task.attempts < self.max_retries:
                task.status = TaskStatus.RETRYING
                logger.info(
                    "任务将在 %ss 后重试 (%s/%s): %s",
                    self.retry_delay, task.attempts, self.max_retries, task.task_id,
                )
                await asyncio.sleep(self.retry_delay)
                await self._task_queue.put(task.task_id)
            else:
                task.status = TaskStatus.FAILED
                self._failed_tasks += 1
                await self._notify_task_failed(task)

    async def _execute_on_remote_node(self, task: TaskAssignment) -> bool:
        """在远程节点执行任务"""
        try:
            node = await self.node_pool.get_best_node(self.load_balance_strategy)
            if node is None:
                logger.warning("无可用节点，回退到本地执行")
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
                    raise RuntimeError(f"节点返回错误: {error_data.get('error', '未知错误')}")

                audio_data = await response.read()

                task.output_path.parent.mkdir(parents=True, exist_ok=True)
                task.output_path.write_bytes(audio_data)

                duration_header = response.headers.get("X-Duration", "0")
                task.duration = float(duration_header)
                task.completed_at = datetime.now(timezone.utc)
                task.status = TaskStatus.COMPLETED
                self._completed_tasks += 1

                await self.node_pool.update_node_stats(
                    node.node_id,
                    {
                        "total_requests": node.total_requests + 1,
                        "successful_requests": node.successful_requests + 1,
                        "current_concurrency": node.current_concurrency,
                    },
                )

                logger.info("任务完成 %s (节点: %s)", task.task_id, node.node_id)
                await self._notify_task_complete(task)
                return True

        except Exception as e:
            logger.warning("远程节点执行失败 %s: %s", task.task_id, e)
            return False

    async def _execute_on_local_node(self, task: TaskAssignment):
        """在本地节点执行任务"""
        if self._local_tts_engine is None:
            raise RuntimeError("本地 TTS 引擎未初始化")

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
                self._completed_tasks += 1
                logger.info("任务完成 %s (本地执行)", task.task_id)
                await self._notify_task_complete(task)
            else:
                raise RuntimeError(result.error or "本地合成失败")

        except Exception as e:
            logger.error("本地执行失败 %s: %s", task.task_id, e)
            raise

    def _should_use_remote_node(self) -> bool:
        """判断是否应该使用远程节点"""
        return len(self.node_pool.get_available_nodes()) > 0

    async def _notify_task_complete(self, task: TaskAssignment):
        """通知任务完成"""
        await self._emit_callbacks(self._on_task_complete, task)

    async def _notify_task_failed(self, task: TaskAssignment):
        """通知任务失败"""
        await self._emit_callbacks(self._on_task_failed, task)

    @staticmethod
    async def _emit_callbacks(callbacks: List[Callable], *args):
        """执行回调列表，支持同步和异步回调"""
        for callback in callbacks:
            try:
                result = callback(*args)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning("回调执行失败: %s", e)
