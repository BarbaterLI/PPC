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
from typing import Optional, Dict, Any, List, Callable

try:
    from aiohttp import ClientSession, ClientTimeout, FormData
except ImportError:
    raise RuntimeError("aiohttp 未安装，请运行: pip install aiohttp")

from ..config import PPC8Config
from .node_pool import NodePool, NodeInfo, NodeStatus

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
    """任务分配"""
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
        return {
            "task_id": self.task_id,
            "text": self.text[:50] + "..." if len(self.text) > 50 else self.text,
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


class MasterScheduler:
    """主控端任务调度器"""

    def __init__(
        self,
        config: PPC8Config,
        node_pool: NodePool,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        task_timeout: float = 300.0,
        load_balance_strategy: str = "round_robin",
        local_execution: bool = True,
    ):
        self.config = config
        self.node_pool = node_pool
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.task_timeout = task_timeout
        self.load_balance_strategy = load_balance_strategy
        self.local_execution = local_execution

        # 任务队列
        self._tasks: Dict[str, TaskAssignment] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()

        # HTTP 客户端
        self._http_session: Optional[ClientSession] = None

        # Worker 协程
        self._workers: List[asyncio.Task] = []
        self._num_workers = config.tts.concurrency  # 使用配置的并发数

        # 本地 TTS 引擎（用于主控端也执行任务）
        self._local_tts_engine = None

        # 统计
        self._total_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0

        # 回调
        self._on_task_complete: List[Callable] = []
        self._on_task_failed: List[Callable] = []

        # 锁
        self._lock = asyncio.Lock()

        logger.info(
            f"主控端调度器初始化: workers={self._num_workers}, "
            f"strategy={load_balance_strategy}, local_execution={local_execution}"
        )

    async def start(self):
        """启动调度器"""
        # 启动节点池
        await self.node_pool.start()

        # 启动本地 TTS 引擎（如果启用）
        if self.local_execution:
            from ..engines.tts_engine import TTSEngine
            self._local_tts_engine = TTSEngine(self.config)
            await self._local_tts_engine.initialize()

        # 启动 HTTP 客户端
        self._http_session = ClientSession(
            timeout=ClientTimeout(total=self.task_timeout)
        )

        # 启动 Worker 协程
        for i in range(self._num_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._workers.append(worker)

        logger.info(f"主控端调度器已启动，Worker 数量: {self._num_workers}")

    async def stop(self):
        """停止调度器"""
        # 等待所有任务完成
        await self._task_queue.join()

        # 停止 Worker
        for worker in self._workers:
            worker.cancel()

        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

        # 停止 HTTP 客户端
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        # 停止本地 TTS 引擎
        if self._local_tts_engine:
            await self._local_tts_engine.cleanup()
            self._local_tts_engine = None

        # 停止节点池
        await self.node_pool.stop()

        logger.info("主控端调度器已停止")

    async def submit_task(self, task: TaskAssignment) -> str:
        """提交任务"""
        async with self._lock:
            self._tasks[task.task_id] = task
            self._total_tasks += 1

        await self._task_queue.put(task.task_id)
        logger.debug(f"任务已提交: {task.task_id}")

        return task.task_id

    async def submit_batch(self, tasks: List[TaskAssignment]) -> List[str]:
        """批量提交任务"""
        task_ids = []
        for task in tasks:
            task_id = await self.submit_task(task)
            task_ids.append(task_id)
        return task_ids

    async def get_task_status(self, task_id: str) -> Optional[TaskAssignment]:
        """获取任务状态"""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, TaskAssignment]:
        """获取所有任务"""
        return self._tasks.copy()

    def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计"""
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
                # 从队列获取任务
                task_id = await self._task_queue.get()

                try:
                    task = self._tasks.get(task_id)
                    if task is None:
                        continue

                    # 执行任务
                    await self._execute_task(task)

                except Exception as e:
                    logger.error(f"Worker {worker_id} 处理任务失败: {e}")

                finally:
                    self._task_queue.task_done()

        except asyncio.CancelledError:
            logger.info(f"Worker {worker_id} 被取消")

    async def _execute_task(self, task: TaskAssignment):
        """执行单个任务"""
        task.status = TaskStatus.RUNNING
        task.attempts += 1
        task.started_at = datetime.now(timezone.utc)

        try:
            # 尝试远程节点执行
            if self._should_use_remote_node():
                success = await self._execute_on_remote_node(task)
                if success:
                    return

            # 回退到本地执行
            if self.local_execution:
                await self._execute_on_local_node(task)
                return

            # 无可用节点
            task.status = TaskStatus.FAILED
            task.error = "无可用节点且未启用本地执行"
            self._failed_tasks += 1
            await self._notify_task_failed(task)

        except Exception as e:
            task.error = str(e)
            logger.error(f"任务执行失败 {task.task_id}: {e}")

            # 重试逻辑
            if task.attempts < self.max_retries:
                task.status = TaskStatus.RETRYING
                logger.info(
                    f"任务将在 {self.retry_delay}s 后重试 "
                    f"({task.attempts}/{self.max_retries}): {task.task_id}"
                )
                await asyncio.sleep(self.retry_delay)
                await self._task_queue.put(task.task_id)
            else:
                task.status = TaskStatus.FAILED
                self._failed_tasks += 1
                await self._notify_task_failed(task)

    async def _execute_on_remote_node(self, task: TaskAssignment) -> bool:
        """在远程节点执行"""
        try:
            # 获取最佳节点
            node = await self.node_pool.get_best_node(self.load_balance_strategy)
            if node is None:
                logger.warning("无可用节点，回退到本地执行")
                return False

            task.assigned_node = node.node_id

            # 发送合成请求
            url = f"{node.base_url}/api/v1/synthesize"
            payload = {
                "text": task.text,
                "voice": task.voice,
                "rate": task.rate,
            }

            async with self._http_session.post(url, json=payload) as response:
                if response.status == 200:
                    # 读取音频数据
                    audio_data = await response.read()

                    # 保存到文件
                    task.output_path.parent.mkdir(parents=True, exist_ok=True)
                    task.output_path.write_bytes(audio_data)

                    # 更新统计
                    duration_header = response.headers.get("X-Duration", "0")
                    task.duration = float(duration_header)
                    task.completed_at = datetime.now(timezone.utc)
                    task.status = TaskStatus.COMPLETED
                    self._completed_tasks += 1

                    # 更新节点统计
                    await self.node_pool.update_node_stats(
                        node.node_id,
                        {
                            "total_requests": node.total_requests + 1,
                            "successful_requests": node.successful_requests + 1,
                            "current_concurrency": node.current_concurrency,
                        }
                    )

                    logger.info(f"任务完成 {task.task_id} (节点: {node.node_id})")
                    await self._notify_task_complete(task)
                    return True

                else:
                    error_data = await response.json()
                    raise RuntimeError(f"节点返回错误: {error_data.get('error', '未知错误')}")

        except Exception as e:
            logger.warning(f"远程节点执行失败 {task.task_id}: {e}")
            return False

    async def _execute_on_local_node(self, task: TaskAssignment):
        """在本地节点执行"""
        try:
            start_time = time.time()

            if self._local_tts_engine is None:
                raise RuntimeError("本地 TTS 引擎未初始化")

            task.assigned_node = "local"

            # 执行合成
            if self.config.tts.enable_segmentation and len(task.text) > self.config.tts.max_segment_length:
                result = await self._local_tts_engine.synthesize_segmented(task.text, task.output_path)
            else:
                result = await self._local_tts_engine.synthesize(task.text, task.output_path)

            task.duration = time.time() - start_time
            task.completed_at = datetime.now(timezone.utc)

            if result.success:
                task.status = TaskStatus.COMPLETED
                self._completed_tasks += 1
                logger.info(f"任务完成 {task.task_id} (本地执行)")
                await self._notify_task_complete(task)
            else:
                raise RuntimeError(result.error or "本地合成失败")

        except Exception as e:
            logger.error(f"本地执行失败 {task.task_id}: {e}")
            raise

    def _should_use_remote_node(self) -> bool:
        """判断是否应该使用远程节点"""
        available_nodes = len(self.node_pool.get_available_nodes())
        return available_nodes > 0

    async def _notify_task_complete(self, task: TaskAssignment):
        """通知任务完成"""
        for callback in self._on_task_complete:
            try:
                result = callback(task)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"任务完成回调执行失败: {e}")

    async def _notify_task_failed(self, task: TaskAssignment):
        """通知任务失败"""
        for callback in self._on_task_failed:
            try:
                result = callback(task)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"任务失败回调执行失败: {e}")
