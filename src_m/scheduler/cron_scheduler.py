"""定时任务调度器
支持 cron 表达式配置的定时任务。"""

import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import threading

logger = logging.getLogger(__name__)

CRON_AVAILABLE = False
try:
    from croniter import croniter
    CRON_AVAILABLE = True
except ImportError:
    pass


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """任务类型"""
    CONVERT = "convert"
    SCHEDULE = "schedule"
    CUSTOM = "custom"


@dataclass
class ScheduledTask:
    """定时任务"""
    id: str
    name: str
    task_type: TaskType
    cron_expr: str
    command: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    last_status: Optional[TaskStatus] = None
    created_at: datetime = None
    enabled: bool = True

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "task_type": self.task_type.value if isinstance(self.task_type, TaskType) else self.task_type,
            "cron_expr": self.cron_expr,
            "command": self.command,
            "status": self.status.value if isinstance(self.status, TaskStatus) else self.status,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_status": self.last_status.value if isinstance(self.last_status, TaskStatus) else self.last_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduledTask":
        if "task_type" in data and isinstance(data["task_type"], str):
            data["task_type"] = TaskType(data["task_type"])
        if "status" in data and isinstance(data["status"], str):
            data["status"] = TaskStatus(data["status"])
        if "last_status" in data and isinstance(data["last_status"], str):
            data["last_status"] = TaskStatus(data["last_status"])
        if "next_run" in data and data["next_run"]:
            data["next_run"] = datetime.fromisoformat(data["next_run"])
        if "last_run" in data and data["last_run"]:
            data["last_run"] = datetime.fromisoformat(data["last_run"])
        if "created_at" in data and data["created_at"]:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class CronScheduler:
    """Cron 定时任务调度器"""""

    def __init__(self, storage_path: Optional[Path] = None):
        self._tasks: Dict[str, ScheduledTask] = {}
        self._storage_path = storage_path or Path("~/.ppc10/scheduled_tasks.json").expanduser()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        if not CRON_AVAILABLE:
            logger.warning(
                "croniter 未安装，Cron 表达式解析功能将受限。\n"
                "安装命令: pip install croniter"
            )

        self._load_tasks()

    def add_task(
        self,
        name: str,
        task_type: TaskType,
        cron_expr: str,
        command: Dict[str, Any],
        enabled: bool = True,
    ) -> str:
        """添加定时任务"""
        task_id = str(uuid.uuid4())[:8]

        task = ScheduledTask(
            id=task_id,
            name=name,
            task_type=task_type,
            cron_expr=cron_expr,
            command=command,
            enabled=enabled,
        )

        if CRON_AVAILABLE:
            task.next_run = self._calculate_next_run(cron_expr)

        with self._lock:
            self._tasks[task_id] = task
            self._save_tasks()

        logger.info(f"定时任务已添加: {name} (ID: {task_id})")
        return task_id

    def remove_task(self, task_id: str) -> bool:
        """移除定时任务"""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                self._save_tasks()
                logger.info(f"定时任务已移除: {task_id}")
                return True
            return False

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """获取任务"""
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[ScheduledTask]:
        """列出所有任务"""
        return list(self._tasks.values())

    def enable_task(self, task_id: str) -> bool:
        """启用任务"""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].enabled = True
                if CRON_AVAILABLE:
                    self._tasks[task_id].next_run = self._calculate_next_run(
                        self._tasks[task_id].cron_expr
                    )
                self._save_tasks()
                return True
            return False

    def disable_task(self, task_id: str) -> bool:
        """禁用任务"""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].enabled = False
                self._tasks[task_id].next_run = None
                self._save_tasks()
                return True
            return False

    def run_task(self, task_id: str) -> bool:
        """立即执行任务"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False

            task.status = TaskStatus.RUNNING
            task.last_run = datetime.now()

        try:
            success = self._execute_command(task.command)
            with self._lock:
                task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
                task.last_status = task.status
                if CRON_AVAILABLE:
                    task.next_run = self._calculate_next_run(task.cron_expr)
                self._save_tasks()
            return success
        except Exception as e:
            logger.error(f"任务执行失败 {task_id}: {e}")
            with self._lock:
                task.status = TaskStatus.FAILED
                task.last_status = task.status
                self._save_tasks()
            return False

    def _calculate_next_run(self, cron_expr: str) -> Optional[datetime]:
        """计算下次执行时间"""
        if not CRON_AVAILABLE:
            return None

        try:
            cron = croniter(cron_expr, datetime.now())
            return cron.get_next(datetime)
        except Exception as e:
            logger.warning(f"Cron 表达式解析失败: '{cron_expr}': {e}")
            return None

    def _execute_command(self, command: Dict[str, Any]) -> bool:
        """执行命令"""
        try:
            task_type = command.get("type")
            params = command.get("params", {})

            if task_type == "convert":
                return self._execute_convert(params)
            elif task_type == "custom":
                return self._execute_custom(params)
            else:
                logger.warning(f"未知任务类型: {task_type}")
                return False

        except Exception as e:
            logger.error(f"命令执行失败: {e}")
            return False

    def _execute_convert(self, params: Dict[str, Any]) -> bool:
        """执行转换命令"""
        try:
            from src_m.cli.commands.convert import handle_convert
            handle_convert(
                input_dir=Path(params.get("input_dir", "")),
                output_dir=Path(params.get("output_dir", "")),
                voice=params.get("voice"),
                concurrency=params.get("concurrency"),
                preset=params.get("preset", "balanced"),
                resume=params.get("resume", False),
                checkpoint=None,
                timeout_multiplier=params.get("timeout_multiplier"),
                rate=params.get("rate"),
                recursive=params.get("recursive", False),
            )
            return True
        except Exception as e:
            logger.error(f"转换命令执行失败: {e}")
            return False

    def _execute_custom(self, params: Dict[str, Any]) -> bool:
        """执行自定义命令"""
        callback = params.get("callback")
        if callback and callable(callback):
            try:
                callback(params)
                return True
            except Exception as e:
                logger.error(f"自定义回调失败: {e}")
                return False
        return False

    def _save_tasks(self):
        """保存任务到文件"""
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            tasks_data = [task.to_dict() for task in self._tasks.values()]
            with open(self._storage_path, 'w', encoding='utf-8') as f:
                json.dump(tasks_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存任务失败: {e}")

    def _load_tasks(self):
        """从文件加载任务"""
        if not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, 'r', encoding='utf-8') as f:
                tasks_data = json.load(f)
                for task_dict in tasks_data:
                    task = ScheduledTask.from_dict(task_dict)
                    self._tasks[task.id] = task
                    if task.enabled and CRON_AVAILABLE:
                        task.next_run = self._calculate_next_run(task.cron_expr)
            logger.info(f"已加载{len(self._tasks)} 个定时任务")
        except Exception as e:
            logger.error(f"加载任务失败: {e}")

    def start(self):
        """启动调度器"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("定时任务调度器已启动")

    def stop(self):
        """停止调度器"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("定时任务调度器已停止")

    def _run_loop(self):
        """运行循环"""
        while self._running:
            try:
                now = datetime.now()
                with self._lock:
                    for task in self._tasks.values():
                        if task.enabled and task.next_run and task.next_run <= now:
                            threading.Thread(
                                target=self._run_and_update,
                                args=(task.id,),
                                daemon=True
                            ).start()

                threading.Event().wait(60)
            except Exception as e:
                logger.error(f"调度循环错误: {e}")
                threading.Event().wait(60)

    def _run_and_update(self, task_id: str):
        """运行并更新任务状态"""
        try:
            self.run_task(task_id)
        except Exception as e:
            logger.error(f"任务 {task_id} 执行失败: {e}")


_global_scheduler: Optional[CronScheduler] = None


def get_scheduler() -> CronScheduler:
    """获取全局调度器实例"""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = CronScheduler()
    return _global_scheduler
