
"""TTS 断点续传检查点管理器

保存和恢复任务状态，支持中断后继续执行
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    QUARANTINED = "quarantined"


@dataclass
class CheckpointTask:
    """检查点任务"""
    task_id: str
    input_file: str
    output_file: str
    voice: str
    text_len: int
    status: str
    attempts: int
    error: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    no_audio_retries: int = 0


@dataclass
class CheckpointData:
    """检查点数据"""
    checkpoint_id: str
    created_at: str
    updated_at: str
    input_dir: str
    output_dir: str
    voice: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    pending_tasks: int
    tasks: Dict[str, CheckpointTask] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """检查点管理器"""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self._data: Optional[CheckpointData] = None

    def create_checkpoint(
        self,
        input_dir: Path,
        output_dir: Path,
        voice: str,
        tasks: Dict[str, Any]
    ) -> CheckpointData:
        """创建检查点
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            voice: 语音模型
            tasks: 任务字典 {task_id: TTSTask}
        """
        now = datetime.now().isoformat()
        checkpoint_id = f"checkpoint_{int(time.time())}"

        task_data = {}
        completed = failed = pending = 0

        for task_id, task in tasks.items():
            task_status = task.status
            if task_status == TaskStatus.COMPLETED or task_status == "completed":
                completed += 1
            elif task_status in (TaskStatus.FAILED, TaskStatus.QUARANTINED, "failed", "quarantined"):
                failed += 1
            else:
                pending += 1

            task_data[task_id] = CheckpointTask(
                task_id=task_id,
                input_file=str(task.input_file),
                output_file=str(task.output_file),
                voice=task.voice,
                text_len=task.text_len,
                status=task_status,
                attempts=task.attempts,
                error=task.error,
                created_at=task.created_at,
                updated_at=time.time(),
                no_audio_retries=getattr(task, 'no_audio_retries', 0)
            )

        self._data = CheckpointData(
            checkpoint_id=checkpoint_id,
            created_at=now,
            updated_at=now,
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            voice=voice,
            total_tasks=len(tasks),
            completed_tasks=completed,
            failed_tasks=failed,
            pending_tasks=pending,
            tasks=task_data
        )

        return self._data

    def update_checkpoint(self, tasks: Dict[str, Any]) -> CheckpointData:
        """更新检查点
        
        Args:
            tasks: 更新后的任务字典
        """
        if self._data is None:
            raise RuntimeError("检查点尚未创建，请先调用 create_checkpoint()")

        completed = failed = pending = 0

        for task_id, task in tasks.items():
            task_status = task.status
            if task_status == TaskStatus.COMPLETED or task_status == "completed":
                completed += 1
            elif task_status in (TaskStatus.FAILED, TaskStatus.QUARANTINED, "failed", "quarantined"):
                failed += 1
            else:
                pending += 1

            if task_id in self._data.tasks:
                checkpoint_task = self._data.tasks[task_id]
                checkpoint_task.status = task_status
                checkpoint_task.attempts = task.attempts
                checkpoint_task.error = task.error
                checkpoint_task.updated_at = time.time()
                checkpoint_task.no_audio_retries = getattr(task, 'no_audio_retries', 0)
            else:
                self._data.tasks[task_id] = CheckpointTask(
                    task_id=task_id,
                    input_file=str(task.input_file),
                    output_file=str(task.output_file),
                    voice=task.voice,
                    text_len=task.text_len,
                    status=task_status,
                    attempts=task.attempts,
                    error=task.error,
                    created_at=task.created_at,
                    updated_at=time.time(),
                    no_audio_retries=getattr(task, 'no_audio_retries', 0)
                )

        self._data.completed_tasks = completed
        self._data.failed_tasks = failed
        self._data.pending_tasks = pending
        self._data.total_tasks = completed + failed + pending
        self._data.updated_at = datetime.now().isoformat()

        return self._data

    def save(self) -> bool:
        """保存检查点到文件
        
        Returns:
            是否保存成功
        """
        if self._data is None:
            logger.warning("没有检查点数据可保存")
            return False

        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            data_dict = {
                "checkpoint_id": self._data.checkpoint_id,
                "created_at": self._data.created_at,
                "updated_at": self._data.updated_at,
                "input_dir": self._data.input_dir,
                "output_dir": self._data.output_dir,
                "voice": self._data.voice,
                "total_tasks": self._data.total_tasks,
                "completed_tasks": self._data.completed_tasks,
                "failed_tasks": self._data.failed_tasks,
                "pending_tasks": self._data.pending_tasks,
                "tasks": {
                    task_id: {
                        "task_id": task.task_id,
                        "input_file": task.input_file,
                        "output_file": task.output_file,
                        "voice": task.voice,
                        "text_len": task.text_len,
                        "status": task.status,
                        "attempts": task.attempts,
                        "error": task.error,
                        "created_at": task.created_at,
                        "updated_at": task.updated_at,
                        "no_audio_retries": task.no_audio_retries
                    }
                    for task_id, task in self._data.tasks.items()
                },
                "metadata": self._data.metadata
            }

            # Write to temp file first, then atomically rename
            temp_path = self.checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=2)
            
            # Atomic rename (on Windows, os.replace will overwrite)
            import os
            os.replace(str(temp_path), str(self.checkpoint_path))

            logger.info(f"检查点已保存: {self.checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            # Clean up temp file if it exists
            try:
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            return False

    def load(self) -> Optional[CheckpointData]:
        """从文件加载检查点
        
        Returns:
            检查点数据，如果文件不存在或格式错误则返回 None
        """
        if not self.checkpoint_path.exists():
            return None

        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)

            tasks = {}
            for task_id, task_data in data_dict.get("tasks", {}).items():
                tasks[task_id] = CheckpointTask(
                    task_id=task_data["task_id"],
                    input_file=task_data["input_file"],
                    output_file=task_data["output_file"],
                    voice=task_data["voice"],
                    text_len=task_data["text_len"],
                    status=task_data["status"],
                    attempts=task_data["attempts"],
                    error=task_data.get("error"),
                    created_at=task_data.get("created_at", 0.0),
                    updated_at=task_data.get("updated_at", 0.0),
                    no_audio_retries=task_data.get("no_audio_retries", 0)
                )

            self._data = CheckpointData(
                checkpoint_id=data_dict["checkpoint_id"],
                created_at=data_dict["created_at"],
                updated_at=data_dict["updated_at"],
                input_dir=data_dict["input_dir"],
                output_dir=data_dict["output_dir"],
                voice=data_dict["voice"],
                total_tasks=data_dict["total_tasks"],
                completed_tasks=data_dict["completed_tasks"],
                failed_tasks=data_dict["failed_tasks"],
                pending_tasks=data_dict["pending_tasks"],
                tasks=tasks,
                metadata=data_dict.get("metadata", {})
            )

            logger.info(f"检查点已加载: {self.checkpoint_path}")
            return self._data

        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return None

    def get_pending_tasks(self) -> List[CheckpointTask]:
        """获取待处理任务（包括失败和隔离的任务，不包括运行中的）"""
        if self._data is None:
            return []

        return [
            task for task in self._data.tasks.values()
            if task.status in ("pending", "failed", "quarantined")
        ]

    def get_completed_tasks(self) -> List[CheckpointTask]:
        """获取已完成任务"""
        if self._data is None:
            return []

        return [
            task for task in self._data.tasks.values()
            if task.status == "completed"
        ]

    def get_failed_tasks(self) -> List[CheckpointTask]:
        """获取失败任务"""
        if self._data is None:
            return []

        return [
            task for task in self._data.tasks.values()
            if task.status in ("failed", "quarantined")
        ]

    def delete(self) -> bool:
        """删除检查点文件"""
        try:
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
                logger.info(f"检查点已删除: {self.checkpoint_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除检查点失败: {e}")
            return False

    def get_summary(self) -> Dict[str, Any]:
        """获取检查点摘要"""
        if self._data is None:
            return {}

        return {
            "checkpoint_id": self._data.checkpoint_id,
            "created_at": self._data.created_at,
            "updated_at": self._data.updated_at,
            "input_dir": self._data.input_dir,
            "output_dir": self._data.output_dir,
            "voice": self._data.voice,
            "total_tasks": self._data.total_tasks,
            "completed_tasks": self._data.completed_tasks,
            "failed_tasks": self._data.failed_tasks,
            "pending_tasks": self._data.pending_tasks,
            "progress_percent": (
                self._data.completed_tasks / self._data.total_tasks * 100
                if self._data.total_tasks > 0 else 0
            ),
            "checkpoint_file": str(self.checkpoint_path)
        }
