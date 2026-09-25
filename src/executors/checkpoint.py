"""TTS 断点续传检查点管理器

保存和恢复任务状态，支持中断后继续执行
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _is_within(path: Path, root: Path) -> bool:
    """判断 ``path`` 解析后是否位于 ``root`` 之内（拒绝符号链接逃逸）。"""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _checkpoint_paths_within(data: "CheckpointData") -> bool:
    """校验检查点内所有任务文件路径均位于声明的 input/output 目录内。

    由 :meth:`CheckpointManager.load` 调用；文件系统/构建语义上任务文件
    必然在各自根目录之下，越界说明检查点被篡改（任意文件读写风险）。
    """
    in_root = Path(data.input_dir)
    out_root = Path(data.output_dir)
    for task in data.tasks.values():
        if not _is_within(Path(task.input_file), in_root):
            return False
        if not _is_within(Path(task.output_file), out_root):
            return False
    return True


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
    error: str | None = None
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
    tasks: dict[str, CheckpointTask] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """检查点管理器"""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self._data: CheckpointData | None = None

    def _count_status(self, tasks: dict[str, Any]) -> dict[str, int]:
        """统计任务状态计数（completed/failed/pending）。

        ``TaskStatus(str, Enum)`` 使得 ``TaskStatus.COMPLETED == "completed"`` 为真，
        因此无需再额外与字符串字面量比较。
        """
        counts = {"completed": 0, "failed": 0, "pending": 0}
        for task in tasks.values():
            task_status = task.status
            if task_status == TaskStatus.COMPLETED:
                counts["completed"] += 1
            elif task_status in (TaskStatus.FAILED, TaskStatus.QUARANTINED):
                counts["failed"] += 1
            else:
                counts["pending"] += 1
        return counts

    def _build_checkpoint_task(self, task_id: str, task: Any) -> CheckpointTask:
        """从任务对象构建 CheckpointTask。"""
        return CheckpointTask(
            task_id=task_id,
            input_file=str(task.input_file),
            output_file=str(task.output_file),
            voice=task.voice,
            text_len=task.text_len,
            status=task.status,
            attempts=task.attempts,
            error=task.error,
            created_at=task.created_at,
            updated_at=time.time(),
            no_audio_retries=getattr(task, "no_audio_retries", 0),
        )

    def create_checkpoint(self, input_dir: Path, output_dir: Path, voice: str, tasks: dict[str, Any]) -> CheckpointData:
        """创建检查点

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            voice: 语音模型
            tasks: 任务字典 {task_id: TTSTask}
        """
        now = datetime.now().isoformat()
        checkpoint_id = f"checkpoint_{int(time.time())}"

        counts = self._count_status(tasks)
        task_data = {task_id: self._build_checkpoint_task(task_id, task) for task_id, task in tasks.items()}

        self._data = CheckpointData(
            checkpoint_id=checkpoint_id,
            created_at=now,
            updated_at=now,
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            voice=voice,
            total_tasks=len(tasks),
            completed_tasks=counts["completed"],
            failed_tasks=counts["failed"],
            pending_tasks=counts["pending"],
            tasks=task_data,
        )

        return self._data

    def update_checkpoint(self, tasks: dict[str, Any]) -> CheckpointData:
        """更新检查点

        Args:
            tasks: 更新后的任务字典
        """
        if self._data is None:
            raise RuntimeError("检查点尚未创建，请先调用 create_checkpoint()")

        counts = self._count_status(tasks)

        for task_id, task in tasks.items():
            if task_id in self._data.tasks:
                checkpoint_task = self._data.tasks[task_id]
                checkpoint_task.status = task.status
                checkpoint_task.attempts = task.attempts
                checkpoint_task.error = task.error
                checkpoint_task.updated_at = time.time()
                checkpoint_task.no_audio_retries = getattr(task, "no_audio_retries", 0)
            else:
                self._data.tasks[task_id] = self._build_checkpoint_task(task_id, task)

        self._data.completed_tasks = counts["completed"]
        self._data.failed_tasks = counts["failed"]
        self._data.pending_tasks = counts["pending"]
        self._data.total_tasks = counts["completed"] + counts["failed"] + counts["pending"]
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
                        "no_audio_retries": task.no_audio_retries,
                    }
                    for task_id, task in self._data.tasks.items()
                },
                "metadata": self._data.metadata,
            }

            # Write to temp file first, then atomically rename
            temp_path = self.checkpoint_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
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
                if "temp_path" in locals() and temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            return False

    def load(self) -> CheckpointData | None:
        """从文件加载检查点

        Returns:
            检查点数据，如果文件不存在或格式错误则返回 None
        """
        if not self.checkpoint_path.exists():
            return None

        try:
            with open(self.checkpoint_path, encoding="utf-8") as f:
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
                    no_audio_retries=task_data.get("no_audio_retries", 0),
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
                metadata=data_dict.get("metadata", {}),
            )

            # 容错加固：拒绝路径越界的检查点。任务文件按构造必然位于
            # input_dir/output_dir 之内；被篡改的 JSON 令 resume 阶段读写
            # 任意路径时会在此被拦截（CWE-22 任意文件读写缓解）。
            if not _checkpoint_paths_within(self._data):
                logger.error(f"检查点路径越界，拒绝加载: {self.checkpoint_path}")
                self._data = None
                return None

            logger.info(f"检查点已加载: {self.checkpoint_path}")
            return self._data

        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return None

    def rebuild_from_cache(
        self,
        input_dir: Path,
        output_dir: Path,
        voice: str,
        cache_root: Path | None = None,
    ) -> CheckpointData | None:
        """根据已存在的 .cache 分段和 input/output 文件重建检查点。

        扫描 cache_root 下所有子目录，把对应 input 文件已存在 cache 但 output
        文件尚未生成（或不完整）的任务标记为 pending；output 已存在且非空的
        任务标记为 completed。

        Args:
            input_dir: 输入目录（含 .txt 文件）
            output_dir: 输出目录（含 .cache 子目录）
            voice: 语音模型
            cache_root: cache 根目录，默认 output_dir / ".cache"

        Returns:
            重建后的 CheckpointData，若没有任何可恢复任务则返回 None
        """
        from src.utils.files import quick_probe
        cache_root = cache_root or (output_dir / ".cache")
        if not cache_root.exists():
            logger.warning("未找到 cache 目录: %s", cache_root)
            return None

        tasks: dict[str, CheckpointTask] = {}
        completed = 0
        pending = 0
        now = time.time()

        for cache_subdir in sorted(cache_root.iterdir()):
            if not cache_subdir.is_dir():
                continue

            stem = cache_subdir.name
            input_file = input_dir / (stem + ".txt")
            if not input_file.exists():
                logger.debug("跳过无对应 input 的 cache: %s", cache_subdir)
                continue

            output_file = output_dir / (stem + ".mp3")
            task_id = stem

            # output 已存在且非空 -> 视为已完成
            if output_file.exists() and output_file.stat().st_size > 0:
                status = TaskStatus.COMPLETED
                completed += 1
            # cache 目录中存在任何段文件 -> 视为待处理，让 convert --resume 继续补全/合并
            elif any(cache_subdir.glob("*_seg_*.mp3")):
                status = TaskStatus.PENDING
                pending += 1
            else:
                continue

            tasks[task_id] = CheckpointTask(
                task_id=task_id,
                input_file=str(input_file),
                output_file=str(output_file),
                voice=voice,
                # 缓存重建为一次性扫描：text_len 用字节大小探测即可（执行阶段
                # 超时启发式仍按实际解码字符数计算，不受此近似影响）
                text_len=quick_probe(input_file)[1],
                status=status,
                attempts=0,
                error=None,
                created_at=now,
                updated_at=now,
                no_audio_retries=0,
            )

        if not tasks:
            logger.warning("未从 cache 中重建出任何任务")
            return None

        now_iso = datetime.now().isoformat()
        self._data = CheckpointData(
            checkpoint_id=f"rebuilt_checkpoint_{int(now)}",
            created_at=now_iso,
            updated_at=now_iso,
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            voice=voice,
            total_tasks=len(tasks),
            completed_tasks=completed,
            failed_tasks=0,
            pending_tasks=pending,
            tasks=tasks,
            metadata={"rebuilt_from_cache": str(cache_root)},
        )

        return self._data

    def get_pending_tasks(self) -> list[CheckpointTask]:
        """获取待处理任务（包括失败和隔离的任务，不包括运行中的）"""
        if self._data is None:
            return []

        return [task for task in self._data.tasks.values() if task.status in ("pending", "failed", "quarantined")]

    def get_completed_tasks(self) -> list[CheckpointTask]:
        """获取已完成任务"""
        if self._data is None:
            return []

        return [task for task in self._data.tasks.values() if task.status == "completed"]

    def get_failed_tasks(self) -> list[CheckpointTask]:
        """获取失败任务"""
        if self._data is None:
            return []

        return [task for task in self._data.tasks.values() if task.status in ("failed", "quarantined")]

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

    def get_summary(self) -> dict[str, Any]:
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
                self._data.completed_tasks / self._data.total_tasks * 100 if self._data.total_tasks > 0 else 0
            ),
            "checkpoint_file": str(self.checkpoint_path),
        }
