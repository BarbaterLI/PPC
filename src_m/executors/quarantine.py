
"""失败任务隔离机制
管理连续失败任务的隔离与重试
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class QuarantineStats:
    """隔离队列统计信息"""
    total_quarantined: int = 0
    total_retried: int = 0
    total_removed: int = 0
    current_size: int = 0
    max_capacity: int = 0
    oldest_task_age: float = 0.0


@dataclass
class QuarantinedTask:
    """被隔离的任务"""
    task_id: str
    task_data: Dict[str, Any]
    failure_count: int
    last_failure_time: float
    quarantine_time: float
    delay: float = 300.0
    _removed: bool = False

    def is_ready_for_retry(self) -> bool:
        """检查是否可以重试"""
        return time.time() >= self.quarantine_time + self.delay

    def get_retry_delay(self) -> float:
        """获取距离可重试的剩余时间（秒）"""
        now = time.time()
        ready_time = self.quarantine_time + self.delay
        return max(0.0, ready_time - now)


class QuarantineQueue:
    """隔离队列
    管理连续失败任务的隔离与重试
    """

    def __init__(
        self,
        delay: float = 300.0,
        max_failure_count: int = 3,
        capacity_ratio: float = 0.1
    ):
        self._delay = delay
        self._max_failure_count = max_failure_count
        self._capacity_ratio = capacity_ratio

        self._queue: deque[QuarantinedTask] = deque()
        self._task_index: Dict[str, QuarantinedTask] = {}

        self._stats = QuarantineStats()

    def _get_max_capacity(self, total_tasks: int) -> int:
        """计算最大容量"""
        return max(1, int(total_tasks * self._capacity_ratio))

    def add_quarantine(
        self,
        task_id: str,
        task_data: Dict[str, Any],
        failure_count: int,
        total_tasks: int
    ) -> bool:
        """将任务加入隔离队列
        
        Args:
            task_id: 任务 ID
            task_data: 任务数据
            failure_count: 失败次数
            total_tasks: 总任务数
            
        Returns:
            是否成功加入隔离队列
        """
        if failure_count < self._max_failure_count:
            return False

        max_capacity = self._get_max_capacity(total_tasks)
        self._stats.max_capacity = max_capacity

        if len(self._queue) >= max_capacity:
            self._stats.total_removed += 1
            return False

        if task_id in self._task_index:
            return False

        now = time.time()
        quarantined_task = QuarantinedTask(
            task_id=task_id,
            task_data=task_data,
            failure_count=failure_count,
            last_failure_time=now,
            quarantine_time=now,
            delay=self._delay
        )

        self._queue.append(quarantined_task)
        self._task_index[task_id] = quarantined_task
        self._stats.total_quarantined += 1
        self._stats.current_size = len(self._queue)

        return True

    def get_ready_tasks(self) -> List[QuarantinedTask]:
        """获取可以重试的任务"""
        ready_tasks = []

        for task in list(self._queue):
            if task._removed:
                continue
            if task.is_ready_for_retry():
                ready_tasks.append(task)
                task._removed = True
                self._task_index.pop(task.task_id, None)
                self._stats.total_retried += 1

        self._compact_queue()
        self._stats.current_size = len(self._queue) - sum(1 for t in self._queue if t._removed)

        return ready_tasks

    def _compact_queue(self) -> None:
        """定期清理已标记的任务"""
        self._queue = deque(t for t in self._queue if not t._removed)

    def remove_task(self, task_id: str) -> Optional[QuarantinedTask]:
        """移除任务
        
        Args:
            task_id: 任务 ID
            
        Returns:
            被移除的任务，如果不存在则返回 None
        """
        if task_id not in self._task_index:
            return None

        task = self._task_index.pop(task_id)

        try:
            self._queue.remove(task)
        except ValueError:
            pass

        self._stats.total_removed += 1
        self._stats.current_size = len(self._queue)

        return task

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        oldest_task_age = 0.0
        if self._queue:
            now = time.time()
            oldest_task = self._queue[0]
            oldest_task_age = now - oldest_task.quarantine_time

        return {
            "total_quarantined": self._stats.total_quarantined,
            "total_retried": self._stats.total_retried,
            "total_removed": self._stats.total_removed,
            "current_size": self._stats.current_size,
            "max_capacity": self._stats.max_capacity,
            "oldest_task_age": oldest_task_age,
            "delay": self._delay,
            "max_failure_count": self._max_failure_count,
            "capacity_ratio": self._capacity_ratio
        }

    def clear(self) -> None:
        """清空隔离队列"""
        self._queue.clear()
        self._task_index.clear()
        self._stats.current_size = 0

    def get_task(self, task_id: str) -> Optional[QuarantinedTask]:
        """获取任务
        
        Args:
            task_id: 任务 ID
            
        Returns:
            任务对象，如果不存在则返回 None
        """
        return self._task_index.get(task_id)

    def __len__(self) -> int:
        """获取队列长度"""
        return len(self._queue)

    def __bool__(self) -> bool:
        """检查队列是否非空"""
        return bool(self._queue)
