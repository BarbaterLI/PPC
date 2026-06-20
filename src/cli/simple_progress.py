"""Simple progress handler - Display only key information."""

import time
from dataclasses import dataclass

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from .design.layouts import TaskDashboardLayout
from .output import OutputFormatter


@dataclass
class SimpleTaskInfo:
    """Simple task info."""

    name: str
    status: str = "pending"
    start_time: float | None = None
    end_time: float | None = None
    error: str | None = None


class SimpleProgressHandler:
    """Simple progress handler - shows only task name, totals, completed, failed.

    Spec 6 之后使用 :class:`src.cli.output.OutputFormatter` 与
    :class:`src.cli.design.layouts.TaskDashboardLayout` 渲染实时看板，
    不再依赖旧的 ``CLIUI`` / ``UIMode``。
    """

    def __init__(self, total_tasks: int, formatter: OutputFormatter | None = None):
        self.total_tasks = total_tasks
        self._original_total = total_tasks
        self.completed = 0
        self.failed = 0
        self.task_infos: dict[str, SimpleTaskInfo] = {}
        self.current_task: str | None = None
        self._start_time: float | None = None
        self._live: Live | None = None
        self.formatter = formatter

    def start(self):
        """Start progress display."""
        self._start_time = time.time()

        # json / quiet 模式不渲染 live 看板
        if self.formatter and self.formatter.mode != "human":
            return

        console = self.formatter.console if self.formatter else Console()
        self._live = Live(console=console, refresh_per_second=4)
        self._live.start()
        self._update_display()

    def stop(self):
        """Stop progress display."""
        if self._live:
            self._live.stop()
            self._live = None

    def register_task(self, task_id: str, name: str):
        """Register task."""
        self.task_infos[task_id] = SimpleTaskInfo(name=name)

    def set_total_tasks(self, n: int) -> None:
        """设置总任务数（用于 --one 模式预分段后设置段数）。"""
        self.total_tasks = n
        self._original_total = n
        self._update_display()

    def on_segment_complete(self, success: bool, error: str | None = None) -> None:
        """段级完成回调（不需 register_task，直接累加 completed/failed）。

        用于 --one 模式下逐段汇报合成结果。
        """
        if success:
            self.completed += 1
        else:
            self.failed += 1
        self._update_display()

    def on_task_start(self, task_id: str):
        """Task start callback."""
        if task_id in self.task_infos:
            info = self.task_infos[task_id]
            info.status = "running"
            info.start_time = time.time()
            self.current_task = task_id
        self._update_display()

    def on_task_complete(self, task_id: str, success: bool, error: str | None = None):
        """Task complete callback."""
        if task_id in self.task_infos:
            info = self.task_infos[task_id]
            info.status = "completed" if success else "failed"
            info.error = error
            info.end_time = time.time()

            if success:
                self.completed += 1
            else:
                self.failed += 1
        self._update_display()

    def on_retry(self, task_id: str, attempt: int, error: str, delay: float):
        """Retry callback."""
        if task_id in self.task_infos:
            info = self.task_infos[task_id]
            info.status = "retrying"
            info.error = error
        self._update_display()

    def _update_display(self):
        """Update display."""
        if not self._live:
            return

        elapsed = time.time() - self._start_time if self._start_time else 0
        processed = self.completed + self.failed
        if processed > self.total_tasks:
            self.total_tasks = processed
        remaining = self.total_tasks - processed

        speed = self.completed / elapsed if elapsed > 0 and self.completed > 0 else 0

        eta = 0.0
        if remaining > 0 and speed > 0:
            eta = remaining / speed

        current_name = None
        if self.current_task:
            current_info = self.task_infos.get(self.current_task)
            if current_info and current_info.status == "running":
                current_name = current_info.name

        layout = TaskDashboardLayout(
            total=self.total_tasks,
            completed=self.completed,
            failed=self.failed,
            current_task=current_name,
            speed=speed,
            elapsed=elapsed,
            eta=eta,
        )

        panel = Panel(
            layout.to_rich(),
            title="[bold green]TTS 转换中[/bold green]",
            border_style="green",
            expand=False,
        )
        self._live.update(panel)

    def get_stats(self) -> dict:
        """Get statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        speed = self.completed / elapsed if elapsed > 0 and self.completed > 0 else 0
        processed = self.completed + self.failed
        effective_total = max(self.total_tasks, processed)

        return {
            "total": effective_total,
            "completed": self.completed,
            "failed": self.failed,
            "elapsed": elapsed,
            "success_rate": (self.completed / effective_total * 100) if effective_total > 0 else 0,
            "current_speed": speed,
        }

    def get_detailed_stats(self) -> dict:
        """Get detailed stats (compatible interface)."""
        return self.get_stats()
