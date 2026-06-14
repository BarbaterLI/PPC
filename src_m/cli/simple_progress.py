"""Simple progress handler - Display only key information."""

import time
from typing import Dict, Optional, Any
from dataclasses import dataclass
from collections import deque

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

from .ui import CLIUI, UIMode


console = Console()


@dataclass
class SimpleTaskInfo:
    """Simple task info."""
    name: str
    status: str = "pending"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None


class SimpleProgressHandler:
    """Simple progress handler - shows only task name, totals, completed, failed."""

    def __init__(self, total_tasks: int, ui: Optional[CLIUI] = None):
        self.total_tasks = total_tasks
        self._original_total = total_tasks
        self.completed = 0
        self.failed = 0
        self.task_infos: Dict[str, SimpleTaskInfo] = {}
        self.current_task: Optional[str] = None
        self._start_time: Optional[float] = None
        self._live: Optional[Live] = None
        self.ui = ui

    def start(self):
        """Start progress display."""
        self._start_time = time.time()

        if self.ui and self.ui.config.mode == UIMode.CLASSIC:
            return

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

    def on_segment_complete(self, success: bool, error: Optional[str] = None) -> None:
        """段级完成回调（不需 register_task，直接累加 completed/failed）。

        用于 --one 模式下逐段汇报合成结果。
        """
        if success:
            self.completed += 1
        else:
            self.failed += 1
        if self.ui and self.ui.config.mode == UIMode.CLASSIC:
            if not success:
                short = (error or "未知错误")[:80]
                console.print(f"[red]❌ 段合成失败: {short}[/red]")
        self._update_display()

    def on_task_start(self, task_id: str):
        """Task start callback."""
        if task_id in self.task_infos:
            info = self.task_infos[task_id]
            info.status = "running"
            info.start_time = time.time()
            self.current_task = task_id

        if self.ui and self.ui.config.mode == UIMode.CLASSIC:
            current_info = self.task_infos.get(task_id)
            if current_info:
                self.ui.tts_processing(current_info.name)

        self._update_display()

    def on_task_complete(self, task_id: str, success: bool, error: Optional[str] = None):
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

            if self.ui and self.ui.config.mode == UIMode.CLASSIC:
                if success:
                    duration = info.end_time - info.start_time if info.start_time else 0
                    self.ui.tts_success(info.name, duration, 0)
                else:
                    self.ui.tts_failure(info.name, error or "Unknown error")

        self._update_display()

    def on_retry(self, task_id: str, attempt: int, error: str, delay: float):
        """Retry callback."""
        if task_id in self.task_infos:
            info = self.task_infos[task_id]
            info.status = "retrying"
            info.error = error

        if self.ui and self.ui.config.mode == UIMode.CLASSIC:
            current_info = self.task_infos.get(task_id)
            if current_info:
                console.print(f"[yellow]⚠️  任务重试: {current_info.name} (尝试 {attempt}, {delay:.1f}s 后): {error}[/yellow]")

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

        content_lines = []

        if self.current_task:
            current_info = self.task_infos.get(self.current_task)
            if current_info and current_info.status == "running":
                content_lines.append(f"[bold cyan]⚡ 当前任务:[/bold cyan] {current_info.name}")
                content_lines.append("")

        content_lines.append(f"[bold]📊 任务统计:[/bold]")
        content_lines.append(f"  总任务数：   {self.total_tasks}")
        content_lines.append(f"  已完成：     [green]{self.completed}[/green]")
        content_lines.append(f"  剩余：       {remaining}")
        content_lines.append(f"  失败：       [red]{self.failed}[/red]")
        content_lines.append("")

        content_lines.append(f"[bold]⚡ 性能:[/bold]")
        content_lines.append(f"  速度：       {speed:.2f} 任务/秒")
        content_lines.append(f"  预计剩余：   {self._format_eta(eta)}")
        content_lines.append(f"  已用时间：   {self._format_duration(elapsed)}")

        if self.completed + self.failed > 0:
            success_rate = (self.completed / (self.completed + self.failed)) * 100
            content_lines.append("")
            content_lines.append(f"[bold]📈 成功率:[/bold] [{self._get_rate_color(success_rate)}]{success_rate:.1f}%[/{self._get_rate_color(success_rate)}]")

        panel = Panel(
            "\n".join(content_lines),
            title="[bold green]🎤 TTS 转换中[/bold green]",
            border_style="green",
            expand=False
        )

        self._live.update(panel)

    def _format_eta(self, eta: float) -> str:
        """Format ETA."""
        if eta == 0:
            return "计算中..."
        elif eta < 60:
            return f"{eta:.0f}秒"
        elif eta < 3600:
            minutes = int(eta // 60)
            seconds = int(eta % 60)
            return f"{minutes}分{seconds}秒"
        else:
            hours = int(eta // 3600)
            minutes = int((eta % 3600) // 60)
            return f"{hours}小时{minutes}分"

    def _format_duration(self, seconds: float) -> str:
        """Format duration."""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}分{secs}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}小时{minutes}分"

    def _get_rate_color(self, rate: float) -> str:
        """Get color based on success rate."""
        if rate >= 95:
            return "green"
        elif rate >= 80:
            return "yellow"
        else:
            return "red"

    def get_stats(self) -> Dict:
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

    def get_detailed_stats(self) -> Dict:
        """Get detailed stats (compatible interface)."""
        return self.get_stats()
