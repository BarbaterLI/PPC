"""简洁进度显示 - 仅展示关键信息"""

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
    """简单任务信息"""
    name: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None


class SimpleProgressHandler:
    """简洁进度处理器
    仅显示：
    - 当前处理的任务
    - 总任务数
    - 剩余任务数
    - 已完成数
    """
    
    def __init__(self, total_tasks: int, ui: Optional[CLIUI] = None):
        self.total_tasks = total_tasks
        self.completed = 0
        self.failed = 0
        self.task_infos: Dict[str, SimpleTaskInfo] = {}
        self.current_task: Optional[str] = None
        self._start_time: Optional[float] = None
        self._live: Optional[Live] = None
        self.ui = ui
        
    def start(self):
        """开始进度显示"""
        self._start_time = time.time()
        
        # 根据 UI 模式决定是否使用 Live 显示
        if self.ui and self.ui.config.mode == UIMode.CLASSIC:
            # 经典模式不使用 Live
            return
        
        self._live = Live(console=console, refresh_per_second=2)
        self._live.start()
        self._update_display()
    
    def stop(self):
        """停止进度显示"""
        if self._live:
            self._live.stop()
            self._live = None
    
    def register_task(self, task_id: str, name: str):
        """注册任务"""
        self.task_infos[task_id] = SimpleTaskInfo(name=name)
    
    def on_task_start(self, task_id: str):
        """任务开始回调"""
        if task_id in self.task_infos:
            info = self.task_infos[task_id]
            info.status = "running"
            info.start_time = time.time()
            self.current_task = task_id
        
        # 经典模式输出日志
        if self.ui and self.ui.config.mode == UIMode.CLASSIC:
            current_info = self.task_infos.get(task_id)
            if current_info:
                self.ui.tts_processing(current_info.name)
        
        self._update_display()
    
    def on_task_complete(self, task_id: str, success: bool, error: Optional[str] = None):
        """任务完成回调"""
        if task_id in self.task_infos:
            info = self.task_infos[task_id]
            info.status = "completed" if success else "failed"
            info.error = error
            info.end_time = time.time()
            
            if success:
                self.completed += 1
            else:
                self.failed += 1
            
            # 经典模式输出日志
            if self.ui and self.ui.config.mode == UIMode.CLASSIC:
                if success:
                    duration = info.end_time - info.start_time if info.start_time else 0
                    self.ui.tts_success(info.name, duration, 0)
                else:
                    self.ui.tts_failure(info.name, error or "Unknown error")
        
        self._update_display()
    
    def _update_display(self):
        """更新显示"""
        if not self._live:
            return
        
        elapsed = time.time() - self._start_time if self._start_time else 0
        remaining = self.total_tasks - self.completed - self.failed
        pending = self.total_tasks - self.completed - self.failed
        
        # 计算速度
        speed = self.completed / elapsed if elapsed > 0 and self.completed > 0 else 0
        
        # 计算 ETA
        eta = 0.0
        if remaining > 0 and speed > 0:
            eta = remaining / speed
        
        # 构建面板内容
        content_lines = []
        
        # 当前处理的任务
        if self.current_task:
            current_info = self.task_infos.get(self.current_task)
            if current_info and current_info.status == "running":
                content_lines.append(f"[bold cyan]⚡ 当前任务:[/bold cyan] {current_info.name}")
                content_lines.append("")
        
        # 统计信息
        content_lines.append(f"[bold]📊 任务统计:[/bold]")
        content_lines.append(f"  总任务数：   {self.total_tasks}")
        content_lines.append(f"  已完成：     [green]{self.completed}[/green]")
        content_lines.append(f"  剩余：       {remaining}")
        content_lines.append(f"  失败：       [red]{self.failed}[/red]")
        content_lines.append("")
        
        # 性能信息
        content_lines.append(f"[bold]⚡ 性能:[/bold]")
        content_lines.append(f"  速度：       {speed:.2f} 任务/秒")
        content_lines.append(f"  预计剩余：   {self._format_eta(eta)}")
        content_lines.append(f"  已用时间：   {self._format_duration(elapsed)}")
        
        # 成功率
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
        """格式化预计时间"""
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
        """格式化持续时间"""
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
        """根据成功率返回颜色"""
        if rate >= 95:
            return "green"
        elif rate >= 80:
            return "yellow"
        else:
            return "red"
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        elapsed = time.time() - self._start_time if self._start_time else 0
        speed = self.completed / elapsed if elapsed > 0 and self.completed > 0 else 0
        
        return {
            "total": self.total_tasks,
            "completed": self.completed,
            "failed": self.failed,
            "elapsed": elapsed,
            "success_rate": (self.completed / self.total_tasks * 100) if self.total_tasks > 0 else 0,
            "current_speed": speed,
        }
    
    def get_detailed_stats(self) -> Dict:
        """获取详细统计（兼容接口）"""
        return self.get_stats()
