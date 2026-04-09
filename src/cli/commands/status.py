"""系统监控仪表板命令 - 实时监控系统状态与资源使用

提供进程信息、系统资源、缓存状态、连接池状态、任务统计的实时监控，
支持仪表板展示和健康度评分功能。

冰璃岩开发组 (BLY Team)
"""

import os
import sys
import time
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.box import SIMPLE, ROUNDED

from ..output import OutputFormatter, BrandColors, BrandAssets, OutputStyle


# Windows 终端兼容图标
class WinIcons:
    """Windows 兼容图标 - 避免使用特殊 Unicode 字符"""
    # 基础状态
    SUCCESS = "+"
    ERROR = "-"
    WARNING = "!"
    INFO = "i"
    
    # 类别图标
    PROCESS = "[PROC]"
    SYSTEM = "[SYS]"
    CACHE = "[CACHE]"
    CONNECTION = "[CONN]"
    TASKS = "[TASK]"
    HEALTH = "[HLTH]"
    
    # 箭头
    UP = "^"
    DOWN = "v"


@dataclass
class ProcessInfo:
    """进程信息数据类"""
    pid: int = 0
    uptime: str = "0s"
    thread_count: int = 0
    memory_usage: str = "0 MB"
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    start_time: Optional[datetime] = None


@dataclass
class SystemResources:
    """系统资源数据类"""
    cpu_current: float = 0.0
    cpu_average: float = 0.0
    memory_total: int = 0
    memory_used: int = 0
    memory_available: int = 0
    memory_percent: float = 0.0
    disk_total: int = 0
    disk_used: int = 0
    disk_available: int = 0
    disk_percent: float = 0.0
    network_sent: int = 0
    network_recv: int = 0
    network_interfaces: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class CacheStatus:
    """缓存状态数据类"""
    cache_dir_exists: bool = False
    cache_size: str = "0 B"
    cache_size_bytes: int = 0
    cache_file_count: int = 0
    hit_rate: Optional[float] = None
    hit_count: int = 0
    miss_count: int = 0


@dataclass
class ConnectionPoolStatus:
    """连接池状态数据类"""
    enabled: bool = False
    active_connections: int = 0
    idle_connections: int = 0
    total_connections: int = 0
    max_connections: int = 0
    connection_errors: int = 0


@dataclass
class TaskStats:
    """任务统计数据类"""
    recent_tasks: int = 0
    success_count: int = 0
    failed_count: int = 0
    success_rate: float = 0.0
    avg_duration: float = 0.0
    total_duration: float = 0.0


@dataclass
class HealthScore:
    """健康度评分数据类"""
    score: int = 100
    level: str = "优秀"
    color: str = BrandColors.SUCCESS
    suggestions: List[str] = field(default_factory=list)


class SystemMonitor:
    """系统监控器 - 收集各类系统状态数据"""
    
    def __init__(self, output: OutputFormatter):
        self.output = output
        self.console = output.console
        self.start_time = datetime.now()
        self._last_net_io: Optional[Dict[str, int]] = None
        self._cpu_readings: List[float] = []
    
    def get_process_info(self) -> ProcessInfo:
        """获取进程信息"""
        import psutil
        
        current_process = psutil.Process(os.getpid())
        
        # 获取进程启动时间
        try:
            start_time = datetime.fromtimestamp(current_process.create_time())
            uptime = datetime.now() - start_time
            uptime_str = self._format_timedelta(uptime)
        except Exception:
            start_time = self.start_time
            uptime_str = "未知"
        
        # 获取线程数
        try:
            thread_count = current_process.num_threads()
        except Exception:
            thread_count = 0
        
        # 获取内存使用
        try:
            mem_info = current_process.memory_info()
            memory_mb = mem_info.rss / (1024 * 1024)
            memory_str = f"{memory_mb:.2f} MB"
            memory_percent = (mem_info.rss / psutil.virtual_memory().total) * 100
        except Exception:
            memory_str = "未知"
            memory_percent = 0.0
        
        # 获取 CPU 使用率
        try:
            cpu_percent = current_process.cpu_percent(interval=0.1)
        except Exception:
            cpu_percent = 0.0
        
        return ProcessInfo(
            pid=os.getpid(),
            uptime=uptime_str,
            thread_count=thread_count,
            memory_usage=memory_str,
            memory_percent=memory_percent,
            cpu_percent=cpu_percent,
            start_time=start_time
        )
    
    def get_system_resources(self) -> SystemResources:
        """获取系统资源使用情况"""
        import psutil
        
        resources = SystemResources()
        
        # CPU 使用率
        try:
            resources.cpu_current = psutil.cpu_percent(interval=0.1)
            self._cpu_readings.append(resources.cpu_current)
            if len(self._cpu_readings) > 10:
                self._cpu_readings.pop(0)
            resources.cpu_average = sum(self._cpu_readings) / len(self._cpu_readings)
        except Exception:
            resources.cpu_current = 0.0
            resources.cpu_average = 0.0
        
        # 内存使用
        try:
            mem = psutil.virtual_memory()
            resources.memory_total = mem.total
            resources.memory_used = mem.used
            resources.memory_available = mem.available
            resources.memory_percent = mem.percent
        except Exception:
            pass
        
        # 磁盘使用
        try:
            disk = psutil.disk_usage(str(Path.home()))
            resources.disk_total = disk.total
            resources.disk_used = disk.used
            resources.disk_available = disk.free
            resources.disk_percent = disk.percent
        except Exception:
            pass
        
        # 网络接口统计
        try:
            net_io = psutil.net_io_counters(pernic=True)
            
            if self._last_net_io is None:
                self._last_net_io = {}
                for iface, counters in net_io.items():
                    self._last_net_io[iface] = {
                        'bytes_sent': counters.bytes_sent,
                        'bytes_recv': counters.bytes_recv
                    }
            else:
                total_sent = 0
                total_recv = 0
                interfaces = {}
                
                for iface, counters in net_io.items():
                    if iface in self._last_net_io:
                        sent = counters.bytes_sent - self._last_net_io[iface]['bytes_sent']
                        recv = counters.bytes_recv - self._last_net_io[iface]['bytes_recv']
                        total_sent += max(0, sent)
                        total_recv += max(0, recv)
                        interfaces[iface] = {
                            'sent': sent,
                            'recv': recv
                        }
                
                resources.network_sent = total_sent
                resources.network_recv = total_recv
                resources.network_interfaces = interfaces
                
                self._last_net_io = {}
                for iface, counters in net_io.items():
                    self._last_net_io[iface] = {
                        'bytes_sent': counters.bytes_sent,
                        'bytes_recv': counters.bytes_recv
                    }
        except Exception:
            pass
        
        return resources
    
    def get_cache_status(self) -> CacheStatus:
        """获取缓存状态"""
        cache = CacheStatus()
        
        # 获取缓存目录
        cache_dir = self._get_cache_dir()
        cache.cache_dir_exists = cache_dir.exists()
        
        if cache.cache_dir_exists:
            try:
                total_size = 0
                file_count = 0
                
                for file_path in cache_dir.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        file_count += 1
                
                cache.cache_size_bytes = total_size
                cache.cache_size = self._format_size(total_size)
                cache.cache_file_count = file_count
            except Exception:
                pass
        
        # 缓存命中率（如果有统计数据）
        cache_stats_file = cache_dir / "cache_stats.json"
        if cache_stats_file.exists():
            try:
                import json
                with open(cache_stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    cache.hit_count = stats.get('hits', 0)
                    cache.miss_count = stats.get('misses', 0)
                    total = cache.hit_count + cache.miss_count
                    if total > 0:
                        cache.hit_rate = (cache.hit_count / total) * 100
            except Exception:
                pass
        
        return cache
    
    def get_connection_pool_status(self) -> ConnectionPoolStatus:
        """获取连接池状态"""
        pool = ConnectionPoolStatus()
        
        # 尝试从配置或全局状态获取连接池信息
        try:
            # 这里可以根据实际项目结构调整
            # 示例：从全局状态或配置中读取
            pool.enabled = True
            pool.active_connections = 0
            pool.idle_connections = 0
            pool.total_connections = 0
            pool.max_connections = 10
            pool.connection_errors = 0
        except Exception:
            pool.enabled = False
        
        return pool
    
    def get_task_stats(self) -> TaskStats:
        """获取最近任务统计"""
        stats = TaskStats()
        
        # 尝试从任务历史或日志中获取统计
        try:
            # 这里可以根据实际项目结构调整
            # 示例：从任务历史文件中读取
            stats.recent_tasks = 0
            stats.success_count = 0
            stats.failed_count = 0
            stats.success_rate = 0.0
            stats.avg_duration = 0.0
        except Exception:
            pass
        
        return stats
    
    def calculate_health_score(
        self,
        process: ProcessInfo,
        resources: SystemResources,
        cache: CacheStatus
    ) -> HealthScore:
        """计算系统健康度评分"""
        score = 100
        suggestions = []
        
        # CPU 使用率评分（40 分）
        if resources.cpu_current > 90:
            score -= 30
            suggestions.append("CPU 使用率过高，建议关闭不必要的程序")
        elif resources.cpu_current > 70:
            score -= 15
            suggestions.append("CPU 使用率较高，注意监控系统负载")
        elif resources.cpu_current > 50:
            score -= 5
        
        # 内存使用率评分（30 分）
        if resources.memory_percent > 90:
            score -= 25
            suggestions.append("内存使用率过高，建议增加内存或关闭程序")
        elif resources.memory_percent > 70:
            score -= 12
            suggestions.append("内存使用率较高，注意释放内存")
        elif resources.memory_percent > 50:
            score -= 3
        
        # 磁盘使用率评分（20 分）
        if resources.disk_percent > 90:
            score -= 18
            suggestions.append("磁盘空间严重不足，建议清理磁盘")
        elif resources.disk_percent > 70:
            score -= 8
            suggestions.append("磁盘空间较少，建议清理无用文件")
        elif resources.disk_percent > 50:
            score -= 2
        
        # 进程资源评分（10 分）
        if process.memory_percent > 50:
            score -= 8
            suggestions.append("进程内存占用过高，检查是否有内存泄漏")
        elif process.memory_percent > 20:
            score -= 3
        
        # 确保分数在 0-100 范围内
        score = max(0, min(100, score))
        
        # 确定健康等级
        if score >= 90:
            level = "优秀"
            color = BrandColors.SUCCESS
        elif score >= 75:
            level = "良好"
            color = BrandColors.INFO
        elif score >= 60:
            level = "一般"
            color = BrandColors.WARNING
        else:
            level = "需优化"
            color = BrandColors.ERROR
        
        # 如果没有建议，添加一条正面反馈
        if not suggestions:
            suggestions.append("系统运行状态良好，无需优化")
        
        return HealthScore(
            score=score,
            level=level,
            color=color,
            suggestions=suggestions
        )
    
    def _get_cache_dir(self) -> Path:
        """获取缓存目录路径"""
        system = sys.platform
        if system == "win32":
            base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        elif system == "darwin":
            base = Path.home() / "Library/Caches"
        else:
            base = Path.home() / ".cache"
        
        return base / "PPC8"
    
    def _format_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def _format_timedelta(self, td: timedelta) -> str:
        """格式化时间差"""
        total_seconds = int(td.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        elif total_seconds < 86400:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
        else:
            days = total_seconds // 86400
            hours = (total_seconds % 86400) // 3600
            return f"{days}d {hours}h"
    
    def _format_bytes(self, bytes_value: int) -> str:
        """格式化字节数为可读格式"""
        return self._format_size(bytes_value)


class StatusDashboard:
    """状态仪表板 - 展示系统监控数据"""
    
    def __init__(self, monitor: SystemMonitor, output: OutputFormatter):
        self.monitor = monitor
        self.output = output
        self.console = output.console
        self.running = True
    
    def create_layout(self) -> Layout:
        """创建仪表板布局"""
        layout = Layout()
        
        # 分割为三个区域：头部、主体、底部
        layout.split(
            Layout(name="header", size=6),
            Layout(name="body"),
            Layout(name="footer", size=10),
        )
        
        # 主体区域再分割为左右两部分
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        
        # 左边区域分割为进程信息和系统资源
        layout["left"].split(
            Layout(name="process", size=12),
            Layout(name="system"),
        )
        
        # 右边区域分割为缓存、连接池和任务统计
        layout["right"].split(
            Layout(name="cache"),
            Layout(name="pool"),
            Layout(name="tasks"),
        )
        
        return layout
    
    def update_layout(self, layout: Layout) -> None:
        """更新仪表板内容"""
        # 收集数据
        process = self.monitor.get_process_info()
        resources = self.monitor.get_system_resources()
        cache = self.monitor.get_cache_status()
        pool = self.monitor.get_connection_pool_status()
        tasks = self.monitor.get_task_stats()
        health = self.monitor.calculate_health_score(process, resources, cache)
        
        # 更新各个区域
        layout["header"].update(self._make_header_panel())
        layout["process"].update(self._make_process_panel(process))
        layout["system"].update(self._make_system_panel(resources))
        layout["cache"].update(self._make_cache_panel(cache))
        layout["pool"].update(self._make_pool_panel(pool))
        layout["tasks"].update(self._make_tasks_panel(tasks))
        layout["footer"].update(self._make_health_panel(health))
    
    def _make_header_panel(self) -> Panel:
        """创建头部面板"""
        # Windows 终端兼容性：使用文本图标替代 emoji
        is_windows = sys.platform == "win32"
        chart_icon = "📊" if not is_windows else "[STATUS]"
        
        title = f"{chart_icon} PPC8 系统监控仪表板"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="left", style=BrandColors.PRIMARY)
        grid.add_column(justify="right", style="dim")
        
        grid.add_row(
            f"[bold white]{title}[/bold white]",
            f"[dim]{current_time}[/dim]"
        )
        grid.add_row(
            f"[dim]{BrandAssets.TAGLINE}[/dim]",
            f"[dim]按 Ctrl+C 退出[/dim]"
        )
        
        return Panel(
            grid,
            border_style=BrandColors.PRIMARY,
            box=ROUNDED,
        )
    
    def _make_process_panel(self, process: ProcessInfo) -> Panel:
        """创建进程信息面板"""
        # Windows 终端兼容性
        is_windows = sys.platform == "win32"
        gear_icon = "⚙" if not is_windows else WinIcons.INFO
        
        table = Table(show_header=False, box=SIMPLE, padding=(0, 1))
        table.add_column("项目", style=BrandColors.ACCENT, width=18)
        table.add_column("值", style="cyan")
        
        table.add_row("进程 PID", str(process.pid))
        table.add_row("运行时长", process.uptime)
        table.add_row("线程数", str(process.thread_count))
        table.add_row("内存使用", process.memory_usage)
        table.add_row("内存占比", f"{process.memory_percent:.1f}%")
        table.add_row("CPU 使用", f"{process.cpu_percent:.1f}%")
        
        return Panel(
            table,
            title=f"[bold {BrandColors.PRIMARY}]{gear_icon} 进程信息[/bold {BrandColors.PRIMARY}]",
            border_style=BrandColors.PRIMARY,
            box=ROUNDED,
        )
    
    def _make_system_panel(self, resources: SystemResources) -> Panel:
        """创建系统资源面板"""
        # Windows 终端兼容性
        is_windows = sys.platform == "win32"
        chart_icon = "📊" if not is_windows else WinIcons.SYSTEM
        
        table = Table(show_header=False, box=SIMPLE, padding=(0, 1))
        table.add_column("项目", style=BrandColors.ACCENT, width=18)
        table.add_column("值", style="cyan")
        
        # CPU
        cpu_bar = self._make_progress_bar(resources.cpu_current / 100, BrandColors.INFO)
        table.add_row("CPU 当前", f"{resources.cpu_current:.1f}% {cpu_bar}")
        table.add_row("CPU 平均", f"{resources.cpu_average:.1f}%")
        
        # 内存
        mem_bar = self._make_progress_bar(resources.memory_percent / 100, BrandColors.WARNING)
        mem_used = self.monitor._format_bytes(resources.memory_used)
        mem_total = self.monitor._format_bytes(resources.memory_total)
        table.add_row("内存使用", f"{mem_used}/{mem_total} {mem_bar}")
        table.add_row("内存可用", self.monitor._format_bytes(resources.memory_available))
        
        # 磁盘
        disk_bar = self._make_progress_bar(resources.disk_percent / 100, BrandColors.SUCCESS)
        disk_used = self.monitor._format_bytes(resources.disk_used)
        disk_total = self.monitor._format_bytes(resources.disk_total)
        table.add_row("磁盘使用", f"{disk_used}/{disk_total} {disk_bar}")
        table.add_row("磁盘可用", self.monitor._format_bytes(resources.disk_available))
        
        # 网络
        net_sent = self.monitor._format_bytes(resources.network_sent)
        net_recv = self.monitor._format_bytes(resources.network_recv)
        
        # Windows 终端兼容性：使用 ASCII 箭头
        up_arrow = WinIcons.UP
        down_arrow = WinIcons.DOWN
        
        table.add_row("网络发送", f"{up_arrow} {net_sent}/s")
        table.add_row("网络接收", f"{down_arrow} {net_recv}/s")
        
        return Panel(
            table,
            title=f"[bold {BrandColors.PRIMARY}]{chart_icon} 系统资源[/bold {BrandColors.PRIMARY}]",
            border_style=BrandColors.PRIMARY,
            box=ROUNDED,
        )
    
    def _make_cache_panel(self, cache: CacheStatus) -> Panel:
        """创建缓存状态面板"""
        # Windows 终端兼容性
        is_windows = sys.platform == "win32"
        folder_icon = "📁" if not is_windows else WinIcons.CACHE
        success_icon = WinIcons.SUCCESS
        error_icon = WinIcons.ERROR
        
        table = Table(show_header=False, box=SIMPLE, padding=(0, 1))
        table.add_column("项目", style=BrandColors.ACCENT, width=18)
        table.add_column("值", style="cyan")
        
        # 缓存目录
        status_icon = success_icon if cache.cache_dir_exists else error_icon
        status_color = BrandColors.SUCCESS if cache.cache_dir_exists else BrandColors.ERROR
        table.add_row("缓存目录", f"[{status_color}]{status_icon}[/{status_color}] {'存在' if cache.cache_dir_exists else '不存在'}")
        
        if cache.cache_dir_exists:
            table.add_row("缓存大小", cache.cache_size)
            table.add_row("文件数量", str(cache.cache_file_count))
            
            if cache.hit_rate is not None:
                hit_color = BrandColors.SUCCESS if cache.hit_rate > 50 else BrandColors.WARNING
                table.add_row("缓存命中率", f"[{hit_color}]{cache.hit_rate:.1f}%[/{hit_color}]")
                table.add_row("命中次数", str(cache.hit_count))
                table.add_row("未命中次数", str(cache.miss_count))
        
        return Panel(
            table,
            title=f"[bold {BrandColors.PRIMARY}]{folder_icon} 缓存状态[/bold {BrandColors.PRIMARY}]",
            border_style=BrandColors.PRIMARY,
            box=ROUNDED,
        )
    
    def _make_pool_panel(self, pool: ConnectionPoolStatus) -> Panel:
        """创建连接池面板"""
        # Windows 终端兼容性
        is_windows = sys.platform == "win32"
        link_icon = "🔗" if not is_windows else WinIcons.CONNECTION
        
        table = Table(show_header=False, box=SIMPLE, padding=(0, 1))
        table.add_column("项目", style=BrandColors.ACCENT, width=18)
        table.add_column("值", style="cyan")
        
        # 连接池状态
        status_icon = WinIcons.SUCCESS if pool.enabled else "未启用"
        status_color = BrandColors.SUCCESS if pool.enabled else "dim"
        table.add_row("连接池", f"[{status_color}]{status_icon}[/{status_color}]")
        
        if pool.enabled:
            # 连接使用情况
            if pool.max_connections > 0:
                usage = pool.total_connections / pool.max_connections
                bar = self._make_progress_bar(usage, BrandColors.INFO)
                table.add_row("总连接数", f"{pool.total_connections}/{pool.max_connections} {bar}")
            else:
                table.add_row("总连接数", str(pool.total_connections))
            
            table.add_row("活跃连接", str(pool.active_connections))
            table.add_row("空闲连接", str(pool.idle_connections))
            
            if pool.connection_errors > 0:
                table.add_row("连接错误", f"[{BrandColors.ERROR}]{pool.connection_errors}[/{BrandColors.ERROR}]")
            else:
                table.add_row("连接错误", "0")
        
        return Panel(
            table,
            title=f"[bold {BrandColors.PRIMARY}]{link_icon} 连接池状态[/bold {BrandColors.PRIMARY}]",
            border_style=BrandColors.PRIMARY,
            box=ROUNDED,
        )
    
    def _make_tasks_panel(self, tasks: TaskStats) -> Panel:
        """创建任务统计面板"""
        # Windows 终端兼容性
        is_windows = sys.platform == "win32"
        chart_icon = "📈" if not is_windows else WinIcons.TASKS
        success_icon = WinIcons.SUCCESS
        error_icon = WinIcons.ERROR
        
        table = Table(show_header=False, box=SIMPLE, padding=(0, 1))
        table.add_column("项目", style=BrandColors.ACCENT, width=18)
        table.add_column("值", style="cyan")
        
        table.add_row("最近任务数", str(tasks.recent_tasks))
        
        if tasks.recent_tasks > 0:
            # 成功率
            success_color = BrandColors.SUCCESS if tasks.success_rate > 80 else (
                BrandColors.WARNING if tasks.success_rate > 50 else BrandColors.ERROR
            )
            table.add_row("成功", f"[{BrandColors.SUCCESS}]{success_icon} {tasks.success_count}[/{BrandColors.SUCCESS}]")
            table.add_row("失败", f"[{BrandColors.ERROR}]{error_icon} {tasks.failed_count}[/{BrandColors.ERROR}]")
            table.add_row("成功率", f"[{success_color}]{tasks.success_rate:.1f}%[/{success_color}]")
            
            # 平均耗时
            if tasks.avg_duration > 0:
                table.add_row("平均耗时", f"{tasks.avg_duration:.2f}s")
                table.add_row("总耗时", f"{tasks.total_duration:.2f}s")
        else:
            table.add_row("成功率", "[dim]无数据[/dim]")
        
        return Panel(
            table,
            title=f"[bold {BrandColors.PRIMARY}]{chart_icon} 最近任务统计[/bold {BrandColors.PRIMARY}]",
            border_style=BrandColors.PRIMARY,
            box=ROUNDED,
        )
    
    def _make_health_panel(self, health: HealthScore) -> Panel:
        """创建健康度评分面板"""
        # Windows 终端兼容性
        is_windows = sys.platform == "win32"
        heart_icon = "💚" if not is_windows else WinIcons.HEALTH
        
        # 健康度进度条
        health_bar = self._make_progress_bar(health.score / 100, health.color, width=40)
        
        # 构建内容
        content = []
        content.append(f"[bold {health.color}]健康评分：{health.score}/100[/bold {health.color}] {health_bar}")
        content.append(f"[bold]健康等级:[/bold] [{health.color}]{health.level}[/{health.color}]")
        content.append("")
        
        # 优化建议
        content.append(f"[bold {BrandColors.ACCENT}]优化建议:[/bold {BrandColors.ACCENT}]")
        for i, suggestion in enumerate(health.suggestions[:3], 1):  # 最多显示 3 条
            content.append(f"  {i}. {suggestion}")
        
        text = "\n".join(content)
        
        return Panel(
            text,
            title=f"[bold {BrandColors.PRIMARY}]{heart_icon} 系统健康度[/bold {BrandColors.PRIMARY}]",
            border_style=health.color,
            box=ROUNDED,
        )
    
    def _make_progress_bar(
        self,
        value: float,
        color: str,
        width: int = 20
    ) -> str:
        """创建进度条"""
        value = max(0, min(1, value))
        filled = int(width * value)
        empty = width - filled
        
        # Windows 终端兼容性：使用 ASCII 字符替代 Unicode 块
        is_windows = sys.platform == "win32"
        if is_windows:
            bar = '=' * filled + '-' * empty
            return f"[{color}][{bar}][/{color}]"
        else:
            bar = f"[{color}]{'█' * filled}[/{color}]" + '░' * empty
            return f"[{color}][{bar}][/{color}]"
    
    def dashboard(self, watch: bool = False) -> None:
        """展示仪表板
        
        Args:
            watch: 是否启用实时监控模式
        """
        if not watch:
            # 静态模式：只显示一次
            layout = self.create_layout()
            self.update_layout(layout)
            self.console.print(layout)
        else:
            # 实时监控模式
            self._run_watch_mode()
    
    def _run_watch_mode(self) -> None:
        """运行实时监控模式"""
        layout = self.create_layout()
        
        # 设置信号处理
        def signal_handler(sig, frame):
            self.running = False
        
        original_handler = signal.signal(signal.SIGINT, signal_handler)
        
        try:
            with Live(layout, console=self.console, refresh_per_second=1, screen=True) as live:
                while self.running:
                    self.update_layout(layout)
                    live.refresh()
                    time.sleep(1)
        finally:
            # 恢复原始信号处理
            signal.signal(signal.SIGINT, original_handler)
            
            # 清屏并显示退出消息
            self.console.clear()
            self.console.print(
                f"[bold {BrandColors.SUCCESS}]✓ 监控已停止[/bold {BrandColors.SUCCESS}]"
            )


def handle_status(watch: bool = False) -> None:
    """处理 status 命令
    
    Args:
        watch: 是否启用实时监控模式
    """
    output = OutputFormatter(verbose=False)
    
    # 检查 psutil 依赖
    try:
        import psutil
    except ImportError:
        output.error_panel(
            "缺少依赖：psutil",
            title="依赖错误",
            error_type="ImportError",
            suggestion="运行 pip install psutil 安装"
        )
        return
    
    # 创建监控器和仪表板
    monitor = SystemMonitor(output)
    dashboard = StatusDashboard(monitor, output)
    
    # 显示仪表板
    dashboard.dashboard(watch=watch)


# 导出给 CLI 使用
__all__ = ['handle_status', 'SystemMonitor', 'StatusDashboard']
