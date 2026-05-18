"""系统状态命令 - 系统监控面板与健康评分。"""

import sys
import asyncio
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.box import ROUNDED, SIMPLE
from rich.console import Console

from ..output import OutputFormatter, BrandColors, Icons


class SystemStatusMonitor:
    """系统状态监控器 - 提供系统级监控。"""

    def __init__(self, output: OutputFormatter):
        self.output = output
        self.console = output.console
        self.start_time = time.time()

    def _get_process_info(self) -> dict:
        """获取当前进程信息。"""
        import os

        pid = os.getpid()
        import psutil
        process = psutil.Process(pid)

        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)

        try:
            import threading
            thread_count = threading.active_count()
        except Exception:
            thread_count = 0

        return {
            "pid": pid,
            "memory_rss": memory_info.rss,
            "memory_vms": memory_info.vms,
            "cpu_percent": cpu_percent,
            "thread_count": thread_count,
        }

    def _get_system_resources(self) -> dict:
        """获取系统资源使用情况。"""
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        try:
            import shutil
            disk = shutil.disk_usage(str(Path.home()))
            disk_usage = (disk.used / disk.total * 100)
        except Exception:
            disk_usage = 0

        return {
            "cpu_percent": cpu_percent,
            "memory_total": memory.total,
            "memory_used": memory.used,
            "memory_percent": memory.percent,
            "disk_usage_percent": disk_usage,
        }

    def _get_cache_status(self) -> dict:
        """获取缓存状态。"""
        from ...cache import get_cache
        cache = get_cache()

        stats = cache.get_stats()
        total = stats.get("l1_hits", 0) + stats.get("l1_misses", 0) + stats.get("l2_hits", 0) + stats.get("l2_misses", 0)
        hits = stats.get("l1_hits", 0) + stats.get("l2_hits", 0)
        misses = stats.get("l1_misses", 0) + stats.get("l2_misses", 0)
        hit_rate = (hits / total * 100) if total > 0 else 0.0

        return {
            "total": stats.get("total_sets", 0),
            "hits": hits,
            "misses": misses,
            "hit_rate": hit_rate,
            "memory_usage": stats.get("total_size_bytes", 0),
        }

    def _get_connection_pool_status(self) -> dict:
        """获取连接池状态。"""
        try:
            from ...pool.connection_pool import ConnectionPoolManager
            pool_manager = ConnectionPoolManager()

            pools = pool_manager.get_all_stats()
            active = sum(p.get("active_connections", 0) for p in pools.values())
            total = sum(p.get("total_connections", 0) for p in pools.values())
            return {
                "active_connections": active,
                "max_connections": total,
                "utilization": active / max(1, total),
            }
        except Exception:
            return {
                "active_connections": 0,
                "max_connections": 0,
                "utilization": 0,
            }

    def _get_task_statistics(self) -> dict:
        """获取任务统计信息。"""
        try:
            from ...core.base import EngineStats, BaseEngine
            return {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "current_speed": 0.0,
            }
        except Exception:
            return {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "current_speed": 0.0,
            }

    def calculate_health_score(self) -> float:
        """计算系统健康评分。"""
        try:
            process_info = self._get_process_info()
            resources = self._get_system_resources()
            cache_status = self._get_cache_status()
            pool_status = self._get_connection_pool_status()
            task_stats = self._get_task_statistics()
        except Exception:
            return 0.0

        process_score = 100
        if process_info["memory_rss"] > 256 * 1024 * 1024:
            process_score -= 30
        elif process_info["memory_rss"] > 128 * 1024 * 1024:
            process_score -= 15

        if process_info["cpu_percent"] > 80:
            process_score -= 20
        elif process_info["cpu_percent"] > 60:
            process_score -= 10

        resource_score = 100
        if resources["memory_percent"] > 90:
            resource_score -= 40
        elif resources["memory_percent"] > 70:
            resource_score -= 20

        if resources["cpu_percent"] > 90:
            resource_score -= 30
        elif resources["cpu_percent"] > 70:
            resource_score -= 15

        cache_score = 100
        hit_rate = cache_status.get("hit_rate", 0)
        if hit_rate < 30:
            cache_score -= 30
        elif hit_rate < 60:
            cache_score -= 15

        pool_score = 100
        utilization = pool_status.get("utilization", 0)
        if utilization > 0.9:
            pool_score -= 40
        elif utilization > 0.7:
            pool_score -= 20

        task_score = 100
        success_rate = task_stats.get("success_rate", 0)
        if success_rate < 50:
            task_score -= 50
        elif success_rate < 70:
            task_score -= 30
        elif success_rate < 90:
            task_score -= 10

        weights = {
            "process": 0.2,
            "resource": 0.25,
            "cache": 0.15,
            "pool": 0.15,
            "task": 0.25,
        }

        health_score = (
            process_score * weights["process"] +
            resource_score * weights["resource"] +
            cache_score * weights["cache"] +
            pool_score * weights["pool"] +
            task_score * weights["task"]
        )

        return max(0, min(100, health_score))

    def _format_size(self, size_bytes: int) -> str:
        """格式化字节大小。"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

    def _format_duration(self, seconds: float) -> str:
        """格式化时间长度。"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _create_process_panel(self) -> Panel:
        """创建进程信息面板。"""
        process_info = self._get_process_info()

        content_lines = [
            f"[bold]进程 ID:[/bold] {process_info['pid']}",
            f"[bold]内存 (RSS):[/bold] {self._format_size(process_info['memory_rss'])}",
            f"[bold]内存 (VMS):[/bold] {self._format_size(process_info['memory_vms'])}",
            f"[bold]CPU 使用率:[/bold] {process_info['cpu_percent']:.1f}%",
            f"[bold]线程数:[/bold] {process_info['thread_count']}",
        ]

        return Panel(
            "\n".join(content_lines),
            title="[bold]进程信息[/bold]",
            border_style=BrandColors.PRIMARY,
            box=SIMPLE,
            padding=(0, 1)
        )

    def _create_resources_panel(self) -> Panel:
        """创建系统资源面板。"""
        resources = self._get_system_resources()

        cpu_color = BrandColors.SUCCESS if resources["cpu_percent"] < 70 else (BrandColors.WARNING if resources["cpu_percent"] < 90 else BrandColors.ERROR)
        mem_color = BrandColors.SUCCESS if resources["memory_percent"] < 70 else (BrandColors.WARNING if resources["memory_percent"] < 90 else BrandColors.ERROR)

        content_lines = [
            f"[bold]CPU:[/bold] [{cpu_color}]{resources['cpu_percent']:.1f}%[/{cpu_color}]",
            f"[bold]内存总计:[/bold] {self._format_size(resources['memory_total'])}",
            f"[bold]内存已用:[/bold] [{mem_color}]{self._format_size(resources['memory_used'])}[/{mem_color}]",
            f"[bold]内存使用率:[/bold] [{mem_color}]{resources['memory_percent']:.1f}%[/{mem_color}]",
            f"[bold]磁盘使用率:[/bold] {resources['disk_usage_percent']:.1f}%",
        ]

        return Panel(
            "\n".join(content_lines),
            title="[bold]系统资源[/bold]",
            border_style=BrandColors.ACCENT,
            box=SIMPLE,
            padding=(0, 1)
        )

    def _create_cache_panel(self) -> Panel:
        """创建缓存状态面板。"""
        cache_status = self._get_cache_status()

        hit_rate = cache_status.get("hit_rate", 0)
        hit_color = BrandColors.SUCCESS if hit_rate > 60 else (BrandColors.WARNING if hit_rate > 30 else BrandColors.ERROR)

        content_lines = [
            f"[bold]缓存总数:[/bold] {cache_status['total']}",
            f"[bold]命中数:[/bold] {cache_status['hits']}",
            f"[bold]未命中数:[/bold] {cache_status['misses']}",
            f"[bold]命中率:[/bold] [{hit_color}]{hit_rate:.1f}%[/{hit_color}]",
            f"[bold]内存使用:[/bold] {self._format_size(cache_status['memory_usage'])}",
        ]

        return Panel(
            "\n".join(content_lines),
            title="[bold]缓存状态[/bold]",
            border_style=BrandColors.SECONDARY,
            box=SIMPLE,
            padding=(0, 1)
        )

    def _create_pool_panel(self) -> Panel:
        """创建连接池面板。"""
        pool_status = self._get_connection_pool_status()

        utilization = pool_status.get("utilization", 0)
        util_color = BrandColors.SUCCESS if utilization < 0.7 else (BrandColors.WARNING if utilization < 0.9 else BrandColors.ERROR)

        content_lines = [
            f"[bold]活跃连接:[/bold] {pool_status['active_connections']}",
            f"[bold]最大连接:[/bold] {pool_status['max_connections']}",
            f"[bold]利用率:[/bold] [{util_color}]{utilization * 100:.1f}%[/{util_color}]",
        ]

        return Panel(
            "\n".join(content_lines),
            title="[bold]连接池状态[/bold]",
            border_style=BrandColors.INFO,
            box=SIMPLE,
            padding=(0, 1)
        )

    def _create_tasks_panel(self) -> Panel:
        """创建任务统计面板。"""
        task_stats = self._get_task_statistics()

        success_rate = task_stats.get("success_rate", 0)
        rate_color = BrandColors.SUCCESS if success_rate > 90 else (BrandColors.WARNING if success_rate > 70 else BrandColors.ERROR)

        content_lines = [
            f"[bold]总任务数:[/bold] {task_stats['total_tasks']}",
            f"[bold]成功任务:[/bold] {task_stats['successful_tasks']}",
            f"[bold]失败任务:[/bold] {task_stats['failed_tasks']}",
            f"[bold]成功率:[/bold] [{rate_color}]{success_rate:.1f}%[/{rate_color}]",
            f"[bold]平均耗时:[/bold] {self._format_duration(task_stats['avg_duration'])}",
            f"[bold]当前速度:[/bold] {task_stats['current_speed']:.2f} 任务/秒",
        ]

        return Panel(
            "\n".join(content_lines),
            title="[bold]任务统计[/bold]",
            border_style=BrandColors.WARNING,
            box=SIMPLE,
            padding=(0, 1)
        )

    def _create_health_panel(self, health_score: float) -> Panel:
        """Create health score panel."""
        if health_score >= 90:
            color = BrandColors.SUCCESS
            status = "优秀"
            icon = Icons.SUCCESS
        elif health_score >= 70:
            color = BrandColors.WARNING
            status = "良好"
            icon = Icons.WARNING
        else:
            color = BrandColors.ERROR
            status = "需关注"
            icon = Icons.ERROR

        content_lines = [
            f"[bold {color}]健康评分:[/bold {color}] [{color}]{health_score:.1f}/100[/{color}]",
            f"[bold]状态:[/bold] [{color}]{icon} {status}[/{color}]",
        ]

        elapsed = time.time() - self.start_time
        content_lines.append(f"[bold]监控时长:[/bold] {self._format_duration(elapsed)}")

        return Panel(
            "\n".join(content_lines),
            title="[bold]健康度评分[/bold]",
            border_style=color,
            box=SIMPLE,
            padding=(0, 1)
        )

    def show_dashboard(self):
        """Show dashboard."""
        self.console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
        self.console.print(f"[bold white]  {Icons.CHART} PPC9 系统状态监控[/bold white]")
        self.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n")

        health_score = self.calculate_health_score()

        process_panel = self._create_process_panel()
        resources_panel = self._create_resources_panel()
        cache_panel = self._create_cache_panel()
        pool_panel = self._create_pool_panel()
        tasks_panel = self._create_tasks_panel()
        health_panel = self._create_health_panel(health_score)

        from rich.columns import Columns
        columns1 = Columns([process_panel, resources_panel], equal=True, expand=True)
        columns2 = Columns([cache_panel, pool_panel], equal=True, expand=True)
        columns3 = Columns([tasks_panel, health_panel], equal=True, expand=True)

        self.console.print(columns1)
        self.console.print()
        self.console.print(columns2)
        self.console.print()
        self.console.print(columns3)

        self.console.print(f"\n[bold {BrandColors.PRIMARY}]{'─' * 60}[/bold {BrandColors.PRIMARY}]")
        self.console.print(f"[dim]数据更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")


def handle_status(watch: bool = False):
    """处理状态命令。"""
    output = OutputFormatter(verbose=False)
    monitor = SystemStatusMonitor(output)

    if watch:
        async def run_watch():
            live = Live(output.console, refresh_per_second=1)
            live.start()
            try:
                while True:
                    health_score = monitor.calculate_health_score()

                    process_panel = monitor._create_process_panel()
                    resources_panel = monitor._create_resources_panel()
                    cache_panel = monitor._create_cache_panel()
                    pool_panel = monitor._create_pool_panel()
                    tasks_panel = monitor._create_tasks_panel()
                    health_panel = monitor._create_health_panel(health_score)

                    from rich.columns import Columns
                    columns1 = Columns([process_panel, resources_panel], equal=True, expand=True)
                    columns2 = Columns([cache_panel, pool_panel], equal=True, expand=True)
                    columns3 = Columns([tasks_panel, health_panel], equal=True, expand=True)

                    content = (
                        f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n"
                        f"[bold white]  {Icons.CHART} PPC9 系统状态监控（实时）[/bold white]\n"
                        f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n\n"
                    )

                    try:
                        live.update(content)
                        live.stop()
                        output.console.print(content)
                        output.console.print(columns1)
                        output.console.print()
                        output.console.print(columns2)
                        output.console.print()
                        output.console.print(columns3)
                        output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'─' * 60}[/bold {BrandColors.PRIMARY}]")
                        output.console.print(f"[dim]数据更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（按 Ctrl+C 退出）[/dim]")
                    except Exception:
                        pass

                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                live.stop()
                output.info("\n实时监控已停止")
                return False

        try:
            asyncio.run(run_watch())
        except KeyboardInterrupt:
            output.info("\n用户中断")
            sys.exit(130)
        except Exception as e:
            output.error_panel(
                f"实时监控失败：{e}",
                title="监控错误",
                error_type=type(e).__name__,
                suggestion="使用不带 --watch 参数查看静态状态"
            )
            sys.exit(1)
    else:
        monitor.show_dashboard()
        sys.exit(0)
