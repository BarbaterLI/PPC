"""分布式命令实现 - 多节点 TTS 集群管理。"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

from rich.table import Table
from rich.panel import Panel
from rich.box import SIMPLE, ROUNDED

from ...config import ConfigManager
from ...infrastructure import (
    TTSNode,
    DistributedTTSExecutor,
    NodeStatus,
    HealthCheckConfig,
    create_default_config,
)
from ..output import OutputFormatter, BrandColors, Icons


def node_start(host: str, port: int, max_concurrency: int, config_path: Optional[Path]):
    """处理分布式节点命令 - 启动 TTS 节点服务。"""
    output = OutputFormatter(verbose=False)

    output.show_banner()

    output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    output.console.print(f"[bold white]  {Icons.GEAR} PPC9 分布式 TTS 节点[/bold white]")
    output.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n")

    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()

    node_config = {
        "host": host,
        "port": port,
        "max_concurrency": max_concurrency,
    }

    for key, value in node_config.items():
        output.console.print(f"[dim]{key}:[/dim] [cyan]{value}[/cyan]")

    output.console.print()

    async def run_node():
        node = TTSNode(
            host=host,
            port=port,
            max_concurrency=max_concurrency,
            config=config,
            health_check_config=HealthCheckConfig()
        )

        output.console.print(f"[bold {BrandColors.SUCCESS}]{'═' * 60}[/bold {BrandColors.SUCCESS}]")
        output.console.print(f"[bold white]  {Icons.SUCCESS} TTS 节点已启动[/bold white]")
        output.console.print(f"[bold {BrandColors.SUCCESS}]{'═' * 60}[/bold {BrandColors.SUCCESS}]\n")

        output.console.print(f"  [bold {BrandColors.INFO}]i 服务地址:[/bold {BrandColors.INFO}] [bold cyan]http://{host}:{port}[/bold cyan]")
        output.console.print(f"  [bold {BrandColors.INFO}]i 最大并发:[/bold {BrandColors.INFO}] {max_concurrency}")
        output.console.print(f"  [bold {BrandColors.INFO}]i 健康检查:[/bold {BrandColors.INFO}] http://{host}:{port}/health\n")

        output.console.print(f"[dim]按 Ctrl+C 停止服务[/dim]\n")

        await node.start()

    try:
        asyncio.run(run_node())
    except KeyboardInterrupt:
        output.info("\n用户中断，服务已停止")
        sys.exit(130)
    except Exception as e:
        output.error_panel(
            f"启动节点失败：{e}",
            title="启动错误",
            error_type=type(e).__name__,
            suggestion="检查端口是否被占用或使用其他端口"
        )
        sys.exit(1)


def distributed_status(config_path: Optional[Path], export: Optional[Path]):
    """处理分布式状态命令 - 查看分布式系统状态。"""
    output = OutputFormatter(verbose=False)

    output.show_banner()

    output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    output.console.print(f"[bold white]  {Icons.CHART} PPC9 分布式状态[/bold white]")
    output.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n")

    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()

    async def run_status():
        executor = DistributedTTSExecutor(config)
        try:
            await executor.initialize()
            status = await executor.get_cluster_status()

            nodes = status.get("nodes", [])
            tasks = status.get("tasks", {})
            stats = status.get("stats", {})

            output.console.print(f"[bold {BrandColors.SUCCESS}]{'─' * 60}[/bold {BrandColors.SUCCESS}]")
            output.console.print(f"[bold white]  节点状态[/bold white]")
            output.console.print(f"[bold {BrandColors.SUCCESS}]{'─' * 60}[/bold {BrandColors.SUCCESS}]\n")

            if nodes:
                nodes_table = Table(show_header=True, box=SIMPLE, border_style=BrandColors.SUCCESS)
                nodes_table.add_column("节点 ID", style="cyan", width=12)
                nodes_table.add_column("地址", style="white", width=25)
                nodes_table.add_column("状态", style="green", width=10)
                nodes_table.add_column("活跃任务", style="yellow", width=10, justify="right")
                nodes_table.add_column("CPU", style="magenta", width=10, justify="right")
                nodes_table.add_column("内存", style="magenta", width=10, justify="right")

                for node in nodes:
                    node_id = node.get("node_id", "N/A")
                    address = f"{node.get('host', 'N/A')}:{node.get('port', 0)}"
                    node_status = node.get("status", "N/A")

                    status_icon = {
                        "RUNNING": "+",
                        "STOPPED": "-",
                        "ERROR": "!",
                    }.get(node_status, "?")

                    status_color = {
                        "RUNNING": BrandColors.SUCCESS,
                        "STOPPED": BrandColors.WARNING,
                        "ERROR": BrandColors.ERROR,
                    }.get(node_status, "white")

                    resources = node.get("resources", {})
                    cpu_usage = resources.get("cpu_percent", 0)
                    memory_usage = resources.get("memory_percent", 0)

                    nodes_table.add_row(
                        node_id,
                        address,
                        f"[{status_color}]{status_icon} {node_status}[/{status_color}]",
                        str(node.get("active_tasks", 0)),
                        f"{cpu_usage:.1f}%",
                        f"{memory_usage:.1f}%"
                    )

                output.console.print(nodes_table)
            else:
                output.console.print("[dim]当前无已注册节点[/dim]\n")

            if tasks:
                output.console.print(f"\n[bold {BrandColors.INFO}]{'─' * 60}[/bold {BrandColors.INFO}]")
                output.console.print(f"[bold white]  任务状态[/bold white]")
                output.console.print(f"[bold {BrandColors.INFO}]{'─' * 60}[/bold {BrandColors.INFO}]\n")

                tasks_table = Table(show_header=True, box=SIMPLE, border_style=BrandColors.INFO)
                tasks_table.add_column("状态", style="bold", width=12)
                tasks_table.add_column("数量", style="cyan", width=10, justify="right")

                for task_status, count in tasks.items():
                    if isinstance(count, int):
                        tasks_table.add_row(task_status, str(count))

                output.console.print(tasks_table)

            if stats:
                output.console.print(f"\n[bold {BrandColors.ACCENT}]{'─' * 60}[/bold {BrandColors.ACCENT}]")
                output.console.print(f"[bold white]  集群统计[/bold white]")
                output.console.print(f"[bold {BrandColors.ACCENT}]{'─' * 60}[/bold {BrandColors.ACCENT}]\n")

                stats_table = Table(show_header=False, box=SIMPLE, border_style=BrandColors.ACCENT)
                stats_table.add_column("指标", style="bold", width=20)
                stats_table.add_column("值", style="cyan", width=25)

                if "total_tasks" in stats:
                    stats_table.add_row("总任务数", str(stats["total_tasks"]))
                if "completed_tasks" in stats:
                    stats_table.add_row("完成任务数", str(stats["completed_tasks"]))
                if "failed_tasks" in stats:
                    stats_table.add_row("失败任务数", str(stats["failed_tasks"]))
                if "avg_task_duration" in stats:
                    stats_table.add_row("平均任务耗时", f"{stats['avg_task_duration']:.2f}s")
                if "current_throughput" in stats:
                    stats_table.add_row("当前吞吐量", f"{stats['current_throughput']:.2f} 任务/秒")
                if "cluster_uptime" in stats:
                    uptime_seconds = stats["cluster_uptime"]
                    hours = int(uptime_seconds // 3600)
                    minutes = int((uptime_seconds % 3600) // 60)
                    stats_table.add_row("集群运行时间", f"{hours}小时{minutes}分钟")

                output.console.print(stats_table)

            if export:
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "status": status
                }

                with open(export, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)

                output.console.print(f"\n[bold {BrandColors.SUCCESS}]i 状态已导出：{export}[/bold {BrandColors.SUCCESS}]")

            output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'─' * 60}[/bold {BrandColors.PRIMARY}]")
            output.console.print(f"[dim]数据更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")

            return True

        except Exception as e:
            output.error_panel(
                f"获取状态失败：{e}",
                title="状态错误",
                error_type=type(e).__name__,
                suggestion="确保节点服务已启动并可访问"
            )
            return False
        finally:
            await executor.shutdown()

    try:
        success = asyncio.run(run_status())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        output.warning("\n用户中断")
        sys.exit(130)
    except Exception as e:
        output.error_panel(
            f"执行失败：{e}",
            title="执行错误",
            error_type=type(e).__name__,
            suggestion="使用 --verbose 参数查看详细错误信息"
        )
        sys.exit(1)


def add_node(
    host: str,
    port: int,
    max_concurrency: int,
    config_path: Optional[Path],
    save: bool
):
    """处理分布式添加节点命令 - 向集群添加新节点。"""
    output = OutputFormatter(verbose=False)

    output.show_banner()

    output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    output.console.print(f"[bold white]  {Icons.GEAR} PPC9 添加分布式节点[/bold white]")
    output.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n")

    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()

    async def run_add_node():
        executor = DistributedTTSExecutor(config)
        try:
            await executor.initialize()

            node_id = await executor.add_node(
                host=host,
                port=port,
                max_concurrency=max_concurrency,
                health_check_config=HealthCheckConfig()
            )

            output.console.print(f"[bold {BrandColors.SUCCESS}]{'═' * 60}[/bold {BrandColors.SUCCESS}]")
            output.console.print(f"[bold white]  {Icons.SUCCESS} 节点添加成功[/bold white]")
            output.console.print(f"[bold {BrandColors.SUCCESS}]{'═' * 60}[/bold {BrandColors.SUCCESS}]\n")

            output.console.print(f"  [bold {BrandColors.INFO}]i 节点 ID:[/bold {BrandColors.INFO}] [bold cyan]{node_id}[/bold cyan]")
            output.console.print(f"  [bold {BrandColors.INFO}]i 地址:[/bold {BrandColors.INFO}] [bold cyan]{host}:{port}[/bold cyan]")
            output.console.print(f"  [bold {BrandColors.INFO}]i 最大并发:[/bold {BrandColors.INFO}] {max_concurrency}")

            if save:
                try:
                    config_manager.set("distributed.nodes", host)
                    output.console.print(f"\n[bold {BrandColors.SUCCESS}]i 节点信息已保存[/bold {BrandColors.SUCCESS}]")
                except Exception as e:
                    output.warning(f"\n保存节点信息失败：{e}")

            output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'─' * 60}[/bold {BrandColors.PRIMARY}]")
            output.console.print(f"[dim]添加时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")

            return True

        except Exception as e:
            output.error_panel(
                f"添加节点失败：{e}",
                title="添加错误",
                error_type=type(e).__name__,
                suggestion="检查节点地址和端口是否正确"
            )
            return False
        finally:
            await executor.shutdown()

    try:
        success = asyncio.run(run_add_node())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        output.warning("\n用户中断")
        sys.exit(130)
    except Exception as e:
        output.error_panel(
            f"执行失败：{e}",
            title="执行错误",
            error_type=type(e).__name__,
            suggestion="使用 --verbose 参数查看详细错误信息"
        )
        sys.exit(1)
