"""分布式指标 CLI 命令实现。"""

import sys
import asyncio
from pathlib import Path
from typing import Optional

from rich.table import Table
from rich.box import SIMPLE

from ...config import ConfigManager
from ...distributed import DistributedMetricsCollector
from ..output import OutputFormatter, BrandColors, Icons


def metrics_show(
    config_path: Optional[Path],
    export_format: str = "json",
    export_path: Optional[Path] = None,
):
    """处理分布式指标命令 - 显示分布式指标。"""
    output = OutputFormatter(verbose=False)

    output.show_banner()

    output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    output.console.print(f"[bold white]  {Icons.CHART} PPC9 分布式指标[/bold white]")
    output.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n")

    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()

    collector = DistributedMetricsCollector()

    async def show_metrics():
        from ...infrastructure import (
            DistributedTTSExecutor,
            HealthCheckConfig,
        )

        executor = DistributedTTSExecutor(config)
        try:
            await executor.initialize()
            status = await executor.get_cluster_status()

            nodes = status.get("nodes", [])
            active_count = len([n for n in nodes if n.get("status") == "RUNNING"])

            for node in nodes:
                node_id = node.get("node_id", "unknown")
                collector.record_node_metrics(
                    node_id=node_id,
                    latency=0.0,
                    success=True,
                    concurrency=node.get("active_tasks", 0),
                    max_concurrency=node.get("max_concurrency", 4),
                )

            if export_format == "json":
                json_output = collector.export_json(active_count)
                output.console.print(json_output)
            elif export_format == "prometheus":
                prom_output = collector.export_prometheus(active_count)
                output.console.print(prom_output)
            else:
                _show_metrics_table(output, collector, active_count)

            if export_path:
                if export_format == "json":
                    data = collector.export_json(active_count)
                else:
                    data = collector.export_prometheus(active_count)

                with open(export_path, 'w', encoding='utf-8') as f:
                    f.write(data)

                output.console.print(f"\n[{BrandColors.SUCCESS}]i 指标已导出：{export_path}[/{BrandColors.SUCCESS}]")

            return True

        except Exception as e:
            output.error_panel(
                f"获取指标失败：{e}",
                title="指标错误",
                error_type=type(e).__name__,
                suggestion="确保节点服务已启动并可访问"
            )
            return False
        finally:
            await executor.shutdown()

    try:
        success = asyncio.run(show_metrics())
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


def _show_metrics_table(output, collector: DistributedMetricsCollector, active_count: int):
    """以表格形式显示指标。"""
    from ...distributed import NodeMetrics

    node_metrics = collector.get_all_node_metrics()
    cluster = collector.get_cluster_metrics(active_count)

    output.console.print(f"[bold {BrandColors.SUCCESS}]{'─' * 60}[/bold {BrandColors.SUCCESS}]")
    output.console.print(f"[bold white]  集群概览[/bold white]")
    output.console.print(f"[bold {BrandColors.SUCCESS}]{'─' * 60}[/bold {BrandColors.SUCCESS}]\n")

    cluster_table = Table(show_header=False, box=SIMPLE, border_style=BrandColors.SUCCESS)
    cluster_table.add_column("指标", style="bold", width=20)
    cluster_table.add_column("值", style="cyan", width=25)

    cluster_table.add_row("活跃节点", str(cluster.active_nodes))
    cluster_table.add_row("总请求数", str(cluster.total_requests))
    cluster_table.add_row("成功率", f"{cluster.cluster_success_rate:.2%}")
    cluster_table.add_row("平均延迟", f"{cluster.cluster_avg_latency:.3f}s")
    cluster_table.add_row("吞吐量", f"{cluster.cluster_throughput:.1f} 请求/分")
    cluster_table.add_row("运行时间", f"{cluster.uptime_seconds:.0f}s")

    output.console.print(cluster_table)

    if node_metrics:
        output.console.print(f"\n[bold {BrandColors.INFO}]{'─' * 60}[/bold {BrandColors.INFO}]")
        output.console.print(f"[bold white]  节点指标[/bold white]")
        output.console.print(f"[bold {BrandColors.INFO}]{'─' * 60}[/bold {BrandColors.INFO}]\n")

        nodes_table = Table(show_header=True, box=SIMPLE, border_style=BrandColors.INFO)
        nodes_table.add_column("节点 ID", style="cyan", width=15)
        nodes_table.add_column("平均延迟", style="yellow", width=12, justify="right")
        nodes_table.add_column("P95 延迟", style="yellow", width=12, justify="right")
        nodes_table.add_column("吞吐量", style="green", width=10, justify="right")
        nodes_table.add_column("成功率", style="green", width=10, justify="right")
        nodes_table.add_column("总请求", style="white", width=10, justify="right")

        for node_id, metrics in node_metrics.items():
            nodes_table.add_row(
                node_id[:15],
                f"{metrics['avg_latency']:.3f}s",
                f"{metrics['p95_latency']:.3f}s",
                str(metrics['throughput']),
                f"{metrics['success_rate']:.2%}",
                str(metrics['total_requests']),
            )

        output.console.print(nodes_table)
