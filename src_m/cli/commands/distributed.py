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
from ..errors import CLIError, ErrorCode as E


def node_start(host: str, port: int, max_concurrency: int, config_path: Optional[Path]):
    """处理分布式节点命令 - 启动 TTS 节点服务。"""
    output = OutputFormatter(verbose=False)

    output.show_banner()

    output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    output.console.print(f"[bold white]  {Icons.GEAR} PPC10 分布式 TTS 节点[/bold white]")
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
        raise CLIError(E.E_BUSINESS, "用户中断,服务已停止", exit_code=130)
    except CLIError:
        raise
    except Exception as e:
        raise CLIError(
            E.E_BUSINESS,
            f"启动节点失败: {e}",
            hint="检查端口是否被占用或使用其他端口",
        ) from e


def master_start(
    host: str,
    port: int,
    max_concurrency: int,
    config_path: Optional[Path],
    worker_nodes: Optional[list] = None,
    local_fallback: bool = False,
):
    """处理分布式主控命令 - 启动主控服务（仅做调度/转发，不直接执行 TTS）。

    启动 :class:`src_m.infrastructure.processing_unit.MasterUnit` + HTTP 监听
    （``MasterHttpServer``）。主控通过 HTTP 将 convert 请求转发到
    worker 节点，本身不调用 ``TTSExecutor``。当
    ``local_fallback=True`` 且没有 worker 时，会在本地启动一个
    :class:`WorkerUnit` 兜底执行。

    新增路由：
    - ``POST /api/v1/convert``  接收 convert 请求并转发/兜底
    - ``GET  /api/v1/health``   健康检查
    - ``GET  /api/v1/stats``    集群状态/worker 列表
    - ``POST /api/v1/workers``  动态注册 worker
    - ``DELETE /api/v1/workers`` 动态摘除 worker
    """
    output = OutputFormatter(verbose=False)

    output.show_banner()

    output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    output.console.print(f"[bold white]  {Icons.GEAR} PPC10 分布式 TTS 主控[/bold white]")
    output.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n")

    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()

    worker_addresses = list(worker_nodes or [])

    for key, value in {
        "host": host,
        "port": port,
        "max_concurrency": max_concurrency,
        "local_fallback": local_fallback,
    }.items():
        output.console.print(f"[dim]{key}:[/dim] [cyan]{value}[/cyan]")

    if worker_addresses:
        output.console.print(f"[dim]initial workers:[/dim] [cyan]{', '.join(worker_addresses)}[/cyan]")

    output.console.print()

    async def run_master():
        from ...infrastructure.processing_unit import (
            MasterHttpServer,
            MasterUnit,
        )

        master = MasterUnit(
            host=host,
            port=port,
            config=config,
            max_concurrency=max_concurrency,
            worker_addresses=worker_addresses,
            local_fallback=local_fallback,
        )
        await master.start()

        server = MasterHttpServer(master)
        await server.start()

        output.console.print(f"[bold {BrandColors.SUCCESS}]{'═' * 60}[/bold {BrandColors.SUCCESS}]")
        output.console.print(f"[bold white]  {Icons.SUCCESS} PPC10 主控已启动[/bold white]")
        output.console.print(f"[bold {BrandColors.SUCCESS}]{'═' * 60}[/bold {BrandColors.SUCCESS}]\n")

        output.console.print(f"  [bold {BrandColors.INFO}]i 服务地址:[/bold {BrandColors.INFO}] [bold cyan]http://{host}:{port}[/bold cyan]")
        output.console.print(f"  [bold {BrandColors.INFO}]i convert 端点:[/bold {BrandColors.INFO}] [bold cyan]http://{host}:{port}/api/v1/convert[/bold cyan]")
        output.console.print(f"  [bold {BrandColors.INFO}]i 已注册 Worker:[/bold {BrandColors.INFO}] {len(worker_addresses)}")
        output.console.print(f"  [bold {BrandColors.INFO}]i 本地兜底:[/bold {BrandColors.INFO}] {'启用' if local_fallback else '禁用'}")
        output.console.print()
        output.console.print(f"[dim]按 Ctrl+C 停止服务[/dim]\n")

        shutdown = asyncio.Event()
        try:
            await shutdown.wait()
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await server.stop()
            except Exception:  # noqa: BLE001
                pass
            try:
                await master.stop()
            except Exception:  # noqa: BLE001
                pass

    try:
        asyncio.run(run_master())
    except KeyboardInterrupt:
        raise CLIError(E.E_BUSINESS, "用户中断,主控已停止", exit_code=130)
    except CLIError:
        raise
    except Exception as e:
        raise CLIError(
            E.E_BUSINESS,
            f"启动主控失败: {e}",
            hint="检查端口是否被占用或使用其他端口",
        ) from e


def dist_convert(
    input_dir: Path,
    output_dir: Path,
    master_endpoint: str,
    config_path: Optional[Path],
    voice: Optional[str] = None,
    rate: Optional[str] = None,
    concurrency: Optional[int] = None,
    local_fallback: bool = False,
    timeout: float = 3600.0,
):
    """``ppc10 dist convert`` - 把 convert 任务提交到远端主控。

    通过 HTTP 把 :class:`ConvertRequest` 投递到 ``<master_endpoint>/api/v1/convert``。
    这条命令的用途是把 ``ppc10 convert`` 的整段执行搬到主控/worker 集群
    上跑，本机只负责把参数打包。
    """
    output = OutputFormatter(verbose=False)

    output.show_banner()
    output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    output.console.print(f"[bold white]  {Icons.GEAR} PPC10 分布式 convert 提交[/bold white]")
    output.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n")

    config_manager = ConfigManager(config_path)
    _ = config_manager.get_config()  # 触发默认配置初始化

    base = master_endpoint.rstrip("/")
    if not base.startswith("http://") and not base.startswith("https://"):
        base = f"http://{base}"
    url = f"{base}/api/v1/convert"

    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "voice": voice,
        "rate": rate,
        "concurrency": concurrency,
        "local_fallback": local_fallback,
    }

    output.info(f"远端主控: {url}")
    output.info(f"输入目录: {input_dir}")
    output.info(f"输出目录: {output_dir}")
    if voice:
        output.info(f"语音: {voice}")
    if rate:
        output.info(f"音频速度: {rate}")
    if concurrency:
        output.info(f"并发: {concurrency}")
    output.info(f"本地兜底: {'启用' if local_fallback else '禁用'}")

    async def run_submit():
        try:
            import aiohttp  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("aiohttp is required for dist convert") from e

        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.post(url, json=payload) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 400:
                    return False, str(data.get("error") or data)
                return True, data

    try:
        ok, data = asyncio.run(run_submit())
    except KeyboardInterrupt:
        raise CLIError(E.E_BUSINESS, "用户中断", exit_code=130)
    except CLIError:
        raise
    except Exception as e:
        raise CLIError(
            E.E_BUSINESS,
            f"提交到主控失败: {e}",
            hint=f"检查主控是否在 {master_endpoint} 监听且 --local-fallback 状态",
        ) from e

    if not ok:
        raise CLIError(E.E_BUSINESS, f"主控返回错误: {data}")

    output.success_panel(
        "convert 任务已成功完成",
        title="完成",
        details={
            "total": data.get("total", 0),
            "completed": data.get("completed", 0),
            "failed": data.get("failed", 0),
            "duration_seconds": round(float(data.get("duration_seconds", 0.0)), 2),
        },
    )
    return


def distributed_status(config_path: Optional[Path], export: Optional[Path], human: bool = False):
    """处理分布式状态命令 - 查看分布式系统状态。

    默认以单行 JSON 数组输出(脚本友好);``--human`` 切换为 Rich 表格。
    """
    from src_m.cli.typer_app import get_output

    output = get_output()
    if human:
        output.set_mode(json_output=False)
    else:
        output.set_mode(json_output=True)

    if not human:
        # 轻量 banner 不在 JSON 模式展示
        pass
    else:
        output.show_banner()
        output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
        output.console.print(f"[bold white]  {Icons.CHART} PPC10 分布式状态[/bold white]")
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

            # 装配 records + rows
            records = []
            for node in nodes:
                address = f"{node.get('host', 'N/A')}:{node.get('port', 0)}"
                resources = node.get("resources", {})
                records.append({
                    "id": node.get("node_id", "N/A"),
                    "host_port": address,
                    "role": node.get("role", "worker"),
                    "status": node.get("status", "N/A"),
                    "last_seen": node.get("last_seen", "N/A"),
                    "active_tasks": node.get("active_tasks", 0),
                    "cpu_percent": resources.get("cpu_percent", 0),
                    "memory_percent": resources.get("memory_percent", 0),
                })
            headers = ["ID", "Host:Port", "Role", "Status", "Last Seen"]
            rows = [
                [r["id"], r["host_port"], r["role"], r["status"], r["last_seen"]]
                for r in records
            ]

            if output.mode == "json":
                # JSON:输出完整结构(nodes + tasks + stats)
                payload = {
                    "timestamp": datetime.now().isoformat(),
                    "nodes": records,
                    "tasks": tasks,
                    "stats": stats,
                }
                if export:
                    Path(export).write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                sys.stdout.write(json.dumps(payload, ensure_ascii=False))
                sys.stdout.write("\n")
                sys.stdout.flush()
                return True

            # human 模式
            output.console.print(f"[bold {BrandColors.SUCCESS}]{'─' * 60}[/bold {BrandColors.SUCCESS}]")
            output.console.print(f"[bold white]  节点状态[/bold white]")
            output.console.print(f"[bold {BrandColors.SUCCESS}]{'─' * 60}[/bold {BrandColors.SUCCESS}]\n")
            if records:
                output.print_table(headers, rows, title=None)
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
                output.console.print(f"\n[bold {BrandColors.SUCCESS}]i 状态已导出:{export}[/bold {BrandColors.SUCCESS}]")

            output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'─' * 60}[/bold {BrandColors.PRIMARY}]")
            output.console.print(f"[dim]数据更新时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")

            return True

        except CLIError:
            raise
        except Exception as e:
            raise CLIError(
                E.E_BUSINESS,
                f"获取状态失败: {e}",
                hint="确保节点服务已启动并可访问",
            ) from e
        finally:
            try:
                await executor.shutdown()
            except Exception:
                pass

    try:
        success = asyncio.run(run_status())
        if not success:
            raise CLIError(E.E_BUSINESS, "获取分布式状态失败")
    except KeyboardInterrupt:
        raise CLIError(E.E_BUSINESS, "用户中断 (Ctrl+C)", exit_code=130)
    except CLIError:
        raise
    except Exception as e:
        raise CLIError(
            E.E_BUSINESS,
            f"执行失败: {e}",
            hint="使用 --verbose 参数查看详细错误信息",
        ) from e


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
    output.console.print(f"[bold white]  {Icons.GEAR} PPC10 添加分布式节点[/bold white]")
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
        if not success:
            raise CLIError(E.E_BUSINESS, "添加节点失败")
    except KeyboardInterrupt:
        raise CLIError(E.E_BUSINESS, "用户中断", exit_code=130)
    except CLIError:
        raise
    except Exception as e:
        raise CLIError(
            E.E_BUSINESS,
            f"执行失败: {e}",
            hint="使用 --verbose 参数查看详细错误信息",
        ) from e
