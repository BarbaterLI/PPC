"""分布式命令实现 - 多节点 TTS 集群管理。"""

import asyncio
import contextlib
import json
from datetime import datetime
from pathlib import Path

from src.cli.typer_app import get_output

from ...config import ConfigManager
from ...distributed import (
    DistributedTTSExecutor,
    HealthCheckConfig,
    TTSNode,
)
from ..errors import CLIError
from ..errors import ErrorCode as E


def node_start(host: str, port: int, max_concurrency: int, config_path: Path | None):
    """处理分布式节点命令 - 启动 TTS 节点服务。"""
    output = get_output()

    output.show_banner()

    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()

    node_config = {
        "host": host,
        "port": port,
        "max_concurrency": max_concurrency,
    }

    for key, value in node_config.items():
        output.info(f"{key}: {value}")

    async def run_node():
        node = TTSNode(
            host=host,
            port=port,
            max_concurrency=max_concurrency,
            config=config,
            health_check_config=HealthCheckConfig(),
        )

        output.success_panel(
            "TTS 节点已启动",
            title="服务启动",
            details={
                "服务地址": f"http://{host}:{port}",
                "最大并发": max_concurrency,
                "健康检查": f"http://{host}:{port}/health",
            },
        )
        output.info("按 Ctrl+C 停止服务")

        await node.start()

    try:
        asyncio.run(run_node())
    except KeyboardInterrupt:
        raise CLIError(E.E_BUSINESS, "用户中断,服务已停止", exit_code=130) from None
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
    config_path: Path | None,
    worker_nodes: list | None = None,
    local_fallback: bool = False,
):
    """处理分布式主控命令 - 启动主控服务（仅做调度/转发，不直接执行 TTS）。

    启动 :class:`src.distributed.processing_unit.MasterUnit` + HTTP 监听
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
    output = get_output()

    output.show_banner()

    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()

    worker_addresses = list(worker_nodes or [])

    details = {
        "host": host,
        "port": port,
        "max_concurrency": max_concurrency,
        "local_fallback": local_fallback,
    }
    if worker_addresses:
        details["initial_workers"] = ", ".join(worker_addresses)

    for key, value in details.items():
        output.info(f"{key}: {value}")

    async def run_master():
        from ...distributed.processing_unit import (
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

        output.success_panel(
            "PPC10 主控已启动",
            title="服务启动",
            details={
                "服务地址": f"http://{host}:{port}",
                "convert 端点": f"http://{host}:{port}/api/v1/convert",
                "已注册 Worker": len(worker_addresses),
                "本地兜底": "启用" if local_fallback else "禁用",
            },
        )
        output.info("按 Ctrl+C 停止服务")

        shutdown = asyncio.Event()
        try:
            with contextlib.suppress(asyncio.CancelledError):
                await shutdown.wait()
        finally:
            with contextlib.suppress(Exception):
                await server.stop()
            with contextlib.suppress(Exception):
                await master.stop()

    try:
        asyncio.run(run_master())
    except KeyboardInterrupt:
        raise CLIError(E.E_BUSINESS, "用户中断,主控已停止", exit_code=130) from None
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
    config_path: Path | None,
    voice: str | None = None,
    rate: str | None = None,
    concurrency: int | None = None,
    local_fallback: bool = False,
    timeout: float = 3600.0,
):
    """``ppc10 dist convert`` - 把 convert 任务提交到远端主控。

    通过 HTTP 把 :class:`ConvertRequest` 投递到 ``<master_endpoint>/api/v1/convert``。
    这条命令的用途是把 ``ppc10 convert`` 的整段执行搬到主控/worker 集群
    上跑，本机只负责把参数打包。
    """
    output = get_output()

    output.show_banner()

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
        async with aiohttp.ClientSession(timeout=timeout_obj) as session, session.post(url, json=payload) as resp:
            data = await resp.json(content_type=None)
            if resp.status >= 400:
                return False, str(data.get("error") or data)
            return True, data

    try:
        ok, data = asyncio.run(run_submit())
    except KeyboardInterrupt:
        raise CLIError(E.E_BUSINESS, "用户中断", exit_code=130) from None
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


def distributed_status(config_path: Path | None, export: Path | None, human: bool = False):
    """处理分布式状态命令 - 查看分布式系统状态。

    默认以单行 JSON 数组输出(脚本友好);``--human`` 切换为 Rich 表格。
    """
    output = get_output()

    if output.mode == "quiet":
        return

    if human:
        output.show_banner()
        output.print_panel("PPC10 分布式状态", title="分布式状态", style="primary")

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

            records = []
            for node in nodes:
                address = f"{node.get('host', 'N/A')}:{node.get('port', 0)}"
                resources = node.get("resources", {})
                records.append(
                    {
                        "id": node.get("node_id", "N/A"),
                        "host_port": address,
                        "role": node.get("role", "worker"),
                        "status": node.get("status", "N/A"),
                        "last_seen": node.get("last_seen", "N/A"),
                        "active_tasks": node.get("active_tasks", 0),
                        "cpu_percent": resources.get("cpu_percent", 0),
                        "memory_percent": resources.get("memory_percent", 0),
                    }
                )
            headers = ["ID", "Host:Port", "Role", "Status", "Last Seen"]
            rows = [[r["id"], r["host_port"], r["role"], r["status"], r["last_seen"]] for r in records]

            if output.mode == "json":
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
                output.print_table(headers, rows, title=None, json_data=payload)
                return True

            # human 模式
            output.print_panel("节点状态", title="节点状态", style="success")
            if records:
                output.print_table(headers, rows, title=None)
            else:
                output.info("当前无已注册节点")

            if tasks:
                task_headers = ["状态", "数量"]
                task_rows = [
                    [task_status, str(count)] for task_status, count in tasks.items() if isinstance(count, int)
                ]
                output.print_table(task_headers, task_rows, title="任务状态", style="info")

            if stats:
                stat_rows = []
                if "total_tasks" in stats:
                    stat_rows.append(["总任务数", str(stats["total_tasks"])])
                if "completed_tasks" in stats:
                    stat_rows.append(["完成任务数", str(stats["completed_tasks"])])
                if "failed_tasks" in stats:
                    stat_rows.append(["失败任务数", str(stats["failed_tasks"])])
                if "avg_task_duration" in stats:
                    stat_rows.append(["平均任务耗时", f"{stats['avg_task_duration']:.2f}s"])
                if "current_throughput" in stats:
                    stat_rows.append(["当前吞吐量", f"{stats['current_throughput']:.2f} 任务/秒"])
                if "cluster_uptime" in stats:
                    uptime_seconds = stats["cluster_uptime"]
                    hours = int(uptime_seconds // 3600)
                    minutes = int((uptime_seconds % 3600) // 60)
                    stat_rows.append(["集群运行时间", f"{hours}小时{minutes}分钟"])
                if stat_rows:
                    output.print_table(["指标", "值"], stat_rows, title="集群统计", style="accent")

            if export:
                export_data = {"timestamp": datetime.now().isoformat(), "status": status}
                with open(export, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                output.success(f"状态已导出:{export}")

            output.info(f"数据更新时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
            with contextlib.suppress(Exception):
                await executor.shutdown()

    try:
        success = asyncio.run(run_status())
        if not success:
            raise CLIError(E.E_BUSINESS, "获取分布式状态失败")
    except KeyboardInterrupt:
        raise CLIError(E.E_BUSINESS, "用户中断 (Ctrl+C)", exit_code=130) from None
    except CLIError:
        raise
    except Exception as e:
        raise CLIError(
            E.E_BUSINESS,
            f"执行失败: {e}",
            hint="使用 --verbose 参数查看详细错误信息",
        ) from e


def add_node(host: str, port: int, max_concurrency: int, config_path: Path | None, save: bool):
    """处理分布式添加节点命令 - 向集群添加新节点。"""
    output = get_output()

    output.show_banner()
    output.print_panel("PPC10 添加分布式节点", title="添加节点", style="primary")

    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()

    async def run_add_node():
        executor = DistributedTTSExecutor(config)
        try:
            await executor.initialize()

            node_id = await executor.add_node(
                host=host, port=port, max_concurrency=max_concurrency, health_check_config=HealthCheckConfig()
            )

            details = {
                "节点 ID": node_id,
                "地址": f"{host}:{port}",
                "最大并发": max_concurrency,
            }

            if save:
                try:
                    config_manager.set("distributed.nodes", host)
                    details["已保存"] = "是"
                except Exception as e:
                    output.warning(f"保存节点信息失败：{e}")
                    details["已保存"] = "否"

            output.success_panel(
                "节点添加成功",
                title="完成",
                details=details,
            )
            output.info(f"添加时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            return True

        except Exception as e:
            output.error_panel(
                f"添加节点失败：{e}",
                title="添加错误",
                error_type=type(e).__name__,
                suggestion="检查节点地址和端口是否正确",
            )
            return False
        finally:
            await executor.shutdown()

    try:
        success = asyncio.run(run_add_node())
        if not success:
            raise CLIError(E.E_BUSINESS, "添加节点失败")
    except KeyboardInterrupt:
        raise CLIError(E.E_BUSINESS, "用户中断", exit_code=130) from None
    except CLIError:
        raise
    except Exception as e:
        raise CLIError(
            E.E_BUSINESS,
            f"执行失败: {e}",
            hint="使用 --verbose 参数查看详细错误信息",
        ) from e
