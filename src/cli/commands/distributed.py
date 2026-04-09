"""分布式 CLI 命令
提供节点启动、管理和状态查看功能
"""

import typer
import asyncio
import json
from typing import Optional
from pathlib import Path

from ..config import ConfigManager, PPC8Config
from .output import OutputFormatter, console, Icons, BrandColors

app = typer.Typer(
    name="distributed",
    help="分布式 TTS 节点管理",
    add_completion=False,
    rich_markup_mode="rich"
)

_output: Optional[OutputFormatter] = None


def get_output() -> OutputFormatter:
    """获取输出格式化器"""
    global _output
    if _output is None:
        _output = OutputFormatter()
    return _output


@app.command("node")
def node_start(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="节点监听地址"),
    port: int = typer.Option(8000, "--port", "-p", help="节点监听端口"),
    max_concurrency: int = typer.Option(4, "--concurrency", "-c", help="节点最大并发数"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-C", help="配置文件路径"),
):
    """启动 TTS 节点服务
    
    节点服务启动后会在指定端口监听主控端的合成请求。
    
    示例:
        ppc8 dist node --host 0.0.0.0 --port 8000 --concurrency 4
        ppc8 dist node -p 8001 -c 8
    """
    output = get_output()
    
    try:
        from ..distributed.node_server import TTSNodeService
    except ImportError as e:
        output.print_error(f"导入分布式模块失败: {e}")
        output.print_info("请安装依赖: pip install aiohttp edge-tts")
        raise typer.Exit(1)
    
    # 加载配置
    try:
        if config_path:
            config_manager = ConfigManager(config_file=str(config_path))
            config = config_manager.get_config()
        else:
            config_manager = ConfigManager()
            config = config_manager.get_config()
    except Exception as e:
        output.print_warning(f"加载配置失败，使用默认配置: {e}")
        config = PPC8Config()
    
    # 创建节点服务
    node_service = TTSNodeService(
        config=config,
        host=host,
        port=port,
        max_concurrency=max_concurrency,
    )
    
    output.print_header("TTS 节点服务")
    output.print_info(f"节点地址: {host}:{port}")
    output.print_info(f"最大并发: {max_concurrency}")
    output.print_info("按 Ctrl+C 停止服务")
    console.print()
    
    # 启动服务
    async def run_service():
        try:
            await node_service.start()
            # 保持运行
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            output.print_info("\n收到停止信号，正在关闭服务...")
            await node_service.stop()
        except Exception as e:
            output.print_error(f"节点服务异常: {e}")
            await node_service.stop()
            raise
    
    asyncio.run(run_service())


@app.command("status")
def distributed_status(
    config_path: Optional[Path] = typer.Option(None, "--config", "-C", help="配置文件路径"),
    export: Optional[Path] = typer.Option(None, "--export", "-e", help="导出状态为 JSON"),
):
    """查看分布式系统状态
    
    显示所有节点的状态、统计信息和健康状态。
    
    示例:
        ppc8 dist status
        ppc8 dist status --export nodes_status.json
    """
    output = get_output()
    
    # 加载配置
    try:
        if config_path:
            config_manager = ConfigManager(config_file=str(config_path))
            config = config_manager.get_config()
        else:
            config_manager = ConfigManager()
            config = config_manager.get_config()
    except Exception as e:
        output.print_error(f"加载配置失败: {e}")
        raise typer.Exit(1)
    
    if not config.distributed.enabled:
        output.print_warning("分布式模式未启用")
        output.print_info("请在配置文件中设置 distributed.enabled=true")
        raise typer.Exit(0)
    
    output.print_header("分布式系统状态")
    
    # 显示节点信息
    console.print(f"\n[bold]运行模式:[/bold] {config.distributed.mode}")
    console.print(f"[bold]负载均衡:[/bold] {config.distributed.load_balance_strategy}")
    console.print(f"[bold]本地执行:[/bold] {'是' if config.distributed.local_execution else '否'}")
    console.print(f"[bold]节点数量:[/bold] {len(config.distributed.nodes)}")
    console.print()
    
    if not config.distributed.nodes:
        output.print_info("未配置远程节点")
        output.print_info("使用 'ppc8 config set' 命令添加节点:")
        console.print(f"  [cyan]ppc8 config set -k distributed.nodes -v '[{{\"host\": \"192.168.1.100\", \"port\": 8000}}]'[/cyan]")
        return
    
    # 显示节点列表
    from rich.table import Table
    table = Table(title="节点列表")
    table.add_column("节点", style="cyan")
    table.add_column("地址", style="green")
    table.add_column("并发", style="yellow")
    table.add_column("状态", style="magenta")
    
    for node in config.distributed.nodes:
        status = "[green]已配置[/green]" if node.enabled else "[red]已禁用[/red]"
        table.add_row(
            f"node-{node.host}:{node.port}",
            f"{node.host}:{node.port}",
            str(node.max_concurrency),
            status,
        )
    
    console.print(table)
    
    # 导出状态
    if export:
        status_data = {
            "mode": config.distributed.mode,
            "load_balance_strategy": config.distributed.load_balance_strategy,
            "local_execution": config.distributed.local_execution,
            "nodes": [
                {
                    "host": node.host,
                    "port": node.port,
                    "max_concurrency": node.max_concurrency,
                    "enabled": node.enabled,
                }
                for node in config.distributed.nodes
            ],
        }
        
        export.write_text(json.dumps(status_data, indent=2, ensure_ascii=False))
        output.print_success(f"状态已导出: {export}")


@app.command("add-node")
def add_node(
    host: str = typer.Argument(..., help="节点 IP 地址"),
    port: int = typer.Argument(..., help="节点端口"),
    max_concurrency: int = typer.Option(4, "--concurrency", "-c", help="节点最大并发数"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-C", help="配置文件路径"),
    save: bool = typer.Option(True, "--save/--no-save", help="是否保存到配置文件"),
):
    """添加分布式节点
    
    将新节点添加到分布式配置中。
    
    示例:
        ppc8 dist add-node 192.168.1.100 8000
        ppc8 dist add-node 192.168.1.101 8000 -c 8 --save
    """
    output = get_output()
    
    # 加载配置
    try:
        if config_path:
            config_manager = ConfigManager(config_file=str(config_path))
            config = config_manager.get_config()
        else:
            config_manager = ConfigManager()
            config = config_manager.get_config()
    except Exception as e:
        output.print_error(f"加载配置失败: {e}")
        raise typer.Exit(1)
    
    # 添加节点
    from ..config.schema import DistributedNodeConfig
    
    node = DistributedNodeConfig(
        host=host,
        port=port,
        max_concurrency=max_concurrency,
        enabled=True,
    )
    
    config.distributed.nodes.append(node)
    config.distributed.enabled = True
    
    output.print_success(f"节点已添加: {host}:{port}")
    output.print_info(f"最大并发: {max_concurrency}")
    
    if save:
        try:
            config_manager.save()
            output.print_success("配置已保存")
        except Exception as e:
            output.print_warning(f"配置保存失败: {e}")
            output.print_info("配置仅在内存中生效")
    
    # 显示当前所有节点
    console.print("\n[bold]当前节点列表:[/bold]")
    for i, n in enumerate(config.distributed.nodes, 1):
        console.print(f"  {i}. {n.host}:{n.port} (并发: {n.max_concurrency})")
