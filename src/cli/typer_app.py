"""Typer应用定义
PPC8命令行应用入口
"""

import typer
from typing import Optional
from pathlib import Path
from datetime import datetime
import platform
import psutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ppc8 import __version__

from .output import OutputFormatter, console, Icons, BrandColors


app = typer.Typer(
    name="ppc8",
    help="PPC8 - 冰璃岩项目开发组 (BLY Team) - 终极文本转语音工具",
    add_completion=False,
    rich_markup_mode="rich"
)

_output: Optional[OutputFormatter] = None
_config = None


def get_output() -> OutputFormatter:
    """获取输出格式化器"""
    global _output
    if _output is None:
        _output = OutputFormatter()
    return _output


def set_config(config):
    """设置配置"""
    global _config
    _config = config


def verbose_callback(verbose: bool):
    """详细模式回调"""
    output = get_output()
    output.set_verbose(verbose)
    return verbose


@app.callback(invoke_without_command=True)
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
    version: bool = typer.Option(False, "--version", help="显示版本号"),
):
    """PPC8 - 冰璃岩项目开发组 (BLY Team) - 终极文本转语音工具"""

    if version:
        console.print(f"[bold green]PPC8 v{__version__}[/bold green]")
        raise typer.Exit()

    output = get_output()
    output.set_verbose(verbose)

    if verbose:
        output.show_welcome()


@app.command("convert")
def convert(
    input_dir: Path = typer.Argument(..., help="输入目录"),
    output_dir: Path = typer.Argument(..., help="输出目录"),
    voice: Optional[str] = typer.Option(None, "--voice", "-v", help="语音模型（默认使用配置文件）"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", "-c", help="并发数（默认使用配置文件）"),
    preset: str = typer.Option("balanced", "--preset", "-p", help="配置预设"),
    resume: bool = typer.Option(False, "--resume", "-r", help="从上次中断处继续（断点续传）"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", help="检查点文件路径（默认: 输出目录/.ppc8_checkpoint.json）"),
    timeout_multiplier: Optional[float] = typer.Option(None, "--timeout-multiplier", "-t", min=0.5, max=2.0, help="超时倍率（0.5-2.0，默认使用配置文件）"),
    rate: Optional[str] = typer.Option(None, "--rate", help="音频播放速度（如 +10%, -10%, +0%，默认 +0%），范围 -100% 到 +100%"),
):
    """批量TTS转换"""
    from .commands.convert import handle_convert
    handle_convert(input_dir, output_dir, voice, concurrency, preset, resume, checkpoint, timeout_multiplier, rate)


@app.command("split")
def split(
    input_file: Path = typer.Argument(..., help="输入文件"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="输出目录"),
    preset: str = typer.Option("chinese_novel", "--preset", "-p", help="章节预设"),
    custom_rules: Optional[str] = typer.Option(None, "--custom-rules", "-r", help="自定义规则JSON字符串或文件路径"),
    add_title_separator: Optional[bool] = typer.Option(None, "--add-title-separator/--no-add-title-separator", help="是否在章节名后添加等于号分隔符"),
):
    """章节分割"""
    from .commands.split import handle_split
    handle_split(input_file, output_dir, preset, custom_rules, add_title_separator)


@app.command("batch")
def batch(
    source_dir: Path = typer.Argument(..., help="源目录"),
    batch_size: int = typer.Option(100, "--batch-size", "-b", help="批次大小"),
    dry_run: bool = typer.Option(False, "--dry-run", help="预览模式"),
):
    """批量归档"""
    from .commands.batch import handle_batch
    handle_batch(source_dir, batch_size, dry_run)


@app.command("config")
def config_cmd(
    action: str = typer.Argument(..., help="操作: show/get/set/reset/export/import/init/path/wizard"),
    key: Optional[str] = typer.Option(None, "--key", "-k", help="配置键"),
    value: Optional[str] = typer.Option(None, "--value", "-v", help="配置值"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="预设"),
    temp: bool = typer.Option(False, "--temp", help="临时设置"),
    export_path: Optional[Path] = typer.Option(None, "--export", "-e", help="导出路径"),
    import_path: Optional[Path] = typer.Option(None, "--import", "-i", help="导入路径"),
    full: bool = typer.Option(False, "--full", "-f", help="完整配置模式（wizard 专用）"),
):
    """配置管理

    可用操作:
      show    - 显示当前完整配置
      get     - 获取指定配置项的值
      set     - 设置配置项
      reset   - 重置为预设配置
      export  - 导出配置到文件
      import  - 从文件导入配置
      init    - 初始化配置文件
      path    - 显示配置文件路径
      wizard  - 启动交互式配置向导
    """
    from .commands.config import handle_config
    handle_config(action, key, value, preset, temp, export_path, import_path, full)


@app.command("check")
def check_cmd(
    full: bool = typer.Option(False, "--full", "-f", help="完整检查"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="导出检查结果为 JSON 文件"),
):
    """系统检查"""
    from .commands.check import handle_check
    handle_check(full, export)


@app.command("voices")
def voices_cmd():
    """列出可用语音"""
    from .commands.check import handle_voices
    handle_voices()


@app.command("status")
def status_cmd(
    watch: bool = typer.Option(False, "--watch", "-w", help="实时监控模式，每秒刷新数据"),
):
    """显示系统状态监控仪表板

    提供进程信息、系统资源、缓存状态、连接池状态、任务统计的实时监控，
    支持仪表板展示和健康度评分功能。

    示例:
        ppc8 status          # 显示静态状态
        ppc8 status --watch  # 实时监控模式
    """
    from .commands.status import handle_status
    handle_status(watch=watch)


# ==================== 分布式命令组 ====================
dist_app = typer.Typer(help="分布式节点管理")


@dist_app.command("node")
def dist_node(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="节点监听地址"),
    port: int = typer.Option(8000, "--port", "-p", help="节点监听端口"),
    max_concurrency: int = typer.Option(4, "--concurrency", "-c", help="节点最大并发数"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-C", help="配置文件路径"),
):
    """启动 TTS 节点服务"""
    from .commands.distributed import node_start
    node_start(host, port, max_concurrency, config_path)


@dist_app.command("status")
def dist_status(
    config_path: Optional[Path] = typer.Option(None, "--config", "-C", help="配置文件路径"),
    export: Optional[Path] = typer.Option(None, "--export", "-e", help="导出状态为 JSON"),
):
    """查看分布式系统状态"""
    from .commands.distributed import distributed_status
    distributed_status(config_path, export)


@dist_app.command("add-node")
def dist_add_node(
    host: str = typer.Argument(..., help="节点 IP 地址"),
    port: int = typer.Argument(..., help="节点端口"),
    max_concurrency: int = typer.Option(4, "--concurrency", "-c", help="节点最大并发数"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-C", help="配置文件路径"),
    save: bool = typer.Option(True, "--save/--no-save", help="是否保存到配置文件"),
):
    """添加分布式节点"""
    from .commands.distributed import add_node
    add_node(host, port, max_concurrency, config_path, save)


app.add_typer(dist_app, name="dist")
# ======================================================


def run():
    """运行应用"""
    app()


if __name__ == "__main__":
    run()
