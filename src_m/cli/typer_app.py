"""Typer application definition - PPC9 command line entry point."""

import typer
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import platform
import psutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ppc9 import __version__

from .output import OutputFormatter, console, Icons, BrandColors


app = typer.Typer(
    name="ppc9",
    help="PPC9 - 冰璃岩项目开发组 (BLY Team) - 终极文本转语音工具",
    add_completion=False,
    rich_markup_mode="rich"
)

_output: Optional[OutputFormatter] = None
_config = None
_extension_loader = None


def get_output() -> OutputFormatter:
    """Get output formatter."""
    global _output
    if _output is None:
        _output = OutputFormatter()
    return _output


def get_extension_loader():
    """Get extension loader, initializing on first access."""
    global _extension_loader
    if _extension_loader is None:
        from src_m.extensions.loader import ExtensionLoader
        from src_m.extensions.fanqie.extension import FanqieExtension
        _extension_loader = ExtensionLoader()
        fanqie_ext = FanqieExtension()
        _extension_loader._loaded_extensions[fanqie_ext.metadata.name] = fanqie_ext
    return _extension_loader


def set_config(config):
    """Set configuration."""
    global _config
    _config = config


def verbose_callback(verbose: bool):
    """Verbose mode callback."""
    output = get_output()
    output.set_verbose(verbose)
    return verbose


@app.callback(invoke_without_command=True)
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
    version: bool = typer.Option(False, "--version", help="显示版本号"),
):
    """PPC9 - 冰璃岩项目开发组 (BLY Team) - 终极文本转语音工具"""

    if version:
        console.print(f"[bold green]PPC9 v{__version__}[/bold green]")
        raise typer.Exit()

    output = get_output()
    output.set_verbose(verbose)

    if verbose:
        output.show_welcome()


@app.command("convert")
def convert(
    input_dir: Path = typer.Argument(..., help="输入目录"),
    output_dir: Path = typer.Argument(..., help="输出目录"),
    voice: Optional[str] = typer.Option(None, "--voice", "-V", help="语音模型（默认使用配置文件）"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", "-c", help="并发数（默认使用配置文件）"),
    preset: str = typer.Option("balanced", "--preset", "-p", help="配置预设"),
    resume: bool = typer.Option(False, "--resume", "-r", help="从上次中断处继续（断点续传）"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", help="检查点文件路径（默认: 输出目录/.ppc9_checkpoint.json）"),
    timeout_multiplier: Optional[float] = typer.Option(None, "--timeout-multiplier", "-t", min=0.5, max=2.0, help="超时倍率（0.5-2.0，默认使用配置文件）"),
    rate: Optional[str] = typer.Option(None, "--rate", help="音频播放速度（如 +10%, -10%, +0%，默认 +0%），范围 -100% 到 +100%"),
    recursive: bool = typer.Option(False, "--recursive", "-R", help="递归处理子目录，保持目录结构"),
    ramp_up: Optional[float] = typer.Option(None, "--ramp-up", help="并发预热时间（秒），从1并发逐步增加到设定并发，规避风控（如 30 表示30秒内完成预热）"),
):
    """批量TTS转换"""
    from .commands.convert import handle_convert
    handle_convert(input_dir, output_dir, voice, concurrency, preset, resume, checkpoint, timeout_multiplier, rate, recursive, ramp_up)


@app.command("split")
def split(
    input_file: Path = typer.Argument(..., help="输入文件"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="输出目录"),
    preset: str = typer.Option("chinese_novel", "--preset", "-p", help="章节预设"),
    custom_rules: Optional[str] = typer.Option(None, "--custom-rules", "-r", help="自定义规则JSON字符串或文件路径"),
    add_title_separator: Optional[bool] = typer.Option(None, "--add-title-separator/--no-add-title-separator", help="是否在章节名后添加等于号分隔符"),
    hierarchical: bool = typer.Option(False, "--hierarchical", "-H", help="启用卷章体层级分割"),
):
    """章节分割"""
    from .commands.split import handle_split
    handle_split(input_file, output_dir, preset, custom_rules, add_title_separator, hierarchical)


@app.command("batch")
def batch(
    source_dir: Path = typer.Argument(..., help="源目录"),
    batch_size: int = typer.Option(None, "--batch-size", "-b", help="每批次文件数"),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="预览模式"),
    group_by_volume: bool = typer.Option(False, "--group-by-volume", "-g", help="按卷归档"),
):
    """批量归档"""
    from .commands.batch import handle_batch
    handle_batch(source_dir, batch_size, dry_run, group_by_volume)


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


@app.command("analyze")
def analyze_cmd(
    performance: bool = typer.Option(False, "--performance", "-p", help="性能分析"),
    config: bool = typer.Option(False, "--config", "-c", help="配置冲突检测"),
    errors: bool = typer.Option(False, "--errors", "-e", help="错误模式识别"),
    dependency: bool = typer.Option(False, "--dependency", help="依赖分析"),
    network: bool = typer.Option(False, "--network", help="网络分析"),
    resource: bool = typer.Option(False, "--resource", help="资源分析"),
    quality: bool = typer.Option(False, "--quality", help="代码质量分析"),
    fix: bool = typer.Option(False, "--fix", "-f", help="自动修复（交互式确认）"),
    export: Optional[str] = typer.Option(None, "--export", "-x", help="导出分析报告为 JSON 文件"),
    diff: bool = typer.Option(False, "--diff", help="与最近历史对比"),
    watch: bool = typer.Option(False, "--watch", "-w", help="持续监控模式"),
    interval: int = typer.Option(60, "--interval", "-i", help="监控间隔(秒)"),
    export_html: Optional[str] = typer.Option(None, "--export-html", help="导出HTML报告"),
):
    """系统深度分析

    对性能瓶颈、配置冲突、错误模式、依赖、网络、资源和代码质量进行深度分析，
    生成健康评分和修复建议。默认运行所有分析模块。

    示例:
        ppc9 analyze                    # 运行全部分析
        ppc9 analyze --performance      # 仅性能分析
        ppc9 analyze --config --fix     # 配置分析并尝试自动修复
        ppc9 analyze --diff             # 与最近历史对比
        ppc9 analyze --watch -i 30      # 每30秒持续监控
        ppc9 analyze --export-html report.html  # 导出HTML报告
    """
    from .commands.analyze import handle_analyze
    handle_analyze(
        performance=performance,
        config=config,
        errors=errors,
        dependency=dependency,
        network=network,
        resource=resource,
        quality=quality,
        fix=fix,
        export=export,
        diff=diff,
        watch=watch,
        interval=interval,
        export_html=export_html,
    )


@app.command("status")
def status_cmd(
    watch: bool = typer.Option(False, "--watch", "-w", help="实时监控模式，每秒刷新数据"),
):
    """显示系统状态监控仪表板

    提供进程信息、系统资源、缓存状态、连接池状态、任务统计的实时监控，
    支持仪表板展示和健康度评分功能。

    示例:
        ppc9 status          # 显示静态状态
        ppc9 status --watch  # 实时监控模式
    """
    from .commands.status import handle_status
    handle_status(watch=watch)


# Distributed command group
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

fanqie_app = None


def get_fanqie_app():
    global fanqie_app
    if fanqie_app is None:
        from .commands.fanqie import fanqie_app as fa
        fanqie_app = fa
    return fanqie_app


app.add_typer(get_fanqie_app(), name="fanqie")

ext_app = None


def get_ext_app():
    global ext_app
    if ext_app is None:
        from .commands.ext import ext_app as ea
        ext_app = ea
    return ext_app


app.add_typer(get_ext_app(), name="ext")

merge_app = None
preview_app = None


def get_merge_app():
    global merge_app
    if merge_app is None:
        from .commands.merge import merge_app as ma
        merge_app = ma
    return merge_app


def get_preview_app():
    global preview_app
    if preview_app is None:
        from .commands.preview import preview_app as pa
        preview_app = pa
    return preview_app


@app.command("merge")
def merge_cmd(
    input_files: List[str] = typer.Argument(..., help="输入音频文件（支持通配符，如 *.mp3）"),
    output: Path = typer.Option(..., "--output", "-o", help="输出文件路径"),
    silence: int = typer.Option(500, "--silence", "-s", help="音频片段间的静音间隔（毫秒）"),
    format: str = typer.Option("mp3", "--format", "-f", help="输出格式：mp3、wav、ogg"),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", help="是否归一化音量"),
):
    """合并多个音频文件

    示例:
        ppc9 merge file1.mp3 file2.mp3 -o output.mp3
        ppc9 merge *.mp3 -o all.mp3 --silence 500
        ppc9 merge audio/ -o combined.mp3 --format mp3
    """
    from .commands.merge import handle_merge
    handle_merge(input_files, output, silence, format, normalize)


@app.command("preview")
def preview_cmd(
    text: str = typer.Argument(..., help="要预览的文本内容"),
    voice: str = typer.Option("zh-CN-XiaoxiaoNeural", "--voice", "-V", help="语音名称"),
    rate: str = typer.Option("+0%", "--rate", "-r", help="语速调整"),
    output: Path = typer.Option(None, "--output", "-o", help="输出文件路径"),
    duration: int = typer.Option(10, "--duration", "-d", help="预览最大时长（秒）"),
):
    """预览 TTS 音频片段

    示例:
        ppc9 preview "这是一段预览文本"
        ppc9 preview "快速语音" --voice zh-CN-YunxiNeural --rate +20%
    """
    from .commands.preview import handle_preview
    handle_preview(text, voice, rate, output, duration)


def run():
    """Run application."""
    app()


if __name__ == "__main__":
    run()
