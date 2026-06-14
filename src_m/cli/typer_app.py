"""Typer application definition - PPC10 command line entry point."""

import sys
import logging
from pathlib import Path
from typing import List, Optional

import typer

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ppc10 import __version__

from .output import OutputFormatter, console
from .errors import CLIError, ErrorCode


# ---------------------------------------------------------------------------
# 根 Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="ppc10",
    help="PPC10 - 冰璃岩项目开发组 (BLY Team) - 终极文本转语音工具",
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,  # 由我们的 CLIError handler 接管
    pretty_exceptions_show_locals=False,
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
    """Get extension loader, initializing on first access.

    Loads built-in extensions (fanqie, pipeline) from src_m/extensions/.
    """
    global _extension_loader
    if _extension_loader is None:
        from src_m.extensions.loader import ExtensionLoader
        loader = ExtensionLoader()
        try:
            import asyncio
            try:
                asyncio.get_running_loop()
                # A loop is running - skip extension loading in async context
                import warnings
                warnings.warn("Extension loading skipped in async context", RuntimeWarning)
            except RuntimeError:
                # No running loop, safe to use asyncio.run()
                asyncio.run(loader.load_all_extensions())
        except Exception as e:
            logger.warning("Failed to load extensions: %s", e)
        _extension_loader = loader
    return _extension_loader


def set_config(config):
    """Set configuration."""
    global _config
    _config = config


# ---------------------------------------------------------------------------
# 根回调:统一公共开关 + 自定义 --help / --version
# ---------------------------------------------------------------------------


def _maybe_handle_root_version(ctx: typer.Context) -> bool:
    """在 Typer 自己处理 ``--version / -V`` 之前拦截。

    用途:无子命令时,直接走 Rich 面板输出 :meth:`OutputFormatter.print_version_card`,
    避免 Typer 默认 ``--version`` 的纯文本。``--help`` 一律交由 Typer
    自身渲染(标准简洁版)。
    """
    args = sys.argv[1:]

    # 仅在根(没有子命令)时拦截
    has_subcommand = any(not a.startswith("-") for a in args)

    if not has_subcommand and ("-V" in args or "--version" in args):
        get_output().print_version_card()
        raise typer.Exit()

    return False


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出(追加 stack trace)"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="静默模式(仅打印结果摘要与错误)"),
    json_output: bool = typer.Option(False, "--json", help="结构化 JSON 输出(脚本化场景)"),
    no_color: bool = typer.Option(False, "--no-color", help="关闭 ANSI 颜色"),
    version: bool = typer.Option(False, "--version", "-V", help="显示版本信息并退出"),
    strict: bool = typer.Option(False, "--strict/--no-strict", help="严格模式:将 warning 视作 error(空输入目录 → 退出码 2)"),
):
    """PPC10 - 冰璃岩项目开发组 (BLY Team) - 终极文本转语音工具

    ``--help`` 输出 Typer 标准简洁帮助;``--version`` / ``-V`` 走 Rich 面板。"""

    # 先拦截 --version,标准 --help 由 Typer 自己渲染
    _maybe_handle_root_version(ctx)

    # 共享给子命令
    ctx.ensure_object(dict)
    ctx.obj["strict"] = strict

    output = get_output()
    output.set_mode(
        verbose=verbose,
        quiet=quiet,
        json_output=json_output,
        no_color=no_color,
    )

    if version:
        output.print_version_card()
        raise typer.Exit()

    # 无子命令时让 Typer 自己渲染标准 help(简洁版)


# ---------------------------------------------------------------------------
# 业务命令
# ---------------------------------------------------------------------------


@app.command("convert")
def convert(
    ctx: typer.Context,
    input: Path = typer.Argument(..., help="输入文件(--one)或输入目录"),
    output: Optional[Path] = typer.Argument(None, help="输出目录；--one 缺省时与输入同目录"),
    voice: Optional[str] = typer.Option(None, "--voice", help="语音模型(默认使用配置文件)"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", "-c", help="并发数(默认使用配置文件)"),
    preset: str = typer.Option("balanced", "--preset", "-p", help="配置预设"),
    resume: bool = typer.Option(False, "--resume", "-r", help="从上次中断处继续(断点续传)"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", help="检查点文件路径(默认: 输出目录/.ppc10_checkpoint.json)"),
    timeout_multiplier: Optional[float] = typer.Option(None, "--timeout-multiplier", "-t", min=0.5, max=2.0, help="超时倍率(0.5-2.0,默认使用配置文件)"),
    rate: Optional[str] = typer.Option(None, "--rate", help="音频播放速度(如 +10%, -10%, +0%,范围 -100% 到 +100%)"),
    recursive: bool = typer.Option(False, "--recursive", "-R", help="递归处理子目录,保持目录结构"),
    ramp_up: Optional[float] = typer.Option(None, "--ramp-up", help="并发预热时间(秒),从1并发逐步增加到设定并发,规避风控(如 30 表示30秒内完成预热)"),
    one: bool = typer.Option(False, "--one", help="单文件模式：单次无超时、无限重试，缺省输出与输入同目录"),
):
    """批量 TTS 转换

    把 input 下的 .txt 文件批量转换为 .mp3 音频,输出到 output。
    --one 时 input 必须为单文件,output 可省略(默认与输入同目录)。

    Examples:
        ppc10 convert ./txt ./out
        ppc10 convert ./txt ./out --voice zh-CN-XiaoxiaoNeural -c 8
        ppc10 convert ./txt ./out --rate +10% -r -t 1.5
        ppc10 convert ./txt ./out --resume --ramp-up 30
        ppc10 convert ./book.txt --one
        ppc10 convert ./book.txt ./audios --one
    """
    strict = bool((ctx.obj or {}).get("strict", False))
    from .commands.convert import handle_convert
    handle_convert(input, output, voice, concurrency, preset, resume, checkpoint,
                   timeout_multiplier, rate, recursive, ramp_up, strict, one=one)


@app.command("split")
def split(
    ctx: typer.Context,
    input_file: Path = typer.Argument(..., help="输入文件"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="输出目录"),
    preset: str = typer.Option("chinese_novel", "--preset", "-p", help="章节预设"),
    custom_rules: Optional[str] = typer.Option(None, "--custom-rules", "-r", help="自定义规则JSON字符串或文件路径"),
    add_title_separator: Optional[bool] = typer.Option(None, "--add-title-separator/--no-add-title-separator", help="是否在章节名后添加等于号分隔符"),
    hierarchical: bool = typer.Option(False, "--hierarchical", "-H", help="启用卷章体层级分割"),
):
    """章节分割

    按预设或自定义规则把整本小说文本切分为若干章节,输出到 output_dir。

    Examples:
        ppc10 split novel.txt
        ppc10 split novel.txt -o ./chapters -p chinese_novel
        ppc10 split novel.txt -H --add-title-separator
        ppc10 split novel.txt -r rules.json
    """
    strict = bool((ctx.obj or {}).get("strict", False))
    from .commands.split import handle_split
    handle_split(input_file, output_dir, preset, custom_rules, add_title_separator, hierarchical, strict)


@app.command("batch")
def batch(
    ctx: typer.Context,
    source_dir: Path = typer.Argument(..., help="源目录"),
    batch_size: int = typer.Option(None, "--batch-size", "-b", help="每批次文件数"),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="预览模式"),
    group_by_volume: bool = typer.Option(False, "--group-by-volume", "-g", help="按卷归档"),
):
    """批量归档

    按批次或按卷把源目录中的 .txt 文件分批归档,支持预览模式。

    Examples:
        ppc10 batch ./txt -b 50
        ppc10 batch ./txt --dry-run --group-by-volume
        ppc10 batch ./txt -g -b 100
    """
    strict = bool((ctx.obj or {}).get("strict", False))
    from .commands.batch import handle_batch
    handle_batch(source_dir, batch_size, dry_run, group_by_volume, strict)


@app.command("config")
def config_cmd(
    action: str = typer.Argument(..., help="操作: show/get/set/reset/export/import/init/path/wizard"),
    key: Optional[str] = typer.Option(None, "--key", "-k", help="配置键"),
    value: Optional[str] = typer.Option(None, "--value", "-v", help="配置值"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="预设"),
    temp: bool = typer.Option(False, "--temp", help="临时设置"),
    export_path: Optional[Path] = typer.Option(None, "--export", "-e", help="导出路径"),
    import_path: Optional[Path] = typer.Option(None, "--import", "-i", help="导入路径"),
    full: bool = typer.Option(False, "--full", "-f", help="完整配置模式(wizard 专用)"),
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

    Examples:
        ppc10 config show
        ppc10 config get --key tts.voice
        ppc10 config set --key tts.voice --value zh-CN-XiaoxiaoNeural
        ppc10 config reset --preset aggressive
        ppc10 config wizard
    """
    from .commands.config import handle_config
    handle_config(action, key, value, preset, temp, export_path, import_path, full)


@app.command("voices")
def voices_cmd(
    json_output: bool = typer.Option(False, "--json", help="以单行 JSON 数组输出"),
):
    """列出可用语音

    列出 Edge TTS 服务提供的所有可用语音。中文语音优先。

    Examples:
        ppc10 voices
        ppc10 voices --json
        ppc10 voices --json | jq '.[0]'
    """
    from .commands.check import handle_voices
    handle_voices(json_output=json_output)


@app.command("analyze")
def analyze_cmd(
    deep: bool = typer.Option(False, "--deep", help="启用深度分析(默认仅做健康检查)"),
    performance: bool = typer.Option(False, "--performance", "-p", help="性能分析"),
    config: bool = typer.Option(False, "--config", "-c", help="配置冲突检测"),
    errors: bool = typer.Option(False, "--errors", "-e", help="错误模式识别"),
    dependency: bool = typer.Option(False, "--dependency", help="依赖分析"),
    network: bool = typer.Option(False, "--network", help="网络分析"),
    resource: bool = typer.Option(False, "--resource", help="资源分析"),
    quality: bool = typer.Option(False, "--quality", help="代码质量分析"),
    fix: bool = typer.Option(False, "--fix", "-f", help="自动修复(交互式确认)"),
    export: Optional[str] = typer.Option(None, "--export", "-x", help="导出分析报告为 JSON 文件"),
    diff: bool = typer.Option(False, "--diff", help="与最近历史对比"),
    watch: bool = typer.Option(False, "--watch", "-w", help="持续监控模式"),
    interval: int = typer.Option(60, "--interval", "-i", help="监控间隔(秒)"),
    export_html: Optional[str] = typer.Option(None, "--export-html", help="导出HTML报告"),
    full: bool = typer.Option(False, "--full", help="完整检查(健康检查模式专用)"),
):
    """系统分析与健康检查

    默认运行轻量级系统健康检查(系统环境、依赖、网络、文件系统、
    系统资源、配置验证)。使用 --deep 启用深度分析(性能瓶颈、
    配置冲突、错误模式、依赖、网络、资源、代码质量)。两者可叠加。

    Examples:
        ppc10 analyze                              # 健康检查
        ppc10 analyze --fix                        # 健康检查并尝试一键修复
        ppc10 analyze --export report.json         # 导出健康检查结果
        ppc10 analyze --deep                       # 深度分析(默认全部分析模块)
        ppc10 analyze --deep --performance         # 仅性能深度分析
        ppc10 analyze --deep --diff                # 与最近历史对比
        ppc10 analyze --deep --watch -i 30         # 每30秒持续监控
        ppc10 analyze --deep --export-html r.html  # 导出HTML报告
    """
    from .commands.analyze import handle_analyze
    handle_analyze(
        deep=deep,
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
        full=full,
    )


# Distributed command group
dist_app = typer.Typer(help="分布式节点管理")


@dist_app.command("node")
def dist_node(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="节点监听地址"),
    port: int = typer.Option(8000, "--port", "-p", help="节点监听端口"),
    max_concurrency: int = typer.Option(4, "--concurrency", "-c", help="节点最大并发数"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-C", help="配置文件路径"),
):
    """启动 TTS 节点服务

    启动一个 TTS worker 节点,接收主控的 convert 任务并执行。

    Examples:
        ppc10 dist node
        ppc10 dist node --host 0.0.0.0 --port 8000 -c 4
        ppc10 dist node -p 9001 -C /path/to/config.yml
    """
    from .commands.distributed import node_start
    node_start(host, port, max_concurrency, config_path)


@dist_app.command("master")
def dist_master(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="主控监听地址"),
    port: int = typer.Option(9000, "--port", "-p", help="主控监听端口"),
    max_concurrency: int = typer.Option(4, "--concurrency", "-c", help="主控最大并发数(仅本地兜底使用)"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-C", help="配置文件路径"),
    add_worker: List[str] = typer.Option(
        [],
        "--add-worker",
        help="注册 worker 地址,格式为 host:port 或 http://host:port,可多次传入",
    ),
    local_fallback: bool = typer.Option(
        False, "--local-fallback", help="在没有 worker 时,本地兜底执行 TTS"
    ),
):
    """启动 PPC10 分布式主控服务(仅调度/转发)

    主控把 convert 任务按策略分发给已注册的 worker 节点;无 worker
    且 --local-fallback 时,在本机兜底执行。

    Examples:
        ppc10 dist master
        ppc10 dist master --port 9000
        ppc10 dist master --add-worker 10.0.0.1:8000 --add-worker 10.0.0.2:8000
        ppc10 dist master --local-fallback
    """
    from .commands.distributed import master_start

    normalised: List[str] = []
    for entry in add_worker or []:
        entry = entry.strip()
        if not entry:
            continue
        if "://" not in entry:
            entry = f"http://{entry}"
        normalised.append(entry.rstrip("/"))

    master_start(
        host=host,
        port=port,
        max_concurrency=max_concurrency,
        config_path=config_path,
        worker_nodes=normalised,
        local_fallback=local_fallback,
    )


@dist_app.command("status")
def dist_status(
    config_path: Optional[Path] = typer.Option(None, "--config", "-C", help="配置文件路径"),
    export: Optional[Path] = typer.Option(None, "--export", "-e", help="导出状态为 JSON"),
    human: bool = typer.Option(False, "--human", help="人类可读表格输出(默认 JSON)"),
    json_output: bool = typer.Option(False, "--json", help="结构化 JSON 输出(默认即 JSON)"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="静默模式"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
):
    """查看分布式系统状态

    默认以单行 JSON 输出(dist status 默认脚本友好);``--human`` 切换为
    人类可读表格。

    Examples:
        ppc10 dist status            # 默认 JSON
        ppc10 dist status --human    # 表格输出
        ppc10 dist status --export status.json
    """
    from src_m.cli.typer_app import get_output
    output = get_output()
    output.set_mode(
        verbose=verbose,
        quiet=quiet,
        json_output=json_output or (not human),
    )
    from .commands.distributed import distributed_status
    distributed_status(config_path, export, human=human)


@dist_app.command("add-node")
def dist_add_node(
    host: str = typer.Argument(..., help="节点 IP 地址"),
    port: int = typer.Argument(..., help="节点端口"),
    max_concurrency: int = typer.Option(4, "--concurrency", "-c", help="节点最大并发数"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-C", help="配置文件路径"),
    save: bool = typer.Option(True, "--save/--no-save", help="是否保存到配置文件"),
):
    """添加分布式节点

    把指定 host:port 注册为分布式集群的 worker 节点。

    Examples:
        ppc10 dist add-node 10.0.0.1 8000
        ppc10 dist add-node 10.0.0.1 8000 -c 8 --no-save
    """
    from .commands.distributed import add_node
    add_node(host, port, max_concurrency, config_path, save)


@dist_app.command("convert")
def dist_convert(
    input_dir: Path = typer.Argument(..., help="输入目录(与 ppc10 convert 一致)"),
    output_dir: Path = typer.Argument(..., help="输出目录(与 ppc10 convert 一致)"),
    master_endpoint: str = typer.Option(
        "http://127.0.0.1:9000", "--master", "-m",
        help="主控端点(host:port 或 http://host:port)",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config", "-C", help="配置文件路径"),
    voice: Optional[str] = typer.Option(None, "--voice", "-V", help="语音模型"),
    rate: Optional[str] = typer.Option(None, "--rate", help="音频播放速度"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", "-c", help="并发数"),
    local_fallback: bool = typer.Option(
        False, "--local-fallback/--no-local-fallback",
        help="当主控无可用 worker 时允许本地兜底",
    ),
    timeout: float = typer.Option(3600.0, "--timeout", help="HTTP 请求超时(秒)"),
):
    """把 convert 任务提交到远端主控节点

    等价于 ppc10 convert 的参数,但实际执行发生在
    <master_endpoint>/api/v1/convert 端。用于把单台机器的批量
    convert 派发到分布式集群。

    Examples:
        ppc10 dist convert ./txt ./out
        ppc10 dist convert ./txt ./out --master http://10.0.0.1:9000
        ppc10 dist convert ./txt ./out -V zh-CN-YunxiNeural -c 16
    """
    from .commands.distributed import dist_convert as _dist_convert
    _dist_convert(
        input_dir=input_dir,
        output_dir=output_dir,
        master_endpoint=master_endpoint,
        config_path=config_path,
        voice=voice,
        rate=rate,
        concurrency=concurrency,
        local_fallback=local_fallback,
        timeout=timeout,
    )


app.add_typer(dist_app, name="dist")


def _build_ext_app():
    """构建 ext Typer app,包括子命令 + 扩展的 register_cli。"""
    from .commands.ext import ext_app

    loader = get_extension_loader()
    for ext in loader.get_cli_extensions():
        try:
            sub_app = typer.Typer(
                name=ext.metadata.name,
                help=ext.metadata.description or ext.metadata.name,
                add_completion=False,
            )
            ext.register_cli(sub_app)
            ext_app.add_typer(sub_app, name=ext.metadata.name)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to register CLI for extension {ext.metadata.name}: {e}"
            )
    return ext_app


app.add_typer(_build_ext_app(), name="ext")


# ---------------------------------------------------------------------------
# 全局 CLIError 处理器
# ---------------------------------------------------------------------------


def _cli_error_handler(exc: CLIError) -> None:
    """统一渲染 CLIError 并按 exit_code 退出。"""
    output = get_output()
    output.error(exc)
    # 切断异常链,避免 Python 在 sys.exit 时打印 "During handling of ..."。
    # typer.Exit 内部会构造 __context__,改用 sys.exit(code) 走 stdlib 路径。
    sys.exit(exc.exit_code)


# 异常 hook,会被 typer_app.run() 启用(见 run())
_original_excepthook = None


def _install_cli_error_hook() -> None:
    """安装全局 sys.excepthook,把 unhandled CLIError 转为带 exit_code 的退出。"""
    global _original_excepthook
    _original_excepthook = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb):
        if isinstance(exc_value, CLIError):
            _cli_error_handler(exc_value)
            return
        if _original_excepthook is not None:
            _original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def run():
    """Run application."""
    _install_cli_error_hook()
    # mvp-cleanup:``--help`` 一律由 Typer 渲染(标准简洁版);
    # ``--version / -V`` 走 Rich 面板。
    args = sys.argv[1:]
    has_subcommand = any(not a.startswith("-") for a in args)
    if not has_subcommand:
        # 先应用公共开关(--json/--quiet/--verbose/--no-color),让版本 /
        # help 的输出也能遵循它们。
        output = get_output()
        output.set_mode(
            verbose=("--verbose" in args or "-v" in args),
            quiet=("--quiet" in args or "-q" in args),
            json_output=("--json" in args),
            no_color=("--no-color" in args),
        )
        if "-V" in args or "--version" in args:
            output.print_version_card()
            return

    # 先剥掉 typer 注入的 developer-exception 标记,然后调用 app();
    # 我们自己的 sys.excepthook 会渲染 CLIError 并按 exit_code 退出。
    try:
        try:
            app()
        except SystemExit as e:
            if e.code is not None and e.code != 0:
                sys.exit(e.code)
            return
    except CLIError as exc:
        exc.__cause__ = None
        # 清除 typer 注入的 developer exception 配置(若有),确保走 sys.excepthook
        if hasattr(exc, "__typer_developer_exception__"):
            try:
                delattr(exc, "__typer_developer_exception__")
            except Exception:
                pass
        _cli_error_handler(exc)
    except SystemExit:
        raise
    except Exception as exc:
        from .errors import CLIError as _CE, ErrorCode as _EC
        wrapped = _CE(_EC.E_BUSINESS, str(exc), hint="使用 --verbose 查看堆栈")
        wrapped.__cause__ = None
        _cli_error_handler(wrapped)


if __name__ == "__main__":
    run()


# ---------------------------------------------------------------------------
# docs 子命令注册
# ---------------------------------------------------------------------------

try:
    from .commands.docs import docs_app  # noqa: E402
    app.add_typer(docs_app, name="docs")
except Exception as e:  # pragma: no cover - 注册失败不应阻塞主流程
    logger.warning("Failed to register docs commands: %s", e)
