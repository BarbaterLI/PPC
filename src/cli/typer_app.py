"""Typer application definition - PPC_NEXT command line entry point."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
import typer

if TYPE_CHECKING:
    from src.config.schema import PPCConfig
    from src.extensions.loader import ExtensionLoader

logger = logging.getLogger(__name__)

import contextlib

from ..core.exceptions import CLIError, ErrorCodes
from .app_registration import (
    is_ext_invocation,
    register_extension_cli_apps,
    register_ext_app,
)
from .output import OutputFormatter, setup_logging

# ---------------------------------------------------------------------------
# 根 Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="ppc",
    help="PPC_NEXT - BarbaterLI项目开发组 (BLY Team) - 终极文本转语音工具",
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,  # 由我们的 CLIError handler 接管
    pretty_exceptions_show_locals=False,
)

_output: OutputFormatter | None = None
_config = None
_extension_loader = None


def get_output() -> OutputFormatter:
    """Get output formatter."""
    global _output
    if _output is None:
        _output = OutputFormatter()
    return _output


def get_extension_loader() -> ExtensionLoader:
    """Get extension loader, initializing on first access.

    Loads built-in extensions (fanqie) from src/extensions/.
    """
    global _extension_loader
    if _extension_loader is None:
        from src.extensions.loader import ExtensionLoader

        loader = ExtensionLoader()
        try:
            import asyncio

            try:
                asyncio.get_running_loop()
                # A loop is running - skip extension loading in async context
                import warnings

                warnings.warn("Extension loading skipped in async context", RuntimeWarning, stacklevel=2)
            except RuntimeError:
                # No running loop, safe to use asyncio.run()
                asyncio.run(loader.load_all_extensions())
        except Exception as e:
            logger.warning("Failed to load extensions: %s", e)
        _extension_loader = loader
    return _extension_loader


def set_config(config: PPCConfig) -> None:
    """Set configuration."""
    global _config
    _config = config


def _build_help_commands() -> dict:
    """构建帮助命令字典。"""
    help_file = Path(__file__).parent / "help_commands.json"
    with help_file.open("r", encoding="utf-8") as f:
        data: dict = json.load(f)
    return data


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
    no_emoji: bool = typer.Option(False, "--no-emoji", help="使用 ASCII 图标替代 emoji"),
    timestamps: bool = typer.Option(False, "--timestamps", help="为人类可读日志添加时间戳前缀"),
    version: bool = typer.Option(False, "--version", "-V", help="显示版本信息并退出"),
    strict: bool = typer.Option(
        False, "--strict/--no-strict", help="严格模式:将 warning 视作 error(空输入目录 → 退出码 2)"
    ),
):
    """PPC_NEXT - BarbaterLI项目开发组 (BLY Team) - 终极文本转语音工具

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
        no_emoji=no_emoji,
        timestamps=timestamps,
    )

    if version:
        output.print_version_card()
        raise typer.Exit()

    # 无子命令时渲染 WelcomeLayout 而非 Typer 默认帮助
    if ctx.invoked_subcommand is None:
        output.show_welcome()
        raise typer.Exit()


# ---------------------------------------------------------------------------
# 帮助命令（交互式浏览器）
# ---------------------------------------------------------------------------


@app.command("help")
def help_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="非交互模式：输出命令索引 JSON 后退出"),
):
    """进入 PPC_NEXT 交互式帮助浏览器

    列出所有命令，支持上下导航、搜索、查看详情。
    按 q 或 Ctrl+C 退出。

    Examples:
        ppc help
        ppc --no-emoji help
        ppc help --json
    """
    output = get_output()
    if json_output:
        output.set_mode(json_output=True)
    commands = _build_help_commands()
    if output.mode in ("json", "quiet"):
        # 非交互模式：输出命令索引 JSON 后退出，避免挂起等待输入
        sys.stdout.write(json.dumps(commands, ensure_ascii=False))
        sys.stdout.write("\n")
        sys.stdout.flush()
        return
    output.interactive_help(commands)


@app.command("webui")
def webui_cmd(
    host: str = typer.Option("0.0.0.0", "--host", help="监听地址（默认 0.0.0.0）"),
    port: int = typer.Option(5000, "--port", "-p", help="监听端口（默认 5000）"),
    debug: bool = typer.Option(False, "--debug", help="Flask 调试模式（启用时 threaded=False）"),
    api_token: str | None = typer.Option(
        None,
        "--api-token",
        help="API Bearer Token（未设置且绑定非回环地址时输出安全警告）",
    ),
):
    """启动 WebUI（Flask + 前端静态资源）

    自动构建前端（若 webui/WebUI-Vue/dist 缺失）、托管 /api/* 接口与 SPA 页面。
    兼容旧用法：``ppc --webui [--host H] [--port P] [--debug] [--api-token T]``。

    Examples:
        ppc webui
        ppc webui --host 127.0.0.1 --port 5001
        ppc webui --debug --api-token my-token
    """
    from .commands.webui import handle_webui

    handle_webui(host=host, port=port, debug=debug, api_token=api_token)


# ---------------------------------------------------------------------------
# 业务命令
# ---------------------------------------------------------------------------


@app.command("convert")
def convert(
    ctx: typer.Context,
    input: Path = typer.Argument(..., help="输入文件(--one)或输入目录"),
    output: Path | None = typer.Argument(None, help="输出目录；--one 缺省时与输入同目录"),
    voice: str | None = typer.Option(None, "--voice", help="语音模型(默认使用配置文件)"),
    concurrency: int | None = typer.Option(None, "--concurrency", "-c", help="并发数(默认使用配置文件)"),
    preset: str = typer.Option("balanced", "--preset", "-p", help="配置预设"),
    resume: bool = typer.Option(False, "--resume", "-r", help="从上次中断处继续(断点续传)"),
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint", help="检查点文件路径(默认: 输出目录/.ppc_checkpoint.json)"
    ),
    timeout_multiplier: float | None = typer.Option(
        None, "--timeout-multiplier", "-t", min=0.5, max=2.0, help="超时倍率(0.5-2.0,默认使用配置文件)"
    ),
    timeout_mode: str | None = typer.Option(
        None, "--timeout-mode", help="超时模式(fixed|auto|adaptive|synthetic,默认使用配置文件)"
    ),
    timeout: int | None = typer.Option(None, "--timeout", help="固定超时时间(秒)，仅 timeout_mode=fixed 时生效"),
    rate: str | None = typer.Option(None, "--rate", help="音频播放速度(如 +10%, -10%, +0%,范围 -100% 到 +100%)"),
    recursive: bool = typer.Option(False, "--recursive", "-R", help="递归处理子目录,保持目录结构"),
    ramp_up: float | None = typer.Option(
        None, "--ramp-up", help="并发预热时间(秒),从1并发逐步增加到设定并发,规避风控(如 30 表示30秒内完成预热)"
    ),
    one: bool = typer.Option(False, "--one", help="单文件模式：单次无超时、无限重试，缺省输出与输入同目录"),
    tts_backend: str | None = typer.Option(
        None,
        "--tts-backend",
        click_type=click.Choice(["auto", "builtin", "env"], case_sensitive=False),
        help="覆盖 config.yml 的 tts.anti_risk_control.backend（优先级：CLI > 配置文件 > 默认值）/ Override tts.anti_risk_control.backend (priority: CLI > config > default)",
    ),
    tts_proxy: str | None = typer.Option(
        None,
        "--tts-proxy",
        help="覆盖 config.yml 的 tts.anti_risk_control.proxy / Override tts.anti_risk_control.proxy",
    ),
):
    """批量 TTS 转换

    把 input 下的 .txt 文件批量转换为 .mp3 音频,输出到 output。
    --one 时 input 必须为单文件,output 可省略(默认与输入同目录)。

    Examples:
        ppc convert ./txt ./out
        ppc convert ./txt ./out --voice zh-CN-XiaoxiaoNeural -c 8
        ppc convert ./txt ./out --rate +10% -r -t 1.5
        ppc convert ./txt ./out --resume --ramp-up 30
        ppc convert ./book.txt --one
        ppc convert ./book.txt ./audios --one
    """
    strict = bool((ctx.obj or {}).get("strict", False))
    from .commands.convert import ConvertOptions, handle_convert

    handle_convert(
        ConvertOptions(
            input=input,
            output=output,
            voice=voice,
            concurrency=concurrency,
            preset=preset,
            resume=resume,
            checkpoint=checkpoint,
            timeout_multiplier=timeout_multiplier,
            rate=rate,
            recursive=recursive,
            ramp_up=ramp_up,
            strict=strict,
            one=one,
            timeout_mode=timeout_mode,
            timeout=timeout,
            tts_backend=tts_backend,
            tts_proxy=tts_proxy,
        )
    )


@app.command("resume")
def resume(
    ctx: typer.Context,
    input: Path = typer.Argument(..., help="输入目录（含原始 .txt 文件）"),
    output: Path = typer.Argument(..., help="输出目录（含 .cache 子目录）"),
    voice: str | None = typer.Option(None, "--voice", help="语音模型(默认使用配置文件)"),
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint", help="检查点文件路径(默认: 输出目录/.ppc_checkpoint.json)"
    ),
):
    """从 .cache 分段重建断点续传检查点

    适用于之前未启用 --resume 但中途中断的场景。
    扫描 output 下的 .cache，根据已存在的段文件和对应 input 文件生成 checkpoint，
    之后可运行 `ppc convert <input> <output> --resume` 继续完成转换。

    Examples:
        ppc resume ./txt ./out
        ppc resume ./txt ./out --voice zh-CN-XiaoxiaoNeural
        ppc resume ./txt ./out --checkpoint ./my_checkpoint.json
    """
    from .commands.resume import handle_resume

    handle_resume(input, output, voice=voice, checkpoint_path=checkpoint)


@app.command("split")
def split(
    ctx: typer.Context,
    input_file: Path = typer.Argument(..., help="输入文件"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="输出目录（默认：<文件名>_chapters）"),
    preset: str = typer.Option("chinese_novel", "--preset", "-p", help="章节预设"),
    custom_rules: str | None = typer.Option(None, "--custom-rules", "-r", help="自定义规则JSON字符串或文件路径"),
    add_title_separator: bool | None = typer.Option(
        None, "--add-title-separator/--no-add-title-separator", help="是否在章节名后添加等于号分隔符"
    ),
    hierarchical: bool = typer.Option(False, "--hierarchical", "-H", help="启用卷章体层级分割"),
):
    """章节分割

    按预设或自定义规则把整本小说文本切分为若干章节,输出到 output_dir。

    Examples:
        ppc split novel.txt
        ppc split novel.txt -o ./chapters -p chinese_novel
        ppc split novel.txt -H --add-title-separator
        ppc split novel.txt -r rules.json
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
        ppc batch ./txt -b 50
        ppc batch ./txt --dry-run --group-by-volume
        ppc batch ./txt -g -b 100
    """
    strict = bool((ctx.obj or {}).get("strict", False))
    from .commands.batch import handle_batch

    handle_batch(source_dir, batch_size, dry_run, group_by_volume, strict)


@app.command("config")
def config_cmd(
    action: str = typer.Argument(..., help="操作: show/get/set/reset/export/import/init/path/wizard"),
    key: str | None = typer.Option(None, "--key", "-k", help="配置键"),
    value: str | None = typer.Option(None, "--value", "-v", help="配置值"),
    preset: str | None = typer.Option(None, "--preset", "-p", help="预设"),
    temp: bool = typer.Option(False, "--temp", help="临时设置"),
    export_path: Path | None = typer.Option(None, "--export", "-e", help="导出路径"),
    import_path: Path | None = typer.Option(None, "--import", "-i", help="导入路径"),
    full: bool = typer.Option(False, "--full", "-f", help="完整配置模式(wizard 专用)"),
    factory: bool = typer.Option(
        False, "--factory", help="出厂重置(reset 专用): 重新写入完整带注释模板"
    ),
):
    """配置管理

    可用操作:
      show    - 显示当前完整配置
      get     - 获取指定配置项的值
      set     - 设置配置项
      reset   - 重置为预设配置（--factory 时重写完整带注释模板）
      export  - 导出配置到文件
      import  - 从文件导入配置
      init    - 初始化配置文件
      path    - 显示配置文件路径
      wizard  - 启动交互式配置向导

    Examples:
        ppc config show
        ppc config get --key tts.voice
        ppc config set --key tts.voice --value zh-CN-XiaoxiaoNeural
        ppc config reset --preset aggressive
        ppc config reset --factory
        ppc config wizard
    """
    from .commands.config import handle_config

    handle_config(action, key, value, preset, temp, export_path, import_path, full, factory=factory)


@app.command("voices")
def voices_cmd():
    """列出可用语音

    列出 Edge TTS 服务提供的所有可用语音。中文语音优先。

    Examples:
        ppc voices
        ppc --json voices
        ppc --json voices | jq '.[0]'
    """
    from .commands.voices import handle_voices

    handle_voices()


@app.command("bench")
def bench_cmd(
    which: str = typer.Argument("all", help="基准项: all|startup|merge|io"),
    iterations: int = typer.Option(5, "--iterations", "-n", help="startup 采样次数"),
    segments: int = typer.Option(60, "--segments", help="merge 段数"),
    files: int = typer.Option(12, "--files", help="io 文件数"),
    save: Path | None = typer.Option(None, "--save", help="保存基线 JSON 到指定路径"),
):
    """性能基线: 冷启动 / 音频合并 / IO 单遍 三元组，可 --save 做前后对照。

    merge 基准需要 ffmpeg（生成 24kHz mono 48k 测试段），缺失时自动跳过。

    Examples:
        ppc bench
        ppc bench startup -n 7
        ppc bench merge --segments 200
        ppc bench --save .bench/baseline.json
    """
    from .commands.bench import handle_bench

    handle_bench(
        which,
        iterations=iterations,
        segments=segments,
        files=files,
        save=save,
        json_output=get_output().mode == "json",
    )


# Distributed command group
dist_app = typer.Typer(help="分布式节点管理")


@dist_app.command("node")
def dist_node(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="节点监听地址"),
    port: int = typer.Option(8000, "--port", "-p", help="节点监听端口"),
    max_concurrency: int = typer.Option(4, "--concurrency", "-c", help="节点最大并发数"),
    config_path: Path | None = typer.Option(None, "--config", "-C", help="配置文件路径"),
):
    """启动 TTS 节点服务

    启动一个 TTS worker 节点,接收主控的 convert 任务并执行。

    Examples:
        ppc dist node
        ppc dist node --host 0.0.0.0 --port 8000 -c 4
        ppc dist node -p 9001 -C /path/to/config.yml
    """
    from .commands.distributed import node_start

    node_start(host, port, max_concurrency, config_path)


@dist_app.command("master")
def dist_master(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="主控监听地址"),
    port: int = typer.Option(9000, "--port", "-p", help="主控监听端口"),
    max_concurrency: int = typer.Option(4, "--concurrency", "-c", help="主控最大并发数(仅本地兜底使用)"),
    config_path: Path | None = typer.Option(None, "--config", "-C", help="配置文件路径"),
    add_worker: list[str] = typer.Option(
        [],
        "--add-worker",
        help="注册 worker 地址,格式为 host:port 或 http://host:port,可多次传入",
    ),
    local_fallback: bool = typer.Option(False, "--local-fallback", help="在没有 worker 时,本地兜底执行 TTS"),
):
    """启动 PPC_NEXT 分布式主控服务(仅调度/转发)

    主控把 convert 任务按策略分发给已注册的 worker 节点;无 worker
    且 --local-fallback 时,在本机兜底执行。

    Examples:
        ppc dist master
        ppc dist master --port 9000
        ppc dist master --add-worker 10.0.0.1:8000 --add-worker 10.0.0.2:8000
        ppc dist master --local-fallback
    """
    from .commands.distributed import master_start

    normalised: list[str] = []
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
    config_path: Path | None = typer.Option(None, "--config", "-C", help="配置文件路径"),
    export: Path | None = typer.Option(None, "--export", "-e", help="导出状态为 JSON"),
    human: bool = typer.Option(False, "--human", help="人类可读表格输出(默认 JSON)"),
):
    """查看分布式系统状态

    默认以单行 JSON 输出(dist status 默认脚本友好);``--human`` 切换为
    人类可读表格。结构化 JSON 输出沿用根级 ``--json``(默认即 JSON)。

    Examples:
        ppc dist status            # 默认 JSON
        ppc dist status --human    # 表格输出
        ppc dist status --export status.json
    """
    output = get_output()
    output.set_mode(json_output=not human)
    from .commands.distributed import distributed_status

    distributed_status(config_path, export, human=human)


@dist_app.command("add-node")
def dist_add_node(
    host: str = typer.Argument(..., help="节点 IP 地址"),
    port: int = typer.Argument(..., help="节点端口"),
    max_concurrency: int = typer.Option(4, "--concurrency", "-c", help="节点最大并发数"),
    config_path: Path | None = typer.Option(None, "--config", "-C", help="配置文件路径"),
    save: bool = typer.Option(True, "--save/--no-save", help="是否保存到配置文件"),
):
    """添加分布式节点

    把指定 host:port 注册为分布式集群的 worker 节点。

    Examples:
        ppc dist add-node 10.0.0.1 8000
        ppc dist add-node 10.0.0.1 8000 -c 8 --no-save
    """
    from .commands.distributed import add_node

    add_node(host, port, max_concurrency, config_path, save)


@dist_app.command("convert")
def dist_convert(
    input_dir: Path = typer.Argument(..., help="输入目录(与 ppc convert 一致)"),
    output_dir: Path = typer.Argument(..., help="输出目录(与 ppc convert 一致)"),
    master_endpoint: str = typer.Option(
        "http://127.0.0.1:9000",
        "--master",
        "-m",
        help="主控端点(host:port 或 http://host:port)",
    ),
    config_path: Path | None = typer.Option(None, "--config", "-C", help="配置文件路径"),
    voice: str | None = typer.Option(None, "--voice", help="语音模型(默认使用配置文件)"),
    rate: str | None = typer.Option(None, "--rate", help="音频播放速度"),
    concurrency: int | None = typer.Option(None, "--concurrency", "-c", help="并发数"),
    local_fallback: bool = typer.Option(
        False,
        "--local-fallback/--no-local-fallback",
        help="当主控无可用 worker 时允许本地兜底",
    ),
    timeout: float = typer.Option(3600.0, "--timeout", help="HTTP 请求超时(秒)"),
):
    """把 convert 任务提交到远端主控节点

    等价于 ppc convert 的参数,但实际执行发生在
    <master_endpoint>/api/v1/convert 端。用于把单台机器的批量
    convert 派发到分布式集群。

    Examples:
        ppc dist convert ./txt ./out
        ppc dist convert ./txt ./out --master http://10.0.0.1:9000
        ppc dist convert ./txt ./out --voice zh-CN-YunxiNeural -c 16
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
# 别名：文档与旧脚本中出现的 `ppc distributed ...` 同样可用（隐藏于 --help）。
app.add_typer(dist_app, name="distributed", hidden=True)

register_ext_app(app)


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

    # 仅在用户调用 ext 命令时加载扩展并注册 CLI 子命令
    if is_ext_invocation(sys.argv[1:]):
        register_extension_cli_apps(app, get_extension_loader)
    # mvp-cleanup:``--help`` 一律由 Typer 渲染(标准简洁版);
    # ``--version / -V`` 走 Rich 面板。
    args = sys.argv[1:]
    has_subcommand = any(not a.startswith("-") for a in args)
    if not has_subcommand:
        # 先应用公共开关(--json/--quiet/--verbose/--no-color/--no-emoji/--timestamps),让版本 /
        # help 的输出也能遵循它们。
        output = get_output()
        output.set_mode(
            verbose=("--verbose" in args or "-v" in args),
            quiet=("--quiet" in args or "-q" in args),
            json_output=("--json" in args),
            no_color=("--no-color" in args),
            no_emoji=("--no-emoji" in args),
            timestamps=("--timestamps" in args),
        )
        if "-V" in args or "--version" in args:
            output.print_version_card()
            return

    # 先剥掉 typer 注入的 developer-exception 标记,然后调用 app();
    # 我们自己的 sys.excepthook 会渲染 CLIError 并按 exit_code 退出。
    try:
        # 提前初始化 logging:Typer callback 会在后续通过 set_mode() 设置
        # OutputFormatter 模式,这里只需让 RichHandler 先挂载到根 logger。
        _args = sys.argv[1:]
        _verbose = "--verbose" in _args or "-v" in _args
        _quiet = "--quiet" in _args or "-q" in _args
        _json_mode = "--json" in _args
        setup_logging(verbose=_verbose, quiet=_quiet, json_mode=_json_mode)
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
            with contextlib.suppress(Exception):
                delattr(exc, "__typer_developer_exception__")
        _cli_error_handler(exc)
    except SystemExit:
        raise
    except Exception as exc:
        # 遇到 Windows 权限问题(win13/1313/ACCESS_DENIED/PermissionError)时,
        # 先尝试 UAC 提权重启,而不是直接包装为业务错误。
        from src.utils.elevation import handle_permission_error_or_elevate

        if handle_permission_error_or_elevate(exc):
            sys.exit(0)  # 已发起提权,原进程退出
        wrapped = CLIError(ErrorCodes.E_BUSINESS, str(exc), hint="使用 --verbose 查看详细错误信息")
        wrapped.__cause__ = None
        _cli_error_handler(wrapped)


if __name__ == "__main__":
    run()
