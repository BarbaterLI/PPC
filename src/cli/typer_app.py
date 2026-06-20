"""Typer application definition - PPC10 command line entry point."""

import json
import logging
import sys
from pathlib import Path

import typer

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import contextlib

from .errors import CLIError, ErrorCode
from .output import OutputFormatter

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

_output: OutputFormatter | None = None
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


def set_config(config):
    """Set configuration."""
    global _config
    _config = config


def _build_help_commands() -> dict[str, dict]:
    """Build static command metadata for the interactive help browser.

    Mirrors the registered Typer commands and their docstrings, providing
    descriptions, usage, examples, options, categories and related commands.
    """
    return {
        "convert": {
            "category": "转换",
            "desc": "批量 TTS 转换。把 input 下的 .txt 文件批量转换为 .mp3 音频，输出到 output。",
            "usage": "ppc10 convert <input> [output] [options]",
            "examples": [
                {"desc": "批量转换目录", "cmd": "ppc10 convert ./txt ./out"},
                {"desc": "指定语音模型与并发", "cmd": "ppc10 convert ./txt ./out --voice zh-CN-XiaoxiaoNeural -c 8"},
                {"desc": "调整语速并断点续传", "cmd": "ppc10 convert ./txt ./out --rate +10% -r -t 1.5"},
                {"desc": "并发预热", "cmd": "ppc10 convert ./txt ./out --resume --ramp-up 30"},
                {"desc": "单文件模式", "cmd": "ppc10 convert ./book.txt --one"},
            ],
            "options": [
                {"name": "--voice", "desc": "语音模型(默认使用配置文件)"},
                {"name": "--concurrency, -c", "desc": "并发数(默认使用配置文件)"},
                {"name": "--preset, -p", "desc": "配置预设"},
                {"name": "--resume, -r", "desc": "从上次中断处继续(断点续传)"},
                {"name": "--checkpoint", "desc": "检查点文件路径"},
                {"name": "--timeout-multiplier, -t", "desc": "超时倍率(0.5-2.0,默认使用配置文件)"},
                {"name": "--rate", "desc": "音频播放速度(如 +10%, -10%, 范围 -100% 到 +100%)"},
                {"name": "--recursive, -R", "desc": "递归处理子目录，保持目录结构"},
                {"name": "--ramp-up", "desc": "并发预热时间(秒)，规避风控"},
                {"name": "--one", "desc": "单文件模式：单次无超时、无限重试"},
            ],
            "see_also": ["split", "batch", "voices", "resume"],
        },
        "resume": {
            "category": "转换",
            "desc": "从已有的 .cache 分段重建断点续传检查点，再使用 convert --resume 继续转换。",
            "usage": "ppc10 resume <input_dir> <output_dir> [options]",
            "examples": [
                {"desc": "重建检查点", "cmd": "ppc10 resume ./txt ./out"},
                {"desc": "指定语音模型", "cmd": "ppc10 resume ./txt ./out --voice zh-CN-XiaoxiaoNeural"},
                {"desc": "指定检查点路径", "cmd": "ppc10 resume ./txt ./out --checkpoint ./ckpt.json"},
            ],
            "options": [
                {"name": "--voice", "desc": "语音模型(默认使用配置文件)"},
                {"name": "--checkpoint", "desc": "检查点文件路径"},
            ],
            "see_also": ["convert"],
        },
        "split": {
            "category": "转换",
            "desc": "章节分割。按预设或自定义规则把整本小说文本切分为若干章节，输出到 output_dir。",
            "usage": "ppc10 split <input_file> [options]",
            "examples": [
                {"desc": "默认分割小说", "cmd": "ppc10 split novel.txt"},
                {"desc": "指定输出目录与预设", "cmd": "ppc10 split novel.txt -o ./chapters -p chinese_novel"},
                {"desc": "启用卷章体层级分割", "cmd": "ppc10 split novel.txt -H --add-title-separator"},
                {"desc": "使用自定义规则", "cmd": "ppc10 split novel.txt -r rules.json"},
            ],
            "options": [
                {"name": "--output, -o", "desc": "输出目录"},
                {"name": "--preset, -p", "desc": "章节预设"},
                {"name": "--custom-rules, -r", "desc": "自定义规则 JSON 字符串或文件路径"},
                {"name": "--add-title-separator/--no-add-title-separator", "desc": "是否在章节名后添加等于号分隔符"},
                {"name": "--hierarchical, -H", "desc": "启用卷章体层级分割"},
            ],
            "see_also": ["convert", "batch"],
        },
        "batch": {
            "category": "转换",
            "desc": "批量归档。按批次或按卷把源目录中的 .txt 文件分批归档，支持预览模式。",
            "usage": "ppc10 batch <source_dir> [options]",
            "examples": [
                {"desc": "按批次归档", "cmd": "ppc10 batch ./txt -b 50"},
                {"desc": "预览并按卷归档", "cmd": "ppc10 batch ./txt --dry-run --group-by-volume"},
                {"desc": "按卷归档指定批次大小", "cmd": "ppc10 batch ./txt -g -b 100"},
            ],
            "options": [
                {"name": "--batch-size, -b", "desc": "每批次文件数"},
                {"name": "--dry-run, -d", "desc": "预览模式"},
                {"name": "--group-by-volume, -g", "desc": "按卷归档"},
            ],
            "see_also": ["convert", "split"],
        },
        "config": {
            "category": "配置",
            "desc": "配置管理。管理 PPC10 配置文件，支持 show/get/set/reset/export/import/init/path/wizard 等操作。",
            "usage": "ppc10 config <action> [options]",
            "examples": [
                {"desc": "显示完整配置", "cmd": "ppc10 config show"},
                {"desc": "获取配置项", "cmd": "ppc10 config get --key tts.voice"},
                {"desc": "设置配置项", "cmd": "ppc10 config set --key tts.voice --value zh-CN-XiaoxiaoNeural"},
                {"desc": "交互式配置向导", "cmd": "ppc10 config wizard"},
            ],
            "options": [
                {"name": "--key, -k", "desc": "配置键"},
                {"name": "--value, -v", "desc": "配置值"},
                {"name": "--preset, -p", "desc": "预设"},
                {"name": "--temp", "desc": "临时设置"},
                {"name": "--export, -e", "desc": "导出路径"},
                {"name": "--import, -i", "desc": "导入路径"},
                {"name": "--full, -f", "desc": "完整配置模式(wizard 专用)"},
            ],
            "see_also": ["analyze"],
        },
        "voices": {
            "category": "工具",
            "desc": "列出可用语音。列出 Edge TTS 服务提供的所有可用语音，中文语音优先。",
            "usage": "ppc10 voices [options]",
            "examples": [
                {"desc": "列出语音", "cmd": "ppc10 voices"},
                {"desc": "JSON 输出", "cmd": "ppc10 voices --json"},
            ],
            "options": [
                {"name": "--json", "desc": "以单行 JSON 数组输出"},
            ],
            "see_also": ["convert"],
        },
        "analyze": {
            "category": "工具",
            "desc": "系统分析与健康检查。默认运行轻量级系统健康检查，使用 --deep 启用深度分析。",
            "usage": "ppc10 analyze [options]",
            "examples": [
                {"desc": "健康检查", "cmd": "ppc10 analyze"},
                {"desc": "健康检查并尝试修复", "cmd": "ppc10 analyze --fix"},
                {"desc": "导出健康检查结果", "cmd": "ppc10 analyze --export report.json"},
                {"desc": "深度分析", "cmd": "ppc10 analyze --deep"},
                {"desc": "持续监控", "cmd": "ppc10 analyze --deep --watch -i 30"},
            ],
            "options": [
                {"name": "--deep", "desc": "启用深度分析"},
                {"name": "--performance, -p", "desc": "性能分析"},
                {"name": "--config, -c", "desc": "配置冲突检测"},
                {"name": "--errors, -e", "desc": "错误模式识别"},
                {"name": "--dependency", "desc": "依赖分析"},
                {"name": "--network", "desc": "网络分析"},
                {"name": "--resource", "desc": "资源分析"},
                {"name": "--quality", "desc": "代码质量分析"},
                {"name": "--fix, -f", "desc": "自动修复(交互式确认)"},
                {"name": "--export, -x", "desc": "导出分析报告为 JSON 文件"},
                {"name": "--diff", "desc": "与最近历史对比"},
                {"name": "--watch, -w", "desc": "持续监控模式"},
                {"name": "--interval, -i", "desc": "监控间隔(秒)"},
                {"name": "--export-html", "desc": "导出 HTML 报告"},
                {"name": "--full", "desc": "完整检查(健康检查模式专用)"},
            ],
            "see_also": ["config"],
        },
        "dist node": {
            "category": "高级",
            "desc": "启动 TTS 节点服务。启动一个 TTS worker 节点，接收主控的 convert 任务并执行。",
            "usage": "ppc10 dist node [options]",
            "examples": [
                {"desc": "默认启动节点", "cmd": "ppc10 dist node"},
                {"desc": "指定监听地址与端口", "cmd": "ppc10 dist node --host 0.0.0.0 --port 8000 -c 4"},
            ],
            "options": [
                {"name": "--host, -h", "desc": "节点监听地址"},
                {"name": "--port, -p", "desc": "节点监听端口"},
                {"name": "--concurrency, -c", "desc": "节点最大并发数"},
                {"name": "--config, -C", "desc": "配置文件路径"},
            ],
            "see_also": ["dist master", "dist status"],
        },
        "dist master": {
            "category": "高级",
            "desc": "启动 PPC10 分布式主控服务。主控把 convert 任务按策略分发给已注册的 worker 节点。",
            "usage": "ppc10 dist master [options]",
            "examples": [
                {"desc": "默认启动主控", "cmd": "ppc10 dist master"},
                {
                    "desc": "注册 worker 节点",
                    "cmd": "ppc10 dist master --add-worker 10.0.0.1:8000 --add-worker 10.0.0.2:8000",
                },
                {"desc": "本地兜底", "cmd": "ppc10 dist master --local-fallback"},
            ],
            "options": [
                {"name": "--host, -h", "desc": "主控监听地址"},
                {"name": "--port, -p", "desc": "主控监听端口"},
                {"name": "--concurrency, -c", "desc": "主控最大并发数(仅本地兜底使用)"},
                {"name": "--config, -C", "desc": "配置文件路径"},
                {"name": "--add-worker", "desc": "注册 worker 地址，可多次传入"},
                {"name": "--local-fallback", "desc": "在没有 worker 时本地兜底执行 TTS"},
            ],
            "see_also": ["dist node", "dist status", "dist convert"],
        },
        "dist status": {
            "category": "高级",
            "desc": "查看分布式系统状态。默认以单行 JSON 输出，--human 切换为表格输出。",
            "usage": "ppc10 dist status [options]",
            "examples": [
                {"desc": "默认 JSON 输出", "cmd": "ppc10 dist status"},
                {"desc": "表格输出", "cmd": "ppc10 dist status --human"},
                {"desc": "导出状态", "cmd": "ppc10 dist status --export status.json"},
            ],
            "options": [
                {"name": "--config, -C", "desc": "配置文件路径"},
                {"name": "--export, -e", "desc": "导出状态为 JSON"},
                {"name": "--human", "desc": "人类可读表格输出(默认 JSON)"},
                {"name": "--json", "desc": "结构化 JSON 输出(默认即 JSON)"},
                {"name": "--quiet, -q", "desc": "静默模式"},
                {"name": "--verbose, -v", "desc": "详细输出"},
            ],
            "see_also": ["dist node", "dist master"],
        },
        "dist add-node": {
            "category": "高级",
            "desc": "添加分布式节点。把指定 host:port 注册为分布式集群的 worker 节点。",
            "usage": "ppc10 dist add-node <host> <port> [options]",
            "examples": [
                {"desc": "添加节点", "cmd": "ppc10 dist add-node 10.0.0.1 8000"},
                {"desc": "指定并发且不保存", "cmd": "ppc10 dist add-node 10.0.0.1 8000 -c 8 --no-save"},
            ],
            "options": [
                {"name": "--concurrency, -c", "desc": "节点最大并发数"},
                {"name": "--config, -C", "desc": "配置文件路径"},
                {"name": "--save/--no-save", "desc": "是否保存到配置文件"},
            ],
            "see_also": ["dist master", "dist status"],
        },
        "dist convert": {
            "category": "高级",
            "desc": "把 convert 任务提交到远端主控节点。等价于 ppc10 convert 的参数，但实际执行发生在主控端。",
            "usage": "ppc10 dist convert <input_dir> <output_dir> [options]",
            "examples": [
                {"desc": "提交到本地主控", "cmd": "ppc10 dist convert ./txt ./out"},
                {"desc": "指定远端主控", "cmd": "ppc10 dist convert ./txt ./out --master http://10.0.0.1:9000"},
                {"desc": "指定语音与并发", "cmd": "ppc10 dist convert ./txt ./out -V zh-CN-YunxiNeural -c 16"},
            ],
            "options": [
                {"name": "--master, -m", "desc": "主控端点"},
                {"name": "--config, -C", "desc": "配置文件路径"},
                {"name": "--voice, -V", "desc": "语音模型"},
                {"name": "--rate", "desc": "音频播放速度"},
                {"name": "--concurrency, -c", "desc": "并发数"},
                {"name": "--local-fallback/--no-local-fallback", "desc": "无可用 worker 时本地兜底"},
                {"name": "--timeout", "desc": "HTTP 请求超时(秒)"},
            ],
            "see_also": ["convert", "dist master"],
        },
        "docs list": {
            "category": "其他",
            "desc": "列出 docs/ 与 .trae/specs/ 下的所有 markdown 文档。",
            "usage": "ppc10 docs list [options]",
            "examples": [
                {"desc": "列出文档", "cmd": "ppc10 docs list"},
                {"desc": "JSON 输出", "cmd": "ppc10 docs list --json"},
            ],
            "options": [
                {"name": "--json", "desc": "以单行 JSON 数组输出"},
            ],
            "see_also": ["docs show", "docs spec"],
        },
        "docs show": {
            "category": "其他",
            "desc": "模糊匹配并渲染指定 markdown 文档。",
            "usage": "ppc10 docs show <name>",
            "examples": [
                {"desc": "查看文档", "cmd": "ppc10 docs show exit-codes"},
            ],
            "options": [],
            "see_also": ["docs list", "docs new"],
        },
        "docs new": {
            "category": "其他",
            "desc": "在 docs/dev/ 下创建带 frontmatter 模板的 markdown 文件。",
            "usage": "ppc10 docs new <name>",
            "examples": [
                {"desc": "创建文档", "cmd": "ppc10 docs new my-new-doc"},
            ],
            "options": [],
            "see_also": ["docs show", "docs validate"],
        },
        "docs validate": {
            "category": "其他",
            "desc": "扫描所有 markdown 文档，报告坏链接 / 越界锚点。",
            "usage": "ppc10 docs validate",
            "examples": [
                {"desc": "验证文档链接", "cmd": "ppc10 docs validate"},
            ],
            "options": [],
            "see_also": ["docs show", "docs new"],
        },
        "docs spec": {
            "category": "其他",
            "desc": "显示 .trae/specs/<name>/ 的任务与 checklist 完成度。无 name 参数时列出所有 spec。",
            "usage": "ppc10 docs spec [name]",
            "examples": [
                {"desc": "列出所有 spec", "cmd": "ppc10 docs spec"},
                {"desc": "查看指定 spec", "cmd": "ppc10 docs spec mvp-cleanup"},
            ],
            "options": [],
            "see_also": ["docs list", "docs validate"],
        },
    }


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
):
    """进入 PPC10 交互式帮助浏览器

    列出所有命令，支持上下导航、搜索、查看详情。
    按 q 或 Ctrl+C 退出。

    Examples:
        ppc10 help
        ppc10 --no-emoji help
    """
    output = get_output()
    commands = _build_help_commands()
    if output.mode in ("json", "quiet"):
        # 非交互模式：输出命令索引 JSON 后退出，避免挂起等待输入
        sys.stdout.write(json.dumps(commands, ensure_ascii=False))
        sys.stdout.write("\n")
        sys.stdout.flush()
        return
    output.interactive_help(commands)


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
        None, "--checkpoint", help="检查点文件路径(默认: 输出目录/.ppc10_checkpoint.json)"
    ),
    timeout_multiplier: float | None = typer.Option(
        None, "--timeout-multiplier", "-t", min=0.5, max=2.0, help="超时倍率(0.5-2.0,默认使用配置文件)"
    ),
    timeout_mode: str | None = typer.Option(
        None, "--timeout-mode", help="超时模式(fixed|auto|adaptive,默认使用配置文件)"
    ),
    timeout: int | None = typer.Option(None, "--timeout", help="固定超时时间(秒)，仅 timeout_mode=fixed 时生效"),
    rate: str | None = typer.Option(None, "--rate", help="音频播放速度(如 +10%, -10%, +0%,范围 -100% 到 +100%)"),
    recursive: bool = typer.Option(False, "--recursive", "-R", help="递归处理子目录,保持目录结构"),
    ramp_up: float | None = typer.Option(
        None, "--ramp-up", help="并发预热时间(秒),从1并发逐步增加到设定并发,规避风控(如 30 表示30秒内完成预热)"
    ),
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

    handle_convert(
        input,
        output,
        voice,
        concurrency,
        preset,
        resume,
        checkpoint,
        timeout_multiplier,
        rate,
        recursive,
        ramp_up,
        strict,
        one=one,
        timeout_mode=timeout_mode,
        timeout=timeout,
    )


@app.command("resume")
def resume(
    ctx: typer.Context,
    input: Path = typer.Argument(..., help="输入目录（含原始 .txt 文件）"),
    output: Path = typer.Argument(..., help="输出目录（含 .cache 子目录）"),
    voice: str | None = typer.Option(None, "--voice", help="语音模型(默认使用配置文件)"),
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint", help="检查点文件路径(默认: 输出目录/.ppc10_checkpoint.json)"
    ),
):
    """从 .cache 分段重建断点续传检查点

    适用于之前未启用 --resume 但中途中断的场景。
    扫描 output 下的 .cache，根据已存在的段文件和对应 input 文件生成 checkpoint，
    之后可运行 `ppc10 convert <input> <output> --resume` 继续完成转换。

    Examples:
        ppc10 resume ./txt ./out
        ppc10 resume ./txt ./out --voice zh-CN-XiaoxiaoNeural
        ppc10 resume ./txt ./out --checkpoint ./my_checkpoint.json
    """
    from .commands.resume import handle_resume

    handle_resume(input, output, voice=voice, checkpoint_path=checkpoint)


@app.command("split")
def split(
    ctx: typer.Context,
    input_file: Path = typer.Argument(..., help="输入文件"),
    output_dir: Path = typer.Option(Path("chapters"), "--output", "-o", help="输出目录"),
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
    key: str | None = typer.Option(None, "--key", "-k", help="配置键"),
    value: str | None = typer.Option(None, "--value", "-v", help="配置值"),
    preset: str | None = typer.Option(None, "--preset", "-p", help="预设"),
    temp: bool = typer.Option(False, "--temp", help="临时设置"),
    export_path: Path | None = typer.Option(None, "--export", "-e", help="导出路径"),
    import_path: Path | None = typer.Option(None, "--import", "-i", help="导入路径"),
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
    export: str | None = typer.Option(None, "--export", "-x", help="导出分析报告为 JSON 文件"),
    diff: bool = typer.Option(False, "--diff", help="与最近历史对比"),
    watch: bool = typer.Option(False, "--watch", "-w", help="持续监控模式"),
    interval: int = typer.Option(60, "--interval", "-i", help="监控间隔(秒)"),
    export_html: str | None = typer.Option(None, "--export-html", help="导出HTML报告"),
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
    config_path: Path | None = typer.Option(None, "--config", "-C", help="配置文件路径"),
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
    config_path: Path | None = typer.Option(None, "--config", "-C", help="配置文件路径"),
    add_worker: list[str] = typer.Option(
        [],
        "--add-worker",
        help="注册 worker 地址,格式为 host:port 或 http://host:port,可多次传入",
    ),
    local_fallback: bool = typer.Option(False, "--local-fallback", help="在没有 worker 时,本地兜底执行 TTS"),
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
    from src.cli.typer_app import get_output

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
    config_path: Path | None = typer.Option(None, "--config", "-C", help="配置文件路径"),
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
        "http://127.0.0.1:9000",
        "--master",
        "-m",
        help="主控端点(host:port 或 http://host:port)",
    ),
    config_path: Path | None = typer.Option(None, "--config", "-C", help="配置文件路径"),
    voice: str | None = typer.Option(None, "--voice", "-V", help="语音模型"),
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

            logging.getLogger(__name__).warning(f"Failed to register CLI for extension {ext.metadata.name}: {e}")
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
        wrapped = CLIError(ErrorCode.E_BUSINESS, str(exc), hint="使用 --verbose 查看堆栈")
        wrapped.__cause__ = None
        _cli_error_handler(wrapped)


if __name__ == "__main__":
    run()


# ---------------------------------------------------------------------------
# docs 子命令注册
# ---------------------------------------------------------------------------

try:
    from .commands.docs import docs_app as _docs_app  # noqa: E402
except Exception as e:  # pragma: no cover - 注册失败不应阻塞主流程
    _docs_app = None
    logger.warning("Failed to register docs commands: %s", e)

if _docs_app is not None:
    app.add_typer(_docs_app, name="docs")
