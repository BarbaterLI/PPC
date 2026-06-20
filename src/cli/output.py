"""Output formatting module - Clean, friendly terminal output using Rich."""

import io
import json
import logging
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ppc10 import __version__

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    if sys.stderr.encoding != "utf-8":
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

import contextlib

from rich.box import ROUNDED, SIMPLE
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.style import Style
from rich.table import Table

from src.cli.design import atoms, layouts
from src.cli.design.renderers import HumanRenderer, JsonRenderer, QuietRenderer
from src.cli.design.tokens import (
    COLORS,
    SPACING,
    get_icon,
    get_style,
)
from src.cli.design.tokens import (
    set_no_color as _tokens_set_no_color,
)
from src.cli.design.tokens import (
    set_no_emoji as _tokens_set_no_emoji,
)

console = Console(file=sys.stdout, legacy_windows=False)


# ---------------------------------------------------------------------------
# 主题与颜色（Spec: 统一颜色主题）
# ---------------------------------------------------------------------------

# 语义颜色 → Rich style。所有需要颜色的代码都应当走 THEME,
# 不再散落 hardcode "[bold green]xxx[/bold green]" 之类的写法。
# mvp-cleanup:仅保留 4 个语义色,未在白名单里的名称 c() 会抛 KeyError。
THEME: dict[str, str] = {
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "info": "cyan",
}


def c(name: str, text: str) -> str:
    """按 :data:`THEME` 包裹 ``text``。

    若 :data:`no_color` 全局标记为 True(由 ``--no-color`` 触发),
    则直接返回 ``text``(不附加 ANSI 转义)。``name`` 必须是
    :data:`THEME` 中已注册的语义键,否则抛出 :class:`KeyError`。
    """
    if name not in THEME:
        raise KeyError(f"Unknown color name: {name!r}. Valid: {list(THEME.keys())}")
    if globals().get("no_color", False):
        return text
    style = THEME[name]
    return f"[{style}]{text}[/{style}]"


# 全局 --no-color 标记(由 typer_app 在解析后写入)
no_color: bool = False


class BrandAssets:
    """Brand assets."""

    LOGO_ASCII = """
██████╗ ██████╗  ██████╗     ██╗ ██████╗
██╔══██╗██╔══██╗██╔════╝    ███║██╔═████╗
██████╔╝██████╔╝██║         ╚██║██║██╔██║
██╔═══╝ ██╔═══╝ ██║          ██║████╔╝██║
██║     ██║     ╚██████╗     ██║╚██████╔╝
╚═╝     ╚═╝      ╚═════╝     ╚═╝ ╚═════╝
                                         """
    VERSION = __version__
    TAGLINE = "Ultimate Text-to-Speech Tool"
    COPYRIGHT = "© 2026 BLY Team. All rights reserved."


class BrandColors:
    """Brand color palette."""

    PRIMARY = "#4A90D9"
    SECONDARY = "#2ECC71"
    ACCENT = "#F39C12"
    SUCCESS = "#27AE60"
    ERROR = "#E74C3C"
    WARNING = "#F1C40F"
    INFO = "#3498DB"
    TEXT_PRIMARY = "#2C3E50"
    TEXT_SECONDARY = "#7F8C8D"
    BACKGROUND = "#ECF0F1"


class Icons:
    """Icon definitions."""

    SUCCESS = "✓"
    ERROR = "✗"
    WARNING = "⚠"
    INFO = "ℹ"
    ROCKET = "🚀"
    GEAR = "⚙"
    MICROPHONE = "🎤"
    SOUND = "🔊"
    FILE = "📄"
    FOLDER = "📁"
    CHART = "📊"
    CLOCK = "⏱"
    STAR = "⭐"
    BOOK = "📖"
    LINK = "🔗"


class StatusIcons:
    """Status icons."""

    RUNNING = ("◐", "cyan")
    COMPLETED = ("+", "green")
    FAILED = ("-", "red")
    PENDING = ("o", "dim")
    WARNING = ("!", "yellow")
    SKIPPED = ("->", "dim")
    INFO = ("i", "blue")
    SUCCESS = ("+", "green")


class OutputStyle:
    """Output style definitions."""

    SUCCESS = Style(color="green", bold=True)
    ERROR = Style(color="red", bold=True)
    WARNING = Style(color="yellow", bold=True)
    INFO = Style(color="blue")
    DEBUG = Style(color="cyan")
    PROGRESS = Style(color="magenta")
    TITLE = Style(color="white", bold=True, blink=False)
    RETRY = Style(color="orange3")
    TASK_RUNNING = Style(color="cyan")
    TASK_COMPLETED = Style(color="green")
    TASK_FAILED = Style(color="red")


class ErrorSuggestions:
    """Error fix suggestions."""

    SUGGESTIONS = {
        "FileNotFoundError": ["检查文件路径是否正确", "确认文件是否存在", "检查文件权限"],
        "PermissionError": ["以管理员权限运行", "检查文件/目录权限", "关闭占用文件的程序"],
        "NetworkError": ["检查网络连接", "检查代理设置", "稍后重试"],
        "ConnectionError": ["检查网络连接", "检查服务器地址是否正确", "检查防火墙设置"],
        "TimeoutError": ["检查网络连接稳定性", "增加超时时间设置", "稍后重试"],
        "ValueError": ["检查输入参数是否正确", "查看命令帮助信息", "确认参数格式是否符合要求"],
        "TypeError": ["检查参数类型是否正确", "查看API文档确认参数要求"],
        "KeyError": ["检查配置文件是否完整", "确认必需的配置项是否存在"],
        "ImportError": ["检查依赖是否已安装", "运行 pip install -r requirements.txt", "检查Python环境是否正确"],
        "OSError": ["检查系统资源是否充足", "检查磁盘空间", "检查文件系统权限"],
        "default": ["查看详细日志获取更多信息", "使用 --verbose 参数获取详细输出"],
    }

    @classmethod
    def get_suggestions(cls, error_type: str) -> list[str]:
        """Get error fix suggestions."""
        return cls.SUGGESTIONS.get(error_type, cls.SUGGESTIONS["default"])

    @classmethod
    def add_suggestion(cls, error_type: str, suggestions: list[str]):
        """Add custom error suggestions."""
        if error_type in cls.SUGGESTIONS:
            cls.SUGGESTIONS[error_type].extend(suggestions)
        else:
            cls.SUGGESTIONS[error_type] = suggestions


@dataclass
class TaskStatus:
    """Task status."""

    name: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    retries: int = 0
    error: str | None = None


@dataclass
class RetryInfo:
    """Retry information."""

    attempt: int
    max_attempts: int
    delay: float
    error: str
    will_retry: bool


class OutputFormatter:
    """Output formatter with Rich console support."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.console = Console()
        self._progress: Progress | None = None
        self._live: Live | None = None
        self._task_statuses: dict = {}
        # 模式:human / json / quiet;由 --json/--quiet 决定
        self.mode: Literal["human", "json", "quiet"] = "human"
        self.quiet = False
        self.json_output = False
        self.no_color = False
        self.no_emoji = False
        self.timestamps = False

        # 渲染器
        self._human_renderer = HumanRenderer(self.console)
        self._json_renderer = JsonRenderer()
        self._quiet_renderer = QuietRenderer()

        # 同步全局 token 标志
        _tokens_set_no_color(self.no_color)
        _tokens_set_no_emoji(self.no_emoji)

    # ------------------------------------------------------------------
    # 模式 / 开关管理
    # ------------------------------------------------------------------

    def set_mode(
        self,
        verbose: bool | None = None,
        quiet: bool | None = None,
        json_output: bool | None = None,
        no_color: bool | None = None,
        no_emoji: bool | None = None,
        timestamps: bool | None = None,
    ) -> None:
        """一次性写入多个开关;按优先级 json > quiet > human 决定 mode。"""
        if verbose is not None:
            self.verbose = bool(verbose)
        if quiet is not None:
            self.quiet = bool(quiet)
        if json_output is not None:
            self.json_output = bool(json_output)
        if no_color is not None:
            self.set_no_color(bool(no_color))
        if no_emoji is not None:
            self.set_no_emoji(bool(no_emoji))
        if timestamps is not None:
            self.set_timestamps(bool(timestamps))

        if self.json_output:
            self.mode = "json"
        elif self.quiet:
            self.mode = "quiet"
        else:
            self.mode = "human"

        # 同步全局 token 状态,供 atoms / renderers 读取
        _tokens_set_no_color(self.no_color)
        _tokens_set_no_emoji(self.no_emoji)

    def set_verbose(self, verbose: bool):
        """Set verbose mode."""
        self.verbose = bool(verbose)

    def set_quiet(self, quiet: bool):
        """Set quiet mode."""
        self.quiet = bool(quiet)
        if quiet and self.mode == "human":
            self.mode = "quiet"

    def set_json(self, json_output: bool):
        """Set json mode."""
        self.json_output = bool(json_output)
        if json_output:
            self.mode = "json"
        elif self.mode == "json":
            self.mode = "human"

    def set_no_color(self, no_color: bool):
        """Set no-color mode (全局 + 本地)。"""
        self.no_color = bool(no_color)
        globals()["no_color"] = bool(no_color)
        _tokens_set_no_color(bool(no_color))
        with contextlib.suppress(Exception):
            self.console.no_color = bool(no_color)
        with contextlib.suppress(Exception):
            console.no_color = bool(no_color)

    def set_no_emoji(self, no_emoji: bool):
        """Set no-emoji mode."""
        self.no_emoji = bool(no_emoji)
        _tokens_set_no_emoji(bool(no_emoji))

    def set_timestamps(self, timestamps: bool):
        """Set timestamp prefix mode."""
        self.timestamps = bool(timestamps)

    # ------------------------------------------------------------------
    # 渲染入口
    # ------------------------------------------------------------------

    def _render(self, component: Any) -> None:
        """根据当前 mode 选择对应 renderer 输出 component。"""
        if self.mode == "quiet":
            self._quiet_renderer.render(component)
        elif self.mode == "json":
            self._json_renderer.render(component)
        else:
            self._human_renderer.render(component)

    def render_layout(self, layout) -> None:
        """渲染一个 Layout：依次输出其包含的所有 Atom。"""
        for component in layout.to_components():
            self._render(component)

    def _timestamp(self) -> str:
        """当前时间戳字符串。"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def print_table(
        self,
        headers: list[str],
        rows: list[list[Any]],
        title: str | None = None,
        json_data: Any = None,
    ) -> None:
        """统一表格输出。

        - ``json`` 模式:输出单行 JSON(``json_data`` 优先,否则由 headers+rows 装配)。
        - ``human`` 模式:渲染 Rich ``Table``。
        - ``quiet`` 模式:不输出。
        """
        if self.mode == "json":
            if json_data is None:
                json_data = [{h: (r[i] if i < len(r) else None) for i, h in enumerate(headers)} for r in rows]
            sys.stdout.write(json.dumps(json_data, ensure_ascii=False))
            sys.stdout.write("\n")
            sys.stdout.flush()
            return

        table = atoms.Table(headers=headers, rows=rows, title=title)
        self._render(table)

    def print_panel(
        self,
        text: str,
        title: str | None = None,
        style: str = "info",
        border_style: str | None = None,
    ) -> None:
        """统一面板输出。

        - ``json`` 模式:不渲染 Rich Panel,仅在文本非空时作为 ``{"message": text}`` 单行输出。
        - ``quiet`` 模式:不输出。
        - ``human`` 模式:渲染 Rich ``Panel``。
        """
        if self.mode == "json":
            payload = {"message": text}
            if title:
                payload["title"] = title
            sys.stdout.write(json.dumps(payload, ensure_ascii=False))
            sys.stdout.write("\n")
            sys.stdout.flush()
            return

        panel = atoms.Panel(title=title, content=text, style=border_style or style)
        self._render(panel)

    def print_markdown(self, text: str) -> None:
        """渲染 Markdown 内容（仅在 human 模式下输出）。"""
        if self.mode != "human":
            return
        from rich.markdown import Markdown

        self.console.print(Markdown(text))

    def print_version_card(self) -> None:
        """``--version`` 卡片。

        - ``json`` 模式:单行 JSON,含 version / commit / python / platform / edge_tts / rich / repository。
        - 其它:Rich Panel,内嵌 Table。
        """
        commit = os.environ.get("PPC10_COMMIT", "-")
        py_ver = platform.python_version()
        plat = platform.platform()

        edge_tts_ver = "-"
        try:
            import edge_tts  # type: ignore

            edge_tts_ver = getattr(edge_tts, "__version__", "-")
        except Exception:
            pass

        rich_ver = "-"
        try:
            import rich  # type: ignore

            rich_ver = getattr(rich, "__version__", "-")
        except Exception:
            pass

        repository = os.environ.get("PPC10_REPOSITORY", "https://github.com/BarbaterLI/PPC")

        info = {
            "version": __version__,
            "commit": commit,
            "python": py_ver,
            "platform": plat,
            "edge_tts": edge_tts_ver,
            "rich": rich_ver,
            "repository": repository,
        }

        if self.mode == "json":
            sys.stdout.write(json.dumps(info, ensure_ascii=False))
            sys.stdout.write("\n")
            sys.stdout.flush()
            return

        if self.mode == "quiet":
            return

        table = atoms.Table(
            headers=["项目", "值"],
            rows=[
                ["版本", __version__],
                ["Commit", commit],
                ["Python", py_ver],
                ["平台", plat],
                ["edge-tts", edge_tts_ver],
                ["rich", rich_ver],
                ["仓库", repository],
            ],
            title=f"PPC10 v{__version__}",
        )
        self.console.print(
            Panel(
                table.to_rich(),
                title=f"{get_icon('info')} 版本信息",
                border_style=get_style("info"),
                padding=(SPACING.sm, SPACING.md),
            )
        )

    def error(self, exc) -> None:
        """统一错误渲染入口。

        支持两种调用方式:
        1. ``error(CLIError(...))`` —— 输出错误代码、消息与 Hint。
        2. ``error("xxx")`` / ``error(Exception(...))`` —— 兼容旧用法,渲染单行错误。

        ``--verbose`` 时追加 stack;``--json`` 时输出单行 JSON。
        """
        # 延迟 import 避免循环
        from .errors import CLIError

        if isinstance(exc, CLIError):
            code = exc.code.value
            message = exc.message
            hint = exc.hint
            cause = exc.__cause__
        elif isinstance(exc, BaseException):
            code = "E_BUSINESS"
            message = str(exc)
            hint = None
            cause = exc
        else:
            # 字符串(或其它) —— 兼容旧 output.error("xxx") 调用
            code = "E_BUSINESS"
            message = str(exc)
            hint = None
            cause = None

        if self.mode == "json":
            payload = {"error": {"code": code, "message": message, "hint": hint}}
            sys.stdout.write(json.dumps(payload, ensure_ascii=False))
            sys.stdout.write("\n")
            sys.stdout.flush()
            return

        text = f"[ERROR] {code}  {message}"
        self._render(atoms.Message("error", text, timestamp=self._timestamp() if self.timestamps else None))
        if hint:
            self._render(atoms.Message("warning", f"Hint: {hint}"))
        if self.verbose and cause is not None:
            import traceback

            try:
                tb = "".join(traceback.format_exception(type(cause), cause, cause.__traceback__))
                self._render(atoms.Panel(title="Traceback", content=tb, style="muted"))
            except Exception:
                pass

    def info(self, message: str, **kwargs):
        """Info message."""
        if self.mode == "json":
            return  # JSON 模式下不输出 info
        self._render(atoms.Message("info", message, timestamp=self._timestamp() if self.timestamps else None))

    def success(self, message: str):
        """Success message."""
        if self.mode == "json":
            return
        self._render(atoms.Message("success", message, timestamp=self._timestamp() if self.timestamps else None))

    def error_text(self, message: str):
        """原始错误消息(走 THEME 主题),不渲染 traceback。

        业务错误应优先抛 :class:`CLIError` 并由 :meth:`error` 渲染。
        """
        if self.mode == "json":
            return
        self._render(atoms.Message("error", message, timestamp=self._timestamp() if self.timestamps else None))

    def warning(self, message: str):
        """Warning message."""
        if self.mode == "json":
            return
        self._render(atoms.Message("warning", message, timestamp=self._timestamp() if self.timestamps else None))

    def debug(self, message: str):
        """Debug message."""
        if self.verbose:
            self._render(atoms.Message("debug", message, timestamp=self._timestamp() if self.timestamps else None))

    def _log(self, level: str, message: str):
        """Log with timestamp。"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"{timestamp} | {level:7s} | {message}")

    def title(self, text: str):
        """Output title."""
        console.print(f"\n{text}", style=get_style("header"))
        console.print("=" * len(text))

    def panel(self, content: str, title: str | None = None, style: str = "blue"):
        """Output panel."""
        # 兼容旧调用 style="blue" 等,映射到 tokens 语义名
        style_map = {
            "blue": "info",
            "green": "success",
            "red": "error",
            "yellow": "warning",
            "cyan": "info",
            "magenta": "accent",
        }
        token_style = style_map.get(
            style,
            style
            if style in ("info", "success", "error", "warning", "accent", "primary", "secondary", "muted")
            else "info",
        )
        self._render(atoms.Panel(title=title, content=content, style=token_style))

    def error_panel(
        self,
        message: str,
        title: str = "错误",
        error_type: str | None = None,
        suggestion: str | None = None,
        details: str | None = None,
    ) -> None:
        """Styled error panel."""
        code = error_type or "错误"
        suggestions = []
        if suggestion:
            suggestions.append(suggestion)
        elif error_type:
            suggestions = ErrorSuggestions.get_suggestions(error_type)

        layout = layouts.ErrorLayout(
            code=code,
            message=message,
            hint=details,
            suggestions=suggestions,
        )
        self.render_layout(layout)

    def success_panel(self, message: str, title: str = "成功", details: dict[str, Any] | None = None) -> None:
        """Styled success panel."""
        content_parts = []
        content_parts.append(f"[bold {COLORS.success}]{get_icon('success')} {message}[/bold {COLORS.success}]")

        if details:
            content_parts.append("")
            for key, value in details.items():
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    value_str = ", ".join(f"{k}: {v}" for k, v in value.items())
                else:
                    value_str = str(value)
                content_parts.append(f"[dim]{key}:[/dim] [cyan]{value_str}[/cyan]")

        self._render(
            atoms.Panel(
                title=f"[bold {COLORS.success}]{title}[/bold {COLORS.success}]",
                content=content_parts,
                style="success",
            )
        )

    def warning_panel(
        self,
        message: str,
        title: str = "警告",
        suggestion: str | None = None,
        details: str | None = None,
    ) -> None:
        """Styled warning panel."""
        content_parts = []
        content_parts.append(f"[bold {COLORS.warning}]{get_icon('warning')} {message}[/bold {COLORS.warning}]")

        if suggestion:
            content_parts.append(f"\n[bold {COLORS.accent}]{get_icon('info')} 建议:[/bold {COLORS.accent}]")
            content_parts.append(f"  [green]+[/green] {suggestion}")

        if details:
            content_parts.append(f"\n[dim]{details}[/dim]")

        self._render(
            atoms.Panel(
                title=f"[bold {COLORS.warning}]{title}[/bold {COLORS.warning}]",
                content=content_parts,
                style="warning",
            )
        )

    def collapsible_traceback(self, exception: Exception, expanded: bool = False, max_lines: int = 20) -> None:
        """Collapsible traceback display."""
        self._render(atoms.Trace(exception, expanded=expanded, max_lines=max_lines))

        error_type = type(exception).__name__
        error_msg = str(exception)

        self.console.print(
            f"\n[bold {COLORS.accent}]{get_icon('info')} 错误位置:[/bold {COLORS.accent}] [yellow]{error_type}[/yellow]"
        )
        self.console.print(f"   [dim]{error_msg}[/dim]")

        suggestions = ErrorSuggestions.get_suggestions(error_type)
        if suggestions:
            self.console.print(f"\n[bold {COLORS.info}]{get_icon('info')} 可能的解决方案:[/bold {COLORS.info}]")
            for sug in suggestions[:3]:
                self.console.print(f"   [green]+[/green] {sug}")

    def table(self, title: str, columns: list, rows: list):
        """Output table."""
        table = Table(title=title)
        for col in columns:
            table.add_column(col["header"], **col.get("options", {}))
        for row in rows:
            table.add_row(*row)
        console.print(table)

    def progress_start(self, total: int, description: str = "Processing"):
        """Start progress bar."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        self._progress.start()
        return self._progress.add_task(description, total=total)

    def progress_update(self, task_id: int, advance: int = 1):
        """Update progress."""
        if self._progress:
            self._progress.advance(TaskID(task_id), advance)

    def progress_stop(self):
        """Stop progress bar."""
        if self._progress:
            self._progress.stop()
            self._progress = None

    def progress_update_description(self, task_id: int, description: str):
        """Update progress description."""
        if self._progress:
            self._progress.update(TaskID(task_id), description=description)

    def compact_progress(self, current: int, total: int, filename: str, suffix: str = ""):
        """Compact progress display."""
        bar = atoms.ProgressBar(current, total, width=20).to_plain()
        status = f"[{current}/{total}] {bar}"
        if suffix:
            status += f" {suffix}"
        console.print(f"{status} {filename}")

    def final_progress(self, current: int, total: int, duration: float):
        """Final progress display."""
        bar = atoms.ProgressBar(current, total, width=20).to_plain()
        console.print(f"\r[{current}/{total}] {bar} 完成: {current}/{total} 用时: {duration:.1f}s")

    def stat(self, key: str, value: Any):
        """Output single stat."""
        console.print(f"  {key}: {value}")

    def stats(self, stats: dict, title: str = "统计"):
        """Output stats."""
        self._render(atoms.StatGrid(items=stats, title=title))

    def config_show(self, config: dict):
        """Show config."""
        flat_config = {}
        labels = {}
        for section, values in config.items():
            if section.startswith("_"):
                continue
            if isinstance(values, dict):
                for key, value in values.items():
                    full_key = f"{section}.{key}"
                    flat_config[full_key] = value
                    labels[full_key] = f"{section} › {key}"
            else:
                flat_config[section] = values
                labels[section] = section

        layout = layouts.ConfigPreviewLayout(config=flat_config, labels=labels)
        self.render_layout(layout)

    def help_command(self, command: str, description: str, usage: str, examples: list):
        """Show command help."""
        content = f"[bold]描述[/bold]\n{description}\n\n"
        content += f"[bold]用法[/bold]\n{usage}\n\n"
        if examples:
            content += "[bold]示例[/bold]\n"
            for example in examples:
                content += f"  {example}\n"

        self._render(atoms.Panel(title=f"ppc10 {command}", content=content, style="success"))

    def check_result(self, checks: list):
        """Show check results."""
        headers = ["项目", "状态", "详情"]
        rows = []
        for check in checks:
            status = "+" if check["status"] else "-"
            rows.append([check["name"], status, check.get("detail", "")])
        self._render(atoms.Table(headers=headers, rows=rows, title="检查结果"))

    def retry_status(self, info: RetryInfo):
        """Show retry status."""
        if self.mode != "human":
            return
        if info.will_retry:
            msg = f"{get_icon('running')} 第 {info.attempt}/{info.max_attempts} 次尝试失败：{info.error}，{info.delay:.1f}s 后重试"
            console.print(msg, style=OutputStyle.RETRY)
        else:
            msg = f"{get_icon('error')} 已重试 {info.attempt} 次，最终失败：{info.error}"
            console.print(msg, style=OutputStyle.ERROR)

    def task_status(self, task: TaskStatus):
        """Show single task status."""
        status_icons = {
            "pending": get_icon("pending"),
            "running": get_icon("running"),
            "completed": get_icon("completed"),
            "failed": get_icon("error"),
        }
        status_styles = {
            "pending": OutputStyle.INFO,
            "running": OutputStyle.TASK_RUNNING,
            "completed": OutputStyle.TASK_COMPLETED,
            "failed": OutputStyle.TASK_FAILED,
        }

        icon = status_icons.get(task.status, "?")
        style = status_styles.get(task.status, OutputStyle.INFO)

        msg = f"{icon} {task.name}"
        if task.status == "running" and task.progress > 0:
            msg += f" ({task.progress:.0%})"
        if task.retries > 0:
            msg += f" [重试{task.retries}次]"
        if task.error:
            msg += f" - {task.error}"

        console.print(msg, style=style)

    def batch_summary(self, total: int, succeeded: int, failed: int, duration: float, retries: int = 0):
        """Show batch processing summary."""
        layout = layouts.CompletionReportLayout(
            total=total,
            completed=succeeded,
            failed=failed,
            elapsed=duration,
            error_type_counts=None,
            retries=retries,
        )
        self.render_layout(layout)

    def create_advanced_progress(self, description: str = "处理中", show_speed: bool = True) -> Progress:
        """Create advanced progress bar."""
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]

        if show_speed:
            columns.insert(4, TransferSpeedColumn())

        return Progress(*columns, console=console)

    def live_status_start(self) -> Live:
        """Start live status."""
        self._live = Live(console=console, refresh_per_second=4)
        self._live.start()
        return self._live

    def live_status_stop(self):
        """Stop live status."""
        if self._live:
            self._live.stop()
            self._live = None

    def update_task_status(self, task_id: str, status: TaskStatus):
        """Update task status for live display."""
        self._task_statuses[task_id] = status

        if self._live:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("任务")
            table.add_column("状态")
            table.add_column("进度")
            table.add_column("重试")

            for _tid, task in self._task_statuses.items():
                status_icons = {
                    "pending": f"[dim]{get_icon('pending')}[/dim]",
                    "running": f"[cyan]{get_icon('running')}[/cyan]",
                    "completed": f"[green]{get_icon('completed')}[/green]",
                    "failed": f"[red]{get_icon('error')}[/red]",
                }
                icon = status_icons.get(task.status, "?")

                progress_str = f"{task.progress:.0%}" if task.progress > 0 else "-"
                retry_str = str(task.retries) if task.retries > 0 else "-"

                table.add_row(task.name, f"{icon} {task.status}", progress_str, retry_str)

            self._live.update(table)

    def show_banner(self):
        """Show brand banner."""
        if self.mode != "human":
            return
        logo_lines = BrandAssets.LOGO_ASCII.strip().split("\n")
        colored_logo = "\n".join(f"[{COLORS.primary}]{line}[/{COLORS.primary}]" for line in logo_lines)
        self.console.print(colored_logo)
        self.console.print()
        version_text = f"[{COLORS.accent}]版本 {BrandAssets.VERSION}[/{COLORS.accent}]"
        tagline_text = f"[{COLORS.secondary}]{BrandAssets.TAGLINE}[/{COLORS.secondary}]"
        self.console.print(f"  {version_text}  |  {tagline_text}")
        self.console.print()
        copyright_text = f"[{COLORS.muted}]{BrandAssets.COPYRIGHT}[/{COLORS.muted}]"
        self.console.print(f"  {copyright_text}")
        self.console.print()

    def show_welcome(self):
        """Show welcome info."""
        commands = [
            (f"{get_icon('sound')} ppc10 convert", "转换文本为语音"),
            (f"{get_icon('folder')} ppc10 batch", "批量处理文件"),
            (f"{get_icon('gear')} ppc10 config", "配置设置"),
            (f"{get_icon('info')} ppc10 --help", "查看帮助信息"),
        ]
        tips = [
            "使用 ppc10 convert -i input.txt 快速转换单个文件",
            "使用 ppc10 config wizard 进行交互式配置",
            "使用 ppc10 --verbose 获取详细输出信息",
        ]
        layout = layouts.WelcomeLayout(
            version=BrandAssets.VERSION,
            tagline=BrandAssets.TAGLINE,
            commands=commands,
            tips=tips,
        )
        self.render_layout(layout)

    def config_wizard(self, full: bool = False) -> dict[str, Any] | None:
        """交互式配置向导，使用 StepLayout / ConfigPreviewLayout 输出。"""
        if self.mode != "human":
            self.warning("配置向导需要交互式终端，当前模式不支持")
            return None

        config: dict[str, Any] = {}

        total_steps = 8 if full else 3
        current_step = 0

        def _step_header(step_num: int, step_title: str, step_icon: str = ""):
            nonlocal current_step
            current_step = step_num
            layout = layouts.StepLayout(
                step=step_num,
                total=total_steps,
                title=step_title,
                icon=step_icon,
            )
            self.render_layout(layout)

        welcome_lines = [
            f"{get_icon('gear')} PPC10 配置向导",
            f"PPC10 v{BrandAssets.VERSION} | 冰璃岩文本转语音工具",
        ]
        if full:
            welcome_lines.append("完整配置模式：请按照提示完成所有配置。")
        else:
            welcome_lines.append("快速配置模式：仅核心设置。使用 --full 可配置所有项。")

        self.print_panel(
            "\n".join(welcome_lines),
            title="配置向导",
            style="primary",
        )

        _step_header(1, "TTS 核心配置", get_icon("rocket"))

        voice_options = [
            ("zh-CN-XiaoxiaoNeural", "晓晓 - 女声，自然流畅"),
            ("zh-CN-YunxiNeural", "云希 - 男声，年轻活力"),
            ("zh-CN-YunjianNeural", "云健 - 男声，成熟稳重"),
            ("zh-CN-XiaoyiNeural", "晓伊 - 女声，温柔甜美"),
            ("zh-CN-YunyangNeural", "云扬 - 男声，新闻播报"),
            ("zh-CN-XiaochenNeural", "晓辰 - 女声，亲切温暖"),
            ("zh-CN-liaoning-Normal", "辽宁 - 地方口音"),
            ("zh-CN-shaanxi-Normal", "陕西 - 地方口音"),
            ("zh-TW-HsiaoChenNeural", "晓臻 - 台湾女声"),
            ("zh-HK-HiuMaanNeural", "晓曼 - 粤语女声"),
        ]

        self.info("可用语音选项:")
        for i, (voice_id, voice_desc) in enumerate(voice_options, 1):
            self.info(f"  {i}. {voice_desc} ({voice_id})")

        voice_choice = Prompt.ask(
            "\n请选择语音",
            choices=[str(i) for i in range(1, len(voice_options) + 1)],
            default="2",
            console=self.console,
        )
        config["tts.voice"] = voice_options[int(voice_choice) - 1][0]

        concurrency = Prompt.ask(
            "并发数 (1-64)",
            default="8",
            console=self.console,
        )
        try:
            config["tts.concurrency"] = max(1, min(64, int(concurrency)))
        except ValueError:
            config["tts.concurrency"] = 8

        timeout_mode_options = [
            ("auto", "自动 - 根据文本长度动态调整"),
            ("fixed", "固定 - 使用固定超时值"),
            ("adaptive", "自适应 - 基于历史记录调整"),
        ]
        self.info("超时模式选项:")
        for i, (mode_id, mode_desc) in enumerate(timeout_mode_options, 1):
            self.info(f"  {i}. {mode_desc} ({mode_id})")

        timeout_choice = Prompt.ask(
            "\n请选择超时模式",
            choices=[str(i) for i in range(1, len(timeout_mode_options) + 1)],
            default="1",
            console=self.console,
        )
        config["tts.timeout_mode"] = timeout_mode_options[int(timeout_choice) - 1][0]

        timeout = Prompt.ask(
            "固定超时时间 (秒, 0=自动推导)",
            default="0",
            console=self.console,
        )
        try:
            config["tts.timeout"] = max(0, int(timeout))
        except ValueError:
            config["tts.timeout"] = 0

        if full:
            timeout_min = Prompt.ask(
                "最小超时时间 (秒)",
                default="45",
                console=self.console,
            )
            try:
                config["tts.timeout_min"] = max(10, min(450, int(timeout_min)))
            except ValueError:
                config["tts.timeout_min"] = 45

            timeout_max = Prompt.ask(
                "最大超时时间 (秒)",
                default="900",
                console=self.console,
            )
            try:
                config["tts.timeout_max"] = max(60, min(3600, int(timeout_max)))
            except ValueError:
                config["tts.timeout_max"] = 900

            _step_header(2, "文本分段配置", get_icon("sound"))

            enable_segmentation = Confirm.ask(
                "\n启用文本分段?",
                default=True,
                console=self.console,
            )
            config["tts.enable_segmentation"] = enable_segmentation

            max_seg_len = Prompt.ask(
                "最大分段长度 (字符数)",
                default="2500",
                console=self.console,
            )
            try:
                config["tts.max_segment_length"] = max(100, int(max_seg_len))
            except ValueError:
                config["tts.max_segment_length"] = 2500

            min_seg_len = Prompt.ask(
                "最小分段长度 (字符数)",
                default="100",
                console=self.console,
            )
            try:
                config["tts.min_segment_length"] = max(10, min(1000, int(min_seg_len)))
            except ValueError:
                config["tts.min_segment_length"] = 100

            silence_ms = Prompt.ask(
                "分段间静音时长 (毫秒)",
                default="100",
                console=self.console,
            )
            try:
                config["tts.segment_silence_ms"] = max(0, min(1000, int(silence_ms)))
            except ValueError:
                config["tts.segment_silence_ms"] = 100

        _step_header(2 if not full else 3, "可靠性配置", get_icon("running"))

        max_retries = Prompt.ask(
            "TTS 最大重试次数 (0-20)",
            default="3",
            console=self.console,
        )
        try:
            retries_value = max(0, min(20, int(max_retries)))
            config["reliability.tts_retry.max_retries"] = retries_value
            config["tts.retries"] = retries_value
        except ValueError:
            config["reliability.tts_retry.max_retries"] = 3
            config["tts.retries"] = 3

        base_delay = Prompt.ask(
            "重试基础延迟 (秒)",
            default="1.0",
            console=self.console,
        )
        try:
            config["reliability.tts_retry.base_delay"] = max(0.1, min(60.0, float(base_delay)))
        except ValueError:
            config["reliability.tts_retry.base_delay"] = 1.0

        if full:
            max_delay = Prompt.ask(
                "重试最大延迟 (秒)",
                default="60.0",
                console=self.console,
            )
            try:
                config["reliability.tts_retry.max_delay"] = max(1.0, min(300.0, float(max_delay)))
            except ValueError:
                config["reliability.tts_retry.max_delay"] = 60.0

            circuit_threshold = Prompt.ask(
                "熔断器失败阈值",
                default="5",
                console=self.console,
            )
            try:
                config["reliability.tts_circuit.failure_threshold"] = max(1, min(20, int(circuit_threshold)))
            except ValueError:
                config["reliability.tts_circuit.failure_threshold"] = 5

            _step_header(4, "性能配置", get_icon("gear"))

            rate_limit = Prompt.ask(
                "每秒请求数限制",
                default="100",
                console=self.console,
            )
            try:
                config["tts.rate_limit"] = max(1, int(rate_limit))
            except ValueError:
                config["tts.rate_limit"] = 100

            ema_alpha = Prompt.ask(
                "EMA 平滑因子 (0.0-1.0)",
                default="0.3",
                console=self.console,
            )
            try:
                config["tts.ema_alpha"] = max(0.0, min(1.0, float(ema_alpha)))
            except ValueError:
                config["tts.ema_alpha"] = 0.3

            buffer_size = Prompt.ask(
                "缓冲区大小",
                default="32",
                console=self.console,
            )
            try:
                config["tts.buffer_size"] = max(1, int(buffer_size))
            except ValueError:
                config["tts.buffer_size"] = 32

            _step_header(5, "文本正则化配置", get_icon("info"))

            enable_text_norm = Confirm.ask(
                "\n启用文本正则化?",
                default=True,
                console=self.console,
            )
            config["tts.text_normalization.enable_text_normalization"] = enable_text_norm

            if enable_text_norm:
                enable_whitespace = Confirm.ask(
                    "启用空白字符规范化?",
                    default=True,
                    console=self.console,
                )
                config["tts.text_normalization.enable_whitespace_normalization"] = enable_whitespace

                enable_linebreak = Confirm.ask(
                    "启用换行符规范化?",
                    default=True,
                    console=self.console,
                )
                config["tts.text_normalization.enable_linebreak_normalization"] = enable_linebreak

                enable_punct = Confirm.ask(
                    "启用标点符号规范化?",
                    default=True,
                    console=self.console,
                )
                config["tts.text_normalization.enable_punctuation_normalization"] = enable_punct

                enable_trim = Confirm.ask(
                    "启用行首尾空白去除?",
                    default=True,
                    console=self.console,
                )
                config["tts.text_normalization.enable_trim_whitespace"] = enable_trim

            _step_header(6, "章节分割配置", get_icon("folder"))

            split_presets = [
                ("chinese_novel", "中文小说"),
                ("english_novel", "英文小说"),
                ("default", "默认"),
            ]
            self.info("章节预设选项:")
            for i, (preset_id, preset_desc) in enumerate(split_presets, 1):
                self.info(f"  {i}. {preset_desc} ({preset_id})")

            split_choice = Prompt.ask(
                "\n请选择章节预设",
                choices=[str(i) for i in range(1, len(split_presets) + 1)],
                default="1",
                console=self.console,
            )
            config["split.preset"] = split_presets[int(split_choice) - 1][0]

            min_chap_len = Prompt.ask(
                "最小章节长度 (字符数)",
                default="100",
                console=self.console,
            )
            try:
                config["split.min_chapter_length"] = max(10, int(min_chap_len))
            except ValueError:
                config["split.min_chapter_length"] = 100

            add_sep = Confirm.ask(
                "在章节名后添加等于号分隔符?",
                default=True,
                console=self.console,
            )
            config["split.add_title_separator"] = add_sep

        _step_header(3 if not full else 7, "界面配置", get_icon("info"))

        verbose = Confirm.ask(
            "\n启用详细输出模式?",
            default=False,
            console=self.console,
        )
        config["ui.verbose"] = verbose

        if full:
            show_progress = Confirm.ask(
                "显示进度条?",
                default=True,
                console=self.console,
            )
            config["ui.show_progress"] = show_progress

            show_timestamps = Confirm.ask(
                "显示时间戳?",
                default=False,
                console=self.console,
            )
            config["ui.show_timestamps"] = show_timestamps

        if full:
            _step_header(8, "功能开关", get_icon("gear"))

            smart_detect = Confirm.ask(
                "\n启用智能检测?",
                default=True,
                console=self.console,
            )
            config["features.smart_detection"] = smart_detect

            merge_chapters = Confirm.ask(
                "启用合并短章节?",
                default=True,
                console=self.console,
            )
            config["features.merge_short_chapters"] = merge_chapters

            auto_retry = Confirm.ask(
                "启用自动重试?",
                default=True,
                console=self.console,
            )
            config["features.auto_retry"] = auto_retry

        self.print_panel("配置预览", title="配置预览", style="primary")

        config_labels = {
            "tts.voice": "语音模型",
            "tts.concurrency": "并发数",
            "tts.retries": "TTS重试次数",
            "tts.timeout_mode": "超时模式",
            "tts.timeout": "固定超时(秒)",
            "tts.timeout_min": "最小超时(秒)",
            "tts.timeout_max": "最大超时(秒)",
            "tts.enable_segmentation": "启用分段",
            "tts.max_segment_length": "最大分段长度",
            "tts.min_segment_length": "最小分段长度",
            "tts.segment_silence_ms": "分段静音(ms)",
            "reliability.tts_retry.max_retries": "可靠性重试次数",
            "reliability.tts_retry.base_delay": "重试基础延迟(秒)",
            "reliability.tts_retry.max_delay": "重试最大延迟(秒)",
            "reliability.tts_circuit.failure_threshold": "熔断器失败阈值",
            "tts.rate_limit": "每秒请求限制",
            "tts.ema_alpha": "EMA平滑因子",
            "tts.buffer_size": "缓冲区大小",
            "tts.text_normalization.enable_text_normalization": "启用文本正则化",
            "tts.text_normalization.enable_whitespace_normalization": "空白规范化",
            "tts.text_normalization.enable_linebreak_normalization": "换行规范化",
            "tts.text_normalization.enable_punctuation_normalization": "标点规范化",
            "tts.text_normalization.enable_trim_whitespace": "空白去除",
            "split.preset": "章节预设",
            "split.min_chapter_length": "最小章节长度",
            "split.add_title_separator": "章节名分隔符",
            "ui.verbose": "详细输出",
            "ui.show_progress": "进度条",
            "ui.show_timestamps": "时间戳",
            "features.smart_detection": "智能检测",
            "features.merge_short_chapters": "合并短章节",
            "features.auto_retry": "自动重试",
        }

        preview_layout = layouts.ConfigPreviewLayout(config=config, labels=config_labels)
        self.render_layout(preview_layout)

        if Confirm.ask(
            "\n确认保存配置?",
            default=True,
            console=self.console,
        ):
            self.success("配置已保存！")
            return config
        else:
            self.warning("配置已取消")
            return None

    @staticmethod
    def _read_nav_key(console: Console) -> str:
        """Read a single navigation key from stdin.

        Returns one of: 'UP', 'DOWN', 'ENTER', 'q', '/', 's', 'S', 'j', 'k',
        a digit, or any other single character. Falls back to line input when
        stdin is not a TTY (e.g. in tests or piped input).
        """
        import sys

        if not sys.stdin.isatty():
            try:
                return console.input().strip()
            except (KeyboardInterrupt, EOFError):
                raise
            except Exception:
                return ""

        if sys.platform == "win32":
            import msvcrt

            ch = msvcrt.getch()
            if ch == b"\r":
                return "ENTER"
            if ch == b"\x03":
                raise KeyboardInterrupt
            if ch in (b"\x00", b"\xe0"):
                ch2 = msvcrt.getch()
                if ch2 == b"H":
                    return "UP"
                if ch2 == b"P":
                    return "DOWN"
                return ""
            try:
                return ch.decode("utf-8", errors="ignore")
            except Exception:
                return ""
        else:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                if ch == "\x03":
                    raise KeyboardInterrupt
                if ch == "\x1b":
                    seq = sys.stdin.read(2)
                    if seq == "[A":
                        return "UP"
                    if seq == "[B":
                        return "DOWN"
                    return ""
                if ch in ("\r", "\n"):
                    return "ENTER"
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def interactive_help(self, commands: dict[str, dict]):
        """Interactive help browser."""
        self.title(f"{get_icon('book')} PPC10 交互式帮助浏览器")

        command_list = list(commands.keys())
        current_idx = 0
        search_filter = ""

        category_map: dict[str, list[str]] = {}
        for cmd_name, cmd_info in commands.items():
            cat = cmd_info.get("category", "其他")
            if cat not in category_map:
                category_map[cat] = []
            category_map[cat].append(cmd_name)

        category_colors = {
            "基础": COLORS.success,
            "转换": COLORS.info,
            "配置": COLORS.accent,
            "高级": "magenta",
            "工具": COLORS.warning,
            "其他": COLORS.muted,
        }

        while True:
            filtered_commands = [cmd for cmd in command_list if search_filter.lower() in cmd.lower()]

            if not filtered_commands:
                filtered_commands = command_list
                if search_filter:
                    console.print("\n[yellow]未找到匹配的命令，显示全部命令[/yellow]")

            if current_idx >= len(filtered_commands):
                current_idx = 0

            console.clear()
            console.print(f"\n[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")
            console.print(f"[bold white]  {get_icon('book')} PPC10 交互式帮助浏览器[/bold white]")
            console.print(f"[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")
            console.print(f"  [{COLORS.muted}]PPC10 v{BrandAssets.VERSION} | 输入 / 搜索命令[/{COLORS.muted}]")
            console.print(f"  [{COLORS.muted}]{'─' * 56}[/{COLORS.muted}]")

            if search_filter:
                console.print(f"\n[dim]搜索: [bold yellow]{search_filter}[/bold yellow][/dim]")

            for cat, cat_cmds in category_map.items():
                cat_filtered = [c for c in cat_cmds if c in filtered_commands]
                if not cat_filtered:
                    continue
                cat_color = category_colors.get(cat, COLORS.muted)
                console.print(f"\n[bold {cat_color}]  {cat}:[/bold {cat_color}]")
                for cmd in cat_filtered:
                    i = filtered_commands.index(cmd)
                    if i == current_idx:
                        console.print(f"    [bold green]▶ {cmd}[/bold green] - {commands[cmd].get('desc', '无描述')}")
                    else:
                        console.print(f"      [dim]{cmd}[/dim] - {commands[cmd].get('desc', '无描述')}")

            console.print(f"\n  [{COLORS.muted}]{'─' * 56}[/{COLORS.muted}]")
            console.print(
                "[dim]操作: [bold]↑/↓[/bold] 导航 | [bold]Enter[/bold] 查看详情 | [bold]/[/bold] 搜索 | [bold]q[/bold] 退出[/dim]"
            )

            selected_cmd = filtered_commands[current_idx] if filtered_commands else None
            if selected_cmd:
                cmd_info = commands[selected_cmd]
                preview = Panel(
                    f"[bold]描述:[/bold] {cmd_info.get('desc', '无描述')}\n"
                    f"[bold]用法:[/bold] {cmd_info.get('usage', '无用法说明')}",
                    title=f"[bold green]{selected_cmd}[/bold green]",
                    border_style="green",
                    box=SIMPLE,
                )
                console.print(preview)

            try:
                key = self._read_nav_key(console)

                if key == "q" or key == "Q":
                    console.print("\n[dim]退出帮助浏览器[/dim]")
                    break
                elif key == "/" or key == "s" or key == "S":
                    search_filter = Prompt.ask("\n[bold cyan]输入搜索关键词[/bold cyan]", default="")
                    current_idx = 0
                elif key == "UP" or key == "k":
                    current_idx = (current_idx - 1) % len(filtered_commands)
                elif key == "DOWN" or key == "j":
                    current_idx = (current_idx + 1) % len(filtered_commands)
                elif key == "ENTER" or key == "\r" or key == "\n":
                    if selected_cmd:
                        self._show_command_detail(selected_cmd, commands[selected_cmd])
                elif key.isdigit():
                    idx = int(key) - 1
                    if 0 <= idx < len(filtered_commands):
                        self._show_command_detail(filtered_commands[idx], commands[filtered_commands[idx]])

            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]退出帮助浏览器[/dim]")
                break
            except Exception:
                pass

    def _show_command_detail(self, command: str, info: dict):
        """Show command detail using CommandHelpLayout."""
        console.clear()
        console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        console.print(f"[bold white]  命令详情: [bold green]{command}[/bold green][/bold white]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

        examples = []
        for ex in info.get("examples", []):
            if isinstance(ex, dict):
                examples.append(
                    {
                        "command": ex.get("cmd", ""),
                        "description": ex.get("desc", ""),
                    }
                )
            else:
                examples.append({"command": str(ex), "description": ""})

        options = []
        for opt in info.get("options", []):
            if isinstance(opt, dict):
                options.append(
                    {
                        "name": opt.get("name", ""),
                        "description": opt.get("desc", ""),
                    }
                )

        layout = layouts.CommandHelpLayout(
            command=command,
            description=info.get("desc", ""),
            usage=info.get("usage", ""),
            examples=examples,
            options=options,
            see_also=info.get("see_also", []),
        )
        self.render_layout(layout)

        console.print("[bold cyan]───────────────────────────────────────────────────────────────[/bold cyan]")
        Prompt.ask("\n[dim]按 Enter 返回[/dim]")

    def command_examples(self, command: str, examples: list[dict]):
        """Show command examples."""
        console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        console.print(f"[bold white]  {get_icon('info')} {command} 命令示例[/bold white]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

        for i, example in enumerate(examples, 1):
            desc = example.get("desc", "无描述")
            cmd = example.get("cmd", "")
            output = example.get("output", None)

            console.print(f"[bold yellow]示例 {i}:[/bold yellow] {desc}")
            console.print(Panel(f"[bold green]$ {cmd}[/bold green]", border_style="green", box=SIMPLE, padding=(0, 1)))

            if output:
                console.print("[dim]预期输出:[/dim]")
                console.print(Panel(output, border_style="dim", box=SIMPLE, padding=(0, 1)))
            console.print()

        console.print("[bold cyan]───────────────────────────────────────────────────────────────[/bold cyan]")

    def show_shortcuts(self):
        """Show shortcut tips."""
        self.console.print()
        self.console.print(f"[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")
        self.console.print(f"[bold white]  {get_icon('gear')} PPC10 快捷键参考[/bold white]")
        self.console.print(f"[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")
        self.console.print(f"  [{COLORS.muted}]{'─' * 56}[/{COLORS.muted}]")
        self.console.print()

        global_shortcuts = [
            ("Ctrl + C", "中断当前操作"),
            ("Ctrl + D", "退出程序"),
            ("Tab", "自动补全命令"),
            ("↑ / ↓", "浏览历史命令"),
            ("Ctrl + L", "清屏"),
            ("Ctrl + R", "搜索历史命令"),
        ]

        command_shortcuts = [
            ("ppc10 --help", "显示帮助信息"),
            ("ppc10 --version", "显示版本号"),
            ("ppc10 -v", "详细输出模式"),
            ("ppc10 convert -h", "convert 命令帮助"),
            ("ppc10 config show", "显示当前配置"),
        ]

        interactive_shortcuts = [
            ("↑ / ↓ / j / k", "上下导航"),
            ("Enter", "选择/确认"),
            ("/ 或 s", "搜索过滤"),
            ("q", "退出/返回"),
            ("数字键", "快速选择"),
        ]

        table1 = Table(
            show_header=True,
            box=ROUNDED,
            border_style=COLORS.info,
            title=f"[bold {COLORS.info}]  全局快捷键  [/bold {COLORS.info}]",
        )
        table1.add_column("快捷键", style=f"bold {COLORS.accent}", width=20)
        table1.add_column("功能", style="white", width=40)
        for shortcut, desc in global_shortcuts:
            table1.add_row(shortcut, desc)

        table2 = Table(
            show_header=True,
            box=ROUNDED,
            border_style=COLORS.success,
            title=f"[bold {COLORS.success}]  命令快捷方式  [/bold {COLORS.success}]",
        )
        table2.add_column("命令", style=f"bold {COLORS.success}", width=20)
        table2.add_column("功能", style="white", width=40)
        for cmd, desc in command_shortcuts:
            table2.add_row(cmd, desc)

        table3 = Table(
            show_header=True,
            box=ROUNDED,
            border_style="magenta",
            title="[bold magenta]  交互模式快捷键  [/bold magenta]",
        )
        table3.add_column("快捷键", style="bold magenta", width=20)
        table3.add_column("功能", style="white", width=40)
        for shortcut, desc in interactive_shortcuts:
            table3.add_row(shortcut, desc)

        columns = Columns([table1, table2, table3], equal=True, expand=True)
        self.console.print(columns)

        self.console.print()
        self.console.print(f"  [{COLORS.muted}]{'─' * 56}[/{COLORS.muted}]")
        self.console.print(f"[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")

    def help_command_enhanced(
        self,
        command: str,
        description: str,
        usage: str,
        examples: list[dict],
        options: list[dict] | None = None,
        see_also: list[str] | None = None,
    ):
        """Enhanced command help."""
        console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        console.print(f"[bold white]  {get_icon('book')} ppc10 {command}[/bold white]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

        desc_panel = Panel(
            description, title="[bold yellow]描述[/bold yellow]", border_style="yellow", box=SIMPLE, padding=(0, 1)
        )
        console.print(desc_panel)

        usage_panel = Panel(
            f"[bold cyan]{usage}[/bold cyan]",
            title="[bold green]用法[/bold green]",
            border_style="green",
            box=SIMPLE,
            padding=(0, 1),
        )
        console.print(usage_panel)

        if options:
            console.print("\n[bold magenta]选项:[/bold magenta]")
            opt_table = Table(show_header=True, box=SIMPLE, border_style="dim")
            opt_table.add_column("选项", style="bold yellow", width=25)
            opt_table.add_column("说明", style="white", width=50)
            opt_table.add_column("默认值", style="dim", width=15)

            for opt in options:
                opt_name = opt.get("name", "")
                opt_desc = opt.get("desc", "")
                opt_default = opt.get("default", "")
                opt_table.add_row(opt_name, opt_desc, f"[dim]{opt_default}[/dim]")

            console.print(opt_table)

        if examples:
            console.print("\n[bold blue]示例:[/bold blue]")
            for i, ex in enumerate(examples, 1):
                if isinstance(ex, dict):
                    desc = ex.get("desc", "")
                    cmd = ex.get("cmd", "")
                    console.print(f"  [bold]{i}.[/bold] [dim]{desc}[/dim]")
                    console.print(
                        Panel(f"[bold green]$ {cmd}[/bold green]", border_style="green", box=SIMPLE, padding=(0, 1))
                    )
                else:
                    console.print(f"  [bold]{i}.[/bold] {ex}")

        if see_also:
            console.print("\n[bold cyan]相关命令:[/bold cyan]")
            see_also_str = "  ".join(f"[bold green]{cmd}[/bold green]" for cmd in see_also)
            console.print(f"  {see_also_str}")

        console.print("\n[bold cyan]───────────────────────────────────────────────────────────────[/bold cyan]")

    def help_index(self, categories: dict[str, list[str]]):
        """Show help index."""
        self.console.print()
        self.console.print(f"[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")
        self.console.print(f"[bold white]  {get_icon('book')} PPC10 命令索引[/bold white]")
        self.console.print(f"[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")
        self.console.print(f"  [{COLORS.muted}]PPC10 v{BrandAssets.VERSION} | 冰璃岩文本转语音工具[/{COLORS.muted}]")
        self.console.print(f"  [{COLORS.muted}]{'─' * 56}[/{COLORS.muted}]")
        self.console.print()

        category_groups: dict[str, list[tuple[str, list[str]]]] = {
            "基础命令": [],
            "高级命令": [],
            "扩展命令": [],
        }

        category_icons = {
            "转换命令": "R",
            "配置命令": "G",
            "工具命令": "T",
            "信息命令": get_icon("info"),
            "默认": get_icon("folder"),
        }

        basic_keywords = {"转换", "基础", "核心", "基本"}
        advanced_keywords = {"高级", "配置", "可靠性", "性能"}
        extended_keywords = {"扩展", "工具", "信息", "其他"}

        for category, commands_list in categories.items():
            matched = False
            for kw in basic_keywords:
                if kw in category:
                    category_groups["基础命令"].append((category, commands_list))
                    matched = True
                    break
            if not matched:
                for kw in advanced_keywords:
                    if kw in category:
                        category_groups["高级命令"].append((category, commands_list))
                        matched = True
                        break
            if not matched:
                for kw in extended_keywords:
                    if kw in category:
                        category_groups["扩展命令"].append((category, commands_list))
                        matched = True
                        break
            if not matched:
                category_groups["扩展命令"].append((category, commands_list))

        group_icons = {
            "基础命令": f"[{COLORS.success}]+[/{COLORS.success}]",
            "高级命令": f"[{COLORS.accent}]*[/{COLORS.accent}]",
            "扩展命令": f"[{COLORS.info}]i[/{COLORS.info}]",
        }

        for group_name, group_items in category_groups.items():
            if not group_items:
                continue
            group_icon = group_icons.get(group_name, get_icon("folder"))
            self.console.print(f"[bold {COLORS.primary}]{group_icon} {group_name}[/bold {COLORS.primary}]")
            self.console.print(f"  [{COLORS.muted}]{'─' * 40}[/{COLORS.muted}]")

            for category, commands_list in group_items:
                icon = category_icons.get(category, category_icons["默认"])
                self.console.print(f"  [bold]{icon} {category}:[/bold]")

                for cmd in commands_list:
                    self.console.print(f"    [green]•[/green] [bold cyan]{cmd}[/bold cyan]")
                self.console.print()

        self.console.print(f"  [{COLORS.muted}]{'─' * 56}[/{COLORS.muted}]")
        self.console.print(f"[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")
        self.console.print(
            f"[dim]PPC10 v{BrandAssets.VERSION} | 使用 [bold]ppc10 <command> --help[/bold] 查看命令详细帮助[/dim]"
        )
        self.console.print("[dim]使用 [bold]ppc10 help[/bold] 进入交互式帮助浏览器[/dim]")

    def enhanced_stats_panel(self, stats: dict[str, Any], title: str = "详细统计") -> Panel:
        """Create enhanced stats panel."""
        lines = []

        if "total" in stats:
            total = stats["total"]
            completed = stats.get("completed", 0)
            failed = stats.get("failed", 0)
            pending = stats.get("pending", total - completed - failed)

            self._create_mini_progress_bar(completed + failed, total)
            lines.append("[bold]任务统计[/bold]")
            lines.append(f"  总数：{total}")
            lines.append(f"  [{COLORS.success}]{get_icon('success')} 完成:[/{COLORS.success}] {completed}")
            lines.append(f"  [{COLORS.error}]{get_icon('error')} 失败:[/{COLORS.error}] {failed}")
            lines.append(f"  [{COLORS.muted}]{get_icon('pending')} 待处理:[/{COLORS.muted}] {pending}")
            lines.append("")

        if "success_rate" in stats:
            rate = stats["success_rate"]
            rate_color = COLORS.success if rate >= 90 else (COLORS.warning if rate >= 70 else COLORS.error)
            retry_rate = stats.get("retry_rate", 0)
            quarantined_rate = stats.get("quarantined_rate", 0)

            lines.append("[bold]质量指标[/bold]")
            lines.append(f"  成功率：[{rate_color}]{rate:.1f}%[/{rate_color}]")
            lines.append(f"  重试率：[{COLORS.warning}]{retry_rate:.1f}%[/{COLORS.warning}]")
            lines.append(f"  隔离率：[{COLORS.accent}]{quarantined_rate:.1f}%[/{COLORS.accent}]")
            lines.append("")

        if "current_speed" in stats:
            current_speed = stats["current_speed"]
            average_speed = stats.get("average_speed", 0)
            p95_speed = stats.get("p95_speed", 0)

            lines.append("[bold]性能指标[/bold]")
            lines.append(f"  当前速度：[{COLORS.info}]{current_speed:.2f}[/{COLORS.info}] 任务/秒")
            lines.append(f"  平均速度：[{COLORS.info}]{average_speed:.2f}[/{COLORS.info}] 任务/秒")
            if p95_speed > 0:
                lines.append(f"  P95 速度：[{COLORS.accent}]{p95_speed:.2f}[/{COLORS.accent}] 任务/秒")
            lines.append("")

        if "elapsed" in stats:
            elapsed = stats["elapsed"]
            eta = stats.get("eta", 0)
            avg_duration = stats.get("avg_task_duration", 0)

            lines.append("[bold]时间统计[/bold]")
            lines.append(f"  已用时间：{self._format_duration(elapsed)}")
            lines.append(f"  预计剩余：{self._format_duration(eta)}")
            lines.append(f"  平均任务耗时：{avg_duration:.2f}秒")
            lines.append("")

        if (
            stats.get("total_retries", 0) > 0
            or stats.get("quarantined_count", 0) > 0
            or stats.get("circuit_breaker_trips", 0) > 0
        ):
            lines.append("[bold]可靠性[/bold]")
            if stats.get("total_retries", 0) > 0:
                lines.append(f"  总重试次数：[{COLORS.warning}]{stats['total_retries']}[/{COLORS.warning}]")
            if stats.get("quarantined_count", 0) > 0:
                lines.append(f"  隔离任务数：[{COLORS.accent}]{stats['quarantined_count']}[/{COLORS.accent}]")
            if stats.get("circuit_breaker_trips", 0) > 0:
                lines.append(f"  熔断器触发：[{COLORS.error}]{stats['circuit_breaker_trips']}[/{COLORS.error}]")
            lines.append("")

        if stats.get("error_type_counts"):
            lines.append("[bold]错误分类[/bold]")
            for error_type, count in sorted(stats["error_type_counts"].items(), key=lambda x: x[1], reverse=True)[:5]:
                lines.append(f"  {error_type}: {count}")

        return Panel(
            "\n".join(lines),
            title=f"[bold {COLORS.primary}]{get_icon('chart')} {title}[/bold {COLORS.primary}]",
            border_style=COLORS.primary,
            box=ROUNDED,
            padding=(1, 2),
        )

    def completion_report(
        self, stats: dict[str, Any], executor_stats: dict[str, Any] | None = None, title: str = "转换完成报告"
    ) -> None:
        """Show completion report."""
        total = stats.get("total", 0)
        completed = stats.get("completed", 0)
        failed = stats.get("failed", 0)
        elapsed = stats.get("elapsed", 0)
        error_type_counts = stats.get("error_type_counts")

        layout = layouts.CompletionReportLayout(
            total=total,
            completed=completed,
            failed=failed,
            elapsed=elapsed,
            error_type_counts=error_type_counts,
            retries=stats.get("total_retries", 0),
        )
        if self.mode == "human":
            self.console.print(f"\n[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")
            self.console.print(f"[bold white]  {get_icon('star')} {title}[/bold white]")
            self.console.print(f"[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]\n")
        self.render_layout(layout)

        if executor_stats and self.mode == "human":
            self.console.print(f"[bold {COLORS.accent}]{get_icon('gear')} 执行器统计:[/bold {COLORS.accent}]\n")

            exec_table = Table(show_header=False, box=SIMPLE, border_style=COLORS.accent)
            exec_table.add_column("组件", style="bold", width=20)
            exec_table.add_column("指标", style="cyan", width=30)

            if "rate_limiter" in executor_stats:
                rl_stats = executor_stats["rate_limiter"]
                if "current_rate" in rl_stats:
                    exec_table.add_row("限流器", f"当前速率：{rl_stats['current_rate']} req/min")
                if "ema_success_rate" in rl_stats:
                    exec_table.add_row("", f"EMA 成功率：{rl_stats['ema_success_rate']:.1%}")

            if "worker_pool" in executor_stats:
                wp_stats = executor_stats["worker_pool"]
                if "active_workers" in wp_stats:
                    exec_table.add_row("工作池", f"活跃工作数：{wp_stats['active_workers']}")
                if "avg_utilization" in wp_stats:
                    exec_table.add_row("", f"平均利用率：{wp_stats['avg_utilization']:.1%}")

            if "quarantine" in executor_stats:
                q_stats = executor_stats["quarantine"]
                if "current_size" in q_stats:
                    exec_table.add_row("隔离队列", f"当前隔离：{q_stats['current_size']}")
                if "total_quarantined" in q_stats:
                    exec_table.add_row("", f"累计隔离：{q_stats['total_quarantined']}")

            if "circuit_breaker" in executor_stats:
                cb_stats = executor_stats["circuit_breaker"]
                if "current_state" in cb_stats:
                    state_color = (
                        COLORS.success
                        if cb_stats["current_state"] == "CLOSED"
                        else (COLORS.warning if cb_stats["current_state"] == "HALF_OPEN" else COLORS.error)
                    )
                    exec_table.add_row("熔断器", f"[{state_color}]状态：{cb_stats['current_state']}[/{state_color}]")
                if "total_calls" in cb_stats:
                    exec_table.add_row("", f"总调用：{cb_stats['total_calls']}, 失败：{cb_stats['failed_calls']}")

            self.console.print(exec_table)
            self.console.print()

        if self.mode == "human":
            self.console.print(f"[bold {COLORS.primary}]{'─' * 60}[/bold {COLORS.primary}]")
            self.console.print(f"[dim]报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")

    def stats_panel(self, stats: dict[str, Any], title: str = "实时统计") -> Panel:
        """Create real-time stats panel."""
        total = stats.get("total", 0)
        completed = stats.get("completed", 0)
        failed = stats.get("failed", 0)
        speed = stats.get("current_speed", stats.get("speed", 0.0))
        elapsed = stats.get("elapsed", 0.0)
        eta = stats.get("eta", 0.0)

        layout = layouts.TaskDashboardLayout(
            total=total,
            completed=completed,
            failed=failed,
            current_task=stats.get("current_task"),
            speed=speed,
            elapsed=elapsed,
            eta=eta,
        )

        return Panel(
            layout.to_rich(),
            title=f"[bold {COLORS.primary}]{get_icon('chart')} {title}[/bold {COLORS.primary}]",
            border_style=COLORS.primary,
            box=ROUNDED,
            padding=(1, 2),
        )

    def _create_mini_progress_bar(self, current: int, total: int, width: int = 20) -> str:
        """Create mini progress bar."""
        bar_atom = atoms.ProgressBar(current, total, width=width)
        return bar_atom.to_plain()

    def _format_duration(self, seconds: float) -> str:
        """Format duration."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def decorate_status(self, status: str, text: str | None = None) -> str:
        """Decorate status text."""
        status_map = {
            "running": StatusIcons.RUNNING,
            "completed": StatusIcons.COMPLETED,
            "failed": StatusIcons.FAILED,
            "pending": StatusIcons.PENDING,
            "warning": StatusIcons.WARNING,
            "skipped": StatusIcons.SKIPPED,
            "info": StatusIcons.INFO,
            "success": StatusIcons.SUCCESS,
        }

        icon, color = status_map.get(status.lower(), ("?", "white"))

        if text:
            return f"[{color}]{icon}[/{color}] {text}"
        return f"[{color}]{icon}[/{color}]"

    def check_result_enhanced(self, checks: list[dict], title: str = "检查结果", show_summary: bool = True) -> None:
        """Enhanced check results."""
        headers = ["项目", "状态", "详情"]
        rows = []
        passed = 0
        failed = 0

        for check in checks:
            name = check.get("name", "")
            status = check.get("status", False)
            detail = check.get("detail", "")
            icon = check.get("icon", "")

            if icon:
                name = f"{icon} {name}"

            if status:
                status_text = self.decorate_status("success", "通过")
                passed += 1
            else:
                status_text = self.decorate_status("failed", "失败")
                failed += 1

            rows.append([name, status_text, detail])

        self._render(atoms.Table(headers=headers, rows=rows, title=title))

        if show_summary:
            total = passed + failed
            if total > 0:
                pass_rate = (passed / total) * 100
                summary_color = "green" if pass_rate == 100 else ("yellow" if pass_rate >= 50 else "red")

                summary_content = (
                    f"[bold]总计:[/bold] {total}  "
                    f"[green]通过:[/green] {passed}  "
                    f"[red]失败:[/red] {failed}  "
                    f"[bold {summary_color}]通过率:[/bold {summary_color}] {pass_rate:.1f}%"
                )
                self._render(
                    atoms.Panel(
                        title="[bold]汇总[/bold]",
                        content=summary_content,
                        style=summary_color,
                    )
                )


class ParallelProgress:
    """Multi-task parallel progress display."""

    def __init__(self, console: Console, max_workers: int = 4):
        self.console = console
        self.max_workers = max_workers
        self._progress: Progress | None = None
        self._tasks: dict[str, int] = {}
        self._overall_task: int | None = None
        self._total_tasks: int = 0
        self._completed_tasks: int = 0
        self._failed_tasks: int = 0
        self._start_time: float | None = None

    def start(self, total_tasks: int, description: str = "处理中"):
        """Start parallel progress."""
        import time

        self._total_tasks = total_tasks
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._start_time = time.time()

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style=BrandColors.SUCCESS, finished_style=BrandColors.SUCCESS),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        self._progress.start()

        self._overall_task = self._progress.add_task(
            f"[{BrandColors.PRIMARY}]{description}[/{BrandColors.PRIMARY}]", total=total_tasks
        )

    def add_task(self, name: str, total: int = 100) -> None:
        """Add sub-task."""
        if self._progress is None:
            return

        task_id = self._progress.add_task(
            f"  [{BrandColors.TEXT_SECONDARY}]{name}[/{BrandColors.TEXT_SECONDARY}]", total=total
        )
        self._tasks[name] = task_id

    def update_task(self, name: str, advance: int = 1, description: str | None = None) -> None:
        """Update sub-task progress."""
        if self._progress is None or name not in self._tasks:
            return

        task_id = TaskID(self._tasks[name])
        if description:
            self._progress.update(task_id, description=description, advance=advance)
        else:
            self._progress.advance(task_id, advance)

    def complete_task(self, name: str) -> None:
        """Mark task as completed."""
        if self._progress is None or name not in self._tasks:
            return

        task_id = TaskID(self._tasks[name])
        self._progress.update(
            task_id,
            description=f"  [{BrandColors.SUCCESS}]+ {name}[/{BrandColors.SUCCESS}]",
            completed=self._progress.tasks[task_id].total,
        )

        self._completed_tasks += 1
        if self._overall_task is not None:
            self._progress.advance(TaskID(self._overall_task), 1)

    def fail_task(self, name: str, error: str) -> None:
        """Mark task as failed."""
        if self._progress is None or name not in self._tasks:
            return

        task_id = TaskID(self._tasks[name])
        self._progress.update(task_id, description=f"  [{BrandColors.ERROR}]- {name}: {error}[/{BrandColors.ERROR}]")

        self._failed_tasks += 1
        if self._overall_task is not None:
            self._progress.advance(TaskID(self._overall_task), 1)

    def stop(self) -> None:
        """Stop progress display."""
        if self._progress:
            self._progress.stop()
            self._progress = None

    def get_stats(self) -> dict[str, Any]:
        """Get current stats."""
        import time

        elapsed = time.time() - self._start_time if self._start_time else 0

        return {
            "total": self._total_tasks,
            "completed": self._completed_tasks,
            "failed": self._failed_tasks,
            "elapsed": elapsed,
            "success_rate": (self._completed_tasks / self._total_tasks * 100) if self._total_tasks > 0 else 0,
            "speed": self._completed_tasks / elapsed if elapsed > 0 else 0,
        }


def setup_logging(verbose: bool = False):
    """Setup logging."""
    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    return logging.getLogger("ppc10")


def config_wizard(console: Console | None = None, full: bool = False) -> dict[str, Any] | None:
    """Interactive configuration wizard (legacy wrapper).

    Delegates to :meth:OutputFormatter.config_wizard.
    """
    output = OutputFormatter()
    if console is not None:
        output.console = console
    return output.config_wizard(full=full)
