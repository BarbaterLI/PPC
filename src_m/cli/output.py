"""Output formatting module - Clean, friendly terminal output using Rich."""

import json
import sys
import os
import io
import logging
import platform
from typing import Optional, Any, Callable, Dict, List, Literal
from datetime import datetime
from dataclasses import dataclass

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ppc10 import __version__

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn,
    TransferSpeedColumn, ProgressColumn
)
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.style import Style
from rich.theme import Theme
from rich.logging import RichHandler
from rich.live import Live
from rich.layout import Layout
from rich.prompt import Prompt, Confirm
from rich.columns import Columns
from rich.box import ROUNDED, SIMPLE

console = Console(file=sys.stdout, legacy_windows=False)


# ---------------------------------------------------------------------------
# 主题与颜色（Spec: 统一颜色主题）
# ---------------------------------------------------------------------------

# 语义颜色 → Rich style。所有需要颜色的代码都应当走 THEME,
# 不再散落 hardcode "[bold green]xxx[/bold green]" 之类的写法。
# mvp-cleanup:仅保留 4 个语义色,未在白名单里的名称 c() 会抛 KeyError。
THEME: Dict[str, str] = {
    "success": "bold green",
    "warning": "bold yellow",
    "error":   "bold red",
    "info":    "cyan",
}


def c(name: str, text: str) -> str:
    """按 :data:`THEME` 包裹 ``text``。

    若 :data:`no_color` 全局标记为 True(由 ``--no-color`` 触发),
    则直接返回 ``text``(不附加 ANSI 转义)。``name`` 必须是
    :data:`THEME` 中已注册的语义键,否则抛出 :class:`KeyError`。
    """
    if name not in THEME:
        raise KeyError(
            f"Unknown color name: {name!r}. Valid: {list(THEME.keys())}"
        )
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
        "FileNotFoundError": [
            "检查文件路径是否正确",
            "确认文件是否存在",
            "检查文件权限"
        ],
        "PermissionError": [
            "以管理员权限运行",
            "检查文件/目录权限",
            "关闭占用文件的程序"
        ],
        "NetworkError": [
            "检查网络连接",
            "检查代理设置",
            "稍后重试"
        ],
        "ConnectionError": [
            "检查网络连接",
            "检查服务器地址是否正确",
            "检查防火墙设置"
        ],
        "TimeoutError": [
            "检查网络连接稳定性",
            "增加超时时间设置",
            "稍后重试"
        ],
        "ValueError": [
            "检查输入参数是否正确",
            "查看命令帮助信息",
            "确认参数格式是否符合要求"
        ],
        "TypeError": [
            "检查参数类型是否正确",
            "查看API文档确认参数要求"
        ],
        "KeyError": [
            "检查配置文件是否完整",
            "确认必需的配置项是否存在"
        ],
        "ImportError": [
            "检查依赖是否已安装",
            "运行 pip install -r requirements.txt",
            "检查Python环境是否正确"
        ],
        "OSError": [
            "检查系统资源是否充足",
            "检查磁盘空间",
            "检查文件系统权限"
        ],
        "default": [
            "查看详细日志获取更多信息",
            "使用 --verbose 参数获取详细输出"
        ]
    }

    @classmethod
    def get_suggestions(cls, error_type: str) -> List[str]:
        """Get error fix suggestions."""
        return cls.SUGGESTIONS.get(error_type, cls.SUGGESTIONS["default"])

    @classmethod
    def add_suggestion(cls, error_type: str, suggestions: List[str]):
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
    error: Optional[str] = None


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
        self._progress: Optional[Progress] = None
        self._live: Optional[Live] = None
        self._task_statuses: dict = {}
        # 模式:human / json / quiet;由 --json/--quiet 决定
        self.mode: Literal["human", "json", "quiet"] = "human"
        self.no_color: bool = False

    # ------------------------------------------------------------------
    # 模式 / 开关管理
    # ------------------------------------------------------------------

    def set_mode(
        self,
        verbose: Optional[bool] = None,
        quiet: Optional[bool] = None,
        json_output: Optional[bool] = None,
        no_color: Optional[bool] = None,
    ) -> None:
        """一次性写入多个开关;按优先级 json > quiet > human 决定 mode。"""
        if verbose is not None:
            self.verbose = bool(verbose)
        if quiet is not None:
            self.quiet = bool(quiet)
        else:
            self.quiet = getattr(self, "quiet", False)
        if json_output is not None:
            self.json_output = bool(json_output)
        else:
            self.json_output = getattr(self, "json_output", False)
        if no_color is not None:
            self.set_no_color(bool(no_color))

        if self.json_output:
            self.mode = "json"
        elif self.quiet:
            self.mode = "quiet"
        else:
            self.mode = "human"

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
        # 同步全局标记
        globals()["no_color"] = bool(no_color)
        try:
            self.console.no_color = bool(no_color)
        except Exception:
            pass
        try:
            console.no_color = bool(no_color)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 渲染入口
    # ------------------------------------------------------------------

    def print_table(
        self,
        headers: List[str],
        rows: List[List[Any]],
        title: Optional[str] = None,
        json_data: Any = None,
    ) -> None:
        """统一表格输出。

        - ``json`` 模式:输出单行 JSON(``json_data`` 优先,否则由 headers+rows 装配)。
        - ``human`` 模式:渲染 Rich ``Table``。
        - ``quiet`` 模式:不输出。
        """
        if self.mode == "quiet":
            return

        if self.mode == "json":
            if json_data is None:
                json_data = [
                    {h: (r[i] if i < len(r) else None) for i, h in enumerate(headers)}
                    for r in rows
                ]
            sys.stdout.write(json.dumps(json_data, ensure_ascii=False))
            sys.stdout.write("\n")
            sys.stdout.flush()
            return

        from rich.table import Table
        from rich.box import SIMPLE
        table = Table(
            title=title,
            show_header=True,
            header_style="bold",
            box=SIMPLE,
            border_style=BrandColors.PRIMARY,
        )
        for h in headers:
            table.add_column(str(h))
        for row in rows:
            table.add_row(*[("" if v is None else str(v)) for v in row])
        self.console.print(table)

    def print_panel(
        self,
        text: str,
        title: Optional[str] = None,
        style: str = "info",
        border_style: Optional[str] = None,
    ) -> None:
        """统一面板输出。

        - ``json`` 模式:不渲染 Rich Panel,仅在文本非空时作为 ``{"message": text}`` 单行输出。
        - ``quiet`` 模式:不输出。
        - ``human`` 模式:渲染 Rich ``Panel``。
        """
        if self.mode == "quiet":
            return

        if self.mode == "json":
            payload = {"message": text}
            if title:
                payload["title"] = title
            sys.stdout.write(json.dumps(payload, ensure_ascii=False))
            sys.stdout.write("\n")
            sys.stdout.flush()
            return

        from rich.panel import Panel
        panel = Panel(
            text,
            title=title,
            border_style=border_style or style,
            expand=False,
            padding=(0, 1),
        )
        self.console.print(panel)

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

        repository = os.environ.get("PPC10_REPOSITORY", "https://github.com/bly-team/ppc10")

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

        from rich.panel import Panel
        from rich.table import Table
        from rich.box import SIMPLE

        table = Table(show_header=False, box=SIMPLE, border_style=BrandColors.PRIMARY)
        table.add_column("Key", style="bold cyan", width=14)
        table.add_column("Value", style="white")

        for key in ("version", "commit", "python", "platform", "edge_tts", "rich", "repository"):
            table.add_row(key, str(info[key]))

        title = f"[bold green]PPC10 v{__version__}[/bold green]"
        self.console.print(Panel(table, title=title, border_style=BrandColors.PRIMARY, padding=(0, 1)))

    def error(self, exc) -> None:
        """统一错误渲染入口。

        支持两种调用方式:
        1. ``error(CLIError(...))`` —— 输出 ``[ERROR] <CODE>  <message>`` + ``Hint``。
        2. ``error("xxx")`` —— 兼容旧用法,按 E_BUSINESS 渲染单行红字错误。

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

        self.console.print(f"[bold red][ERROR] {code}  {message}[/bold red]")
        if hint:
            self.console.print(f"  [bold yellow]Hint:[/bold yellow]  {hint}")
        if self.verbose and cause is not None:
            import traceback
            try:
                tb = "".join(traceback.format_exception(type(cause), cause, cause.__traceback__))
                self.console.print(f"[dim]{tb}[/dim]")
            except Exception:
                pass

    def info(self, message: str, **kwargs):
        """Info message."""
        if self.mode == "quiet":
            return
        if self.mode == "json":
            return  # JSON 模式下不向 stderr 喷 info
        if self.verbose:
            self._log("INFO", message)
        else:
            console.print(message, style=OutputStyle.INFO)

    def success(self, message: str):
        """Success message."""
        if self.mode == "quiet":
            return
        if self.mode == "json":
            return
        console.print(f"+ {message}", style=OutputStyle.SUCCESS)

    def error_text(self, message: str):
        """原始错误消息(走 THEME 主题),不渲染 traceback。

        业务错误应优先抛 :class:`CLIError` 并由 :meth:`error` 渲染。
        """
        if self.mode == "json":
            return
        console.print(f"- {message}", style=OutputStyle.ERROR)

    def warning(self, message: str):
        """Warning message."""
        if self.mode == "quiet":
            return
        if self.mode == "json":
            return
        console.print(f"! {message}", style=OutputStyle.WARNING)

    def debug(self, message: str):
        """Debug message."""
        if self.verbose:
            self._log("DEBUG", message)

    def _log(self, level: str, message: str):
        """Log with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"{timestamp} | {level:7s} | {message}")

    def title(self, text: str):
        """Output title."""
        console.print(f"\n{text}", style=OutputStyle.TITLE)
        console.print("=" * len(text))

    def panel(self, content: str, title: str = None, style: str = "blue"):
        """Output panel."""
        panel = Panel(content, title=title, style=style, expand=False)
        console.print(panel)

    def error_panel(
        self,
        message: str,
        title: str = "错误",
        error_type: str = None,
        suggestion: str = None,
        details: str = None
    ) -> None:
        """Styled error panel."""
        content_parts = []
        content_parts.append(f"[bold {BrandColors.ERROR}]{Icons.ERROR} {message}[/bold {BrandColors.ERROR}]")

        if error_type:
            content_parts.append(f"\n[dim]错误类型:[/dim] [yellow]{error_type}[/yellow]")

        if details:
            content_parts.append(f"\n[dim]详细信息:[/dim]")
            content_parts.append(f"[dim]{details}[/dim]")

        suggestions = []
        if suggestion:
            suggestions.append(suggestion)
        elif error_type:
            suggestions = ErrorSuggestions.get_suggestions(error_type)

        if suggestions:
            content_parts.append(f"\n[bold {BrandColors.ACCENT}]i 修复建议:[/bold {BrandColors.ACCENT}]")
            for sug in suggestions:
                content_parts.append(f"  [green]+[/green] {sug}")

        panel = Panel(
            "\n".join(content_parts),
            title=f"[bold {BrandColors.ERROR}]{title}[/bold {BrandColors.ERROR}]",
            border_style=BrandColors.ERROR,
            box=ROUNDED,
            expand=False,
            padding=(1, 2)
        )
        self.console.print(panel)

    def success_panel(
        self,
        message: str,
        title: str = "成功",
        details: Dict[str, Any] = None
    ) -> None:
        """Styled success panel."""
        content_parts = []
        content_parts.append(f"[bold {BrandColors.SUCCESS}]{Icons.SUCCESS} {message}[/bold {BrandColors.SUCCESS}]")

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

        panel = Panel(
            "\n".join(content_parts),
            title=f"[bold {BrandColors.SUCCESS}]{title}[/bold {BrandColors.SUCCESS}]",
            border_style=BrandColors.SUCCESS,
            box=ROUNDED,
            expand=False,
            padding=(1, 2)
        )
        self.console.print(panel)

    def warning_panel(
        self,
        message: str,
        title: str = "警告",
        suggestion: str = None
    ) -> None:
        """Styled warning panel."""
        content_parts = []
        content_parts.append(f"[bold {BrandColors.WARNING}]{Icons.WARNING} {message}[/bold {BrandColors.WARNING}]")

        if suggestion:
            content_parts.append(f"\n[bold {BrandColors.ACCENT}]i 建议:[/bold {BrandColors.ACCENT}]")
            content_parts.append(f"  [green]+[/green] {suggestion}")

        panel = Panel(
            "\n".join(content_parts),
            title=f"[bold {BrandColors.WARNING}]{title}[/bold {BrandColors.WARNING}]",
            border_style=BrandColors.WARNING,
            box=ROUNDED,
            expand=False,
            padding=(1, 2)
        )
        self.console.print(panel)

    def collapsible_traceback(
        self,
        exception: Exception,
        expanded: bool = False,
        max_lines: int = 20
    ) -> None:
        """Collapsible traceback display."""
        import traceback

        tb_lines = traceback.format_exception(type(exception), exception, exception.__traceback__)
        tb_text = "".join(tb_lines)

        error_type = type(exception).__name__
        error_msg = str(exception)

        lines = tb_text.strip().split("\n")
        total_lines = len(lines)

        if expanded or total_lines <= max_lines:
            display_lines = lines
            show_indicator = False
        else:
            display_lines = lines[:max_lines]
            show_indicator = True

        header = f"[bold {BrandColors.ERROR}]i 堆栈追踪[/bold {BrandColors.ERROR}]"
        self.console.print(f"\n{header}")

        tb_content = []
        for line in display_lines:
            if line.strip().startswith("File"):
                tb_content.append(f"[dim]{line}[/dim]")
            elif "Error" in line or "Exception" in line:
                tb_content.append(f"[bold red]{line}[/bold red]")
            else:
                tb_content.append(line)

        if show_indicator:
            hidden_count = total_lines - max_lines
            tb_content.append("")
            tb_content.append(f"[dim]... 省略了 {hidden_count} 行 ...[/dim]")
            tb_content.append(f"[dim]使用 --verbose 参数查看完整堆栈追踪[/dim]")

        panel = Panel(
            "\n".join(tb_content),
            title=f"[bold]{error_type}[/bold]",
            border_style=BrandColors.ERROR,
            box=SIMPLE,
            expand=False,
            padding=(0, 1)
        )
        self.console.print(panel)

        self.console.print(f"\n[bold {BrandColors.ACCENT}]i 错误位置:[/bold {BrandColors.ACCENT}] [yellow]{error_type}[/yellow]")
        self.console.print(f"   [dim]{error_msg}[/dim]")

        suggestions = ErrorSuggestions.get_suggestions(error_type)
        if suggestions:
            self.console.print(f"\n[bold {BrandColors.INFO}]i 可能的解决方案:[/bold {BrandColors.INFO}]")
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
            console=console
        )
        self._progress.start()
        return self._progress.add_task(description, total=total)

    def progress_update(self, task_id: int, advance: int = 1):
        """Update progress."""
        if self._progress:
            self._progress.advance(task_id, advance)

    def progress_stop(self):
        """Stop progress bar."""
        if self._progress:
            self._progress.stop()
            self._progress = None

    def progress_update_description(self, task_id: int, description: str):
        """Update progress description."""
        if self._progress:
            self._progress.update(task_id, description=description)

    def compact_progress(self, current: int, total: int, filename: str, suffix: str = ""):
        """Compact progress display."""
        percent = current / total if total > 0 else 0
        bar_length = 20
        filled = int(bar_length * percent)
        bar = "█" * filled + "░" * (bar_length - filled)

        status = f"[{current}/{total}] {bar} {percent:.0%}"
        if suffix:
            status += f" {suffix}"

        console.print(f"{status} {filename}")

    def final_progress(self, current: int, total: int, duration: float):
        """Final progress display."""
        percent = current / total if total > 0 else 0
        bar_length = 20
        filled = int(bar_length * percent)
        bar = "█" * filled + "░" * (bar_length - filled)

        console.print(f"\r[{current}/{total}] {bar} {percent:.0%} 完成: {current}/{total} 用时: {duration:.1f}s")

    def stat(self, key: str, value: Any):
        """Output single stat."""
        console.print(f"  {key}: {value}")

    def stats(self, stats: dict, title: str = "统计"):
        """Output stats."""
        self.panel("\n".join(f"{k}: {v}" for k, v in stats.items()), title)

    def config_show(self, config: dict):
        """Show config."""
        lines = []
        for section, values in config.items():
            if section.startswith("_"):
                continue
            lines.append(f"[bold]{section}[/bold]")
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, list):
                        value = ", ".join(str(v) for v in value)
                    lines.append(f"  {key}: {value}")
            else:
                lines.append(f"  {values}")
            lines.append("")

        self.panel("\n".join(lines), "当前配置", "cyan")

    def help_command(self, command: str, description: str, usage: str, examples: list):
        """Show command help."""
        content = f"[bold]描述[/bold]\n{description}\n\n"
        content += f"[bold]用法[/bold]\n{usage}\n\n"
        if examples:
            content += f"[bold]示例[/bold]\n"
            for example in examples:
                content += f"  {example}\n"

        self.panel(content, f"ppc10 {command}", "green")

    def check_result(self, checks: list):
        """Show check results."""
        table = Table(title="检查结果", show_header=True)
        table.add_column("项目", style="bold")
        table.add_column("状态")
        table.add_column("详情")

        for check in checks:
            status = "+" if check["status"] else "-"
            status_style = OutputStyle.SUCCESS if check["status"] else OutputStyle.ERROR
            table.add_row(
                check["name"],
                Text(status, style=status_style),
                check.get("detail", "")
            )

        console.print(table)

    def retry_status(self, info: RetryInfo):
        """Show retry status."""
        if info.will_retry:
            msg = f"↻ 第 {info.attempt}/{info.max_attempts} 次尝试失败：{info.error}，{info.delay:.1f}s 后重试"
            console.print(msg, style=OutputStyle.RETRY)
        else:
            msg = f"- 已重试 {info.attempt} 次，最终失败：{info.error}"
            console.print(msg, style=OutputStyle.ERROR)

    def task_status(self, task: TaskStatus):
        """Show single task status."""
        status_icons = {
            "pending": "o",
            "running": "◐",
            "completed": "+",
            "failed": "-"
        }
        status_styles = {
            "pending": OutputStyle.INFO,
            "running": OutputStyle.TASK_RUNNING,
            "completed": OutputStyle.TASK_COMPLETED,
            "failed": OutputStyle.TASK_FAILED
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
        success_rate = (succeeded / total * 100) if total > 0 else 0

        content_lines = [
            f"[bold]总任务数:[/bold] {total}",
            f"[bold green]成功:[/bold green] {succeeded}",
            f"[bold red]失败:[/bold red] {failed}",
            f"[bold]成功率:[/bold] {success_rate:.1f}%",
            f"[bold]总用时:[/bold] {duration:.1f}s",
        ]

        if retries > 0:
            content_lines.append(f"[bold yellow]总重试次数:[/bold yellow] {retries}")

        if duration > 0 and total > 0:
            avg_time = duration / total
            content_lines.append(f"[bold]平均耗时:[/bold] {avg_time:.2f}s/任务")

        self.panel("\n".join(content_lines), "处理结果汇总", "cyan")

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

            for tid, task in self._task_statuses.items():
                status_icons = {
                    "pending": "[dim]o[/dim]",
                    "running": "[cyan]◐[/cyan]",
                    "completed": "[green]+[/green]",
                    "failed": "[red]-[/red]"
                }
                icon = status_icons.get(task.status, "?")

                progress_str = f"{task.progress:.0%}" if task.progress > 0 else "-"
                retry_str = str(task.retries) if task.retries > 0 else "-"

                table.add_row(
                    task.name,
                    f"{icon} {task.status}",
                    progress_str,
                    retry_str
                )

            self._live.update(table)

    def show_banner(self):
        """Show brand banner."""
        logo_lines = BrandAssets.LOGO_ASCII.strip().split('\n')
        colored_logo = "\n".join(
            f"[{BrandColors.PRIMARY}]{line}[/{BrandColors.PRIMARY}]"
            for line in logo_lines
        )
        self.console.print(colored_logo)
        self.console.print()
        version_text = f"[{BrandColors.ACCENT}]版本 {BrandAssets.VERSION}[/{BrandColors.ACCENT}]"
        tagline_text = f"[{BrandColors.SECONDARY}]{BrandAssets.TAGLINE}[/{BrandColors.SECONDARY}]"
        self.console.print(f"  {version_text}  |  {tagline_text}")
        self.console.print()
        copyright_text = f"[{BrandColors.TEXT_SECONDARY}]{BrandAssets.COPYRIGHT}[/{BrandColors.TEXT_SECONDARY}]"
        self.console.print(f"  {copyright_text}")
        self.console.print()

    def show_welcome(self):
        """Show welcome info."""
        welcome_content = f"""[{BrandColors.PRIMARY}]欢迎使用 PPC10![/{BrandColors.PRIMARY}]

[{BrandColors.TEXT_SECONDARY}]冰璃岩项目开发组 - 一个功能强大的文本转语音工具，支持多种语音引擎和批量处理。[/{BrandColors.TEXT_SECONDARY}]"""
        self.console.print(Panel(
            welcome_content,
            title=f"[{BrandColors.ACCENT}][ROCKET] 快速开始 [/{BrandColors.ACCENT}]",
            border_style=BrandColors.PRIMARY,
            expand=False
        ))
        self.console.print()
        version_line = f"[bold {BrandColors.PRIMARY}]PPC10[/bold {BrandColors.PRIMARY}] v[{BrandColors.ACCENT}]{BrandAssets.VERSION}[/{BrandColors.ACCENT}] [{BrandColors.TEXT_SECONDARY}]│[/{BrandColors.TEXT_SECONDARY}] [{BrandColors.SECONDARY}]冰璃岩文本转语音工具[/{BrandColors.SECONDARY}]"
        self.console.print(f"  {version_line}")
        self.console.print(f"  [{BrandColors.TEXT_SECONDARY}]{'─' * 50}[/{BrandColors.TEXT_SECONDARY}]")
        self.console.print()
        commands_table = Table(show_header=False, box=None, padding=(0, 2))
        commands_table.add_column("命令", style=f"bold {BrandColors.PRIMARY}")
        commands_table.add_column("说明", style=BrandColors.TEXT_SECONDARY)
        commands_table.add_row(f"{Icons.SOUND} ppc10 convert", "转换文本为语音")
        commands_table.add_row(f"{Icons.FOLDER} ppc10 batch", "批量处理文件")
        commands_table.add_row(f"{Icons.GEAR} ppc10 config", "配置设置")
        commands_table.add_row(f"{Icons.INFO} ppc10 --help", "查看帮助信息")
        self.console.print(Panel(
            commands_table,
            title=f"[{BrandColors.SECONDARY}][BOOK] 常用命令 [/{BrandColors.SECONDARY}]",
            border_style=BrandColors.SECONDARY,
            expand=False
        ))
        self.console.print()
        tips = [
            (f"[bold {BrandColors.ACCENT}]i[/bold {BrandColors.ACCENT}]", f"使用 [bold]ppc10 convert -i input.txt[/bold] 快速转换单个文件"),
            (f"[bold {BrandColors.ACCENT}]i[/bold {BrandColors.ACCENT}]", f"使用 [bold]ppc10 config wizard[/bold] 进行交互式配置"),
            (f"[bold {BrandColors.ACCENT}]i[/bold {BrandColors.ACCENT}]", f"使用 [bold]ppc10 --verbose[/bold] 获取详细输出信息"),
        ]
        tips_table = Table(show_header=False, box=None, padding=(0, 2))
        tips_table.add_column("图标", width=3)
        tips_table.add_column("提示", style=BrandColors.TEXT_SECONDARY)
        for icon, tip in tips:
            tips_table.add_row(icon, tip)
        self.console.print(Panel(
            tips_table,
            title=f"[{BrandColors.ACCENT}][STAR] 快速入门 [/{BrandColors.ACCENT}]",
            border_style=BrandColors.ACCENT,
            expand=False
        ))
        self.console.print()

    def interactive_help(self, commands: Dict[str, Dict]):
        """Interactive help browser."""
        self.title("[BOOK] PPC10 交互式帮助浏览器")

        command_list = list(commands.keys())
        current_idx = 0
        search_filter = ""

        category_map = {}
        for cmd_name, cmd_info in commands.items():
            cat = cmd_info.get("category", "其他")
            if cat not in category_map:
                category_map[cat] = []
            category_map[cat].append(cmd_name)

        category_colors = {
            "基础": BrandColors.SUCCESS,
            "转换": BrandColors.INFO,
            "配置": BrandColors.ACCENT,
            "高级": "magenta",
            "工具": BrandColors.WARNING,
            "其他": BrandColors.TEXT_SECONDARY,
        }

        while True:
            filtered_commands = [
                cmd for cmd in command_list
                if search_filter.lower() in cmd.lower()
            ]

            if not filtered_commands:
                filtered_commands = command_list
                if search_filter:
                    console.print("\n[yellow]未找到匹配的命令，显示全部命令[/yellow]")

            if current_idx >= len(filtered_commands):
                current_idx = 0

            console.clear()
            console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
            console.print(f"[bold white]  [BOOK] PPC10 交互式帮助浏览器[/bold white]")
            console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
            console.print(f"  [{BrandColors.TEXT_SECONDARY}]PPC10 v{BrandAssets.VERSION} | 输入 / 搜索命令[/{BrandColors.TEXT_SECONDARY}]")
            console.print(f"  [{BrandColors.TEXT_SECONDARY}]{'─' * 56}[/{BrandColors.TEXT_SECONDARY}]")

            if search_filter:
                console.print(f"\n[dim]搜索: [bold yellow]{search_filter}[/bold yellow][/dim]")

            for cat, cat_cmds in category_map.items():
                cat_filtered = [c for c in cat_cmds if c in filtered_commands]
                if not cat_filtered:
                    continue
                cat_color = category_colors.get(cat, BrandColors.TEXT_SECONDARY)
                console.print(f"\n[bold {cat_color}]  {cat}:[/bold {cat_color}]")
                for cmd in cat_filtered:
                    i = filtered_commands.index(cmd)
                    if i == current_idx:
                        console.print(f"    [bold green]▶ {cmd}[/bold green] - {commands[cmd].get('desc', '无描述')}")
                    else:
                        console.print(f"      [dim]{cmd}[/dim] - {commands[cmd].get('desc', '无描述')}")

            console.print(f"\n  [{BrandColors.TEXT_SECONDARY}]{'─' * 56}[/{BrandColors.TEXT_SECONDARY}]")
            console.print("[dim]操作: [bold]↑/↓[/bold] 导航 | [bold]Enter[/bold] 查看详情 | [bold]/[/bold] 搜索 | [bold]q[/bold] 退出[/dim]")

            selected_cmd = filtered_commands[current_idx] if filtered_commands else None
            if selected_cmd:
                cmd_info = commands[selected_cmd]
                preview = Panel(
                    f"[bold]描述:[/bold] {cmd_info.get('desc', '无描述')}\n"
                    f"[bold]用法:[/bold] {cmd_info.get('usage', '无用法说明')}",
                    title=f"[bold green]{selected_cmd}[/bold green]",
                    border_style="green",
                    box=SIMPLE
                )
                console.print(preview)

            try:
                key = console.input()

                if key == 'q' or key == 'Q':
                    console.print("\n[dim]退出帮助浏览器[/dim]")
                    break
                elif key == '/' or key == 's' or key == 'S':
                    search_filter = Prompt.ask("\n[bold cyan]输入搜索关键词[/bold cyan]", default="")
                    current_idx = 0
                elif key == 'UP' or key == 'k':
                    current_idx = (current_idx - 1) % len(filtered_commands)
                elif key == 'DOWN' or key == 'j':
                    current_idx = (current_idx + 1) % len(filtered_commands)
                elif key == 'ENTER' or key == '\r' or key == '\n':
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

    def _show_command_detail(self, command: str, info: Dict):
        """Show command detail."""
        console.clear()
        console.print(f"\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        console.print(f"[bold white]  命令详情: [bold green]{command}[/bold green][/bold white]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

        console.print(f"[bold]描述:[/bold]")
        console.print(f"  {info.get('desc', '无描述')}\n")

        console.print(f"[bold]用法:[/bold]")
        console.print(f"  [cyan]{info.get('usage', '无用法说明')}[/cyan]\n")

        examples = info.get('examples', [])
        if examples:
            console.print(f"[bold]示例:[/bold]")
            for ex in examples:
                if isinstance(ex, dict):
                    console.print(f"  [green]•[/green] [dim]{ex.get('desc', '')}[/dim]")
                    console.print(f"    [bold]$ {ex.get('cmd', '')}[/bold]")
                else:
                    console.print(f"  [green]•[/green] {ex}")
            console.print()

        options = info.get('options', [])
        if options:
            console.print(f"[bold]选项:[/bold]")
            for opt in options:
                if isinstance(opt, dict):
                    opt_name = opt.get('name', '')
                    opt_desc = opt.get('desc', '')
                    console.print(f"  [yellow]{opt_name}[/yellow] - {opt_desc}")
            console.print()

        see_also = info.get('see_also', [])
        if see_also:
            console.print(f"[bold]相关命令:[/bold]")
            console.print(f"  {', '.join(see_also)}\n")

        console.print("[bold cyan]───────────────────────────────────────────────────────────────[/bold cyan]")
        Prompt.ask("\n[dim]按 Enter 返回[/dim]")

    def command_examples(self, command: str, examples: List[Dict]):
        """Show command examples."""
        console.print(f"\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        console.print(f"[bold white]  i {command} 命令示例[/bold white]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

        for i, example in enumerate(examples, 1):
            desc = example.get('desc', '无描述')
            cmd = example.get('cmd', '')
            output = example.get('output', None)

            console.print(f"[bold yellow]示例 {i}:[/bold yellow] {desc}")
            console.print(Panel(
                f"[bold green]$ {cmd}[/bold green]",
                border_style="green",
                box=SIMPLE,
                padding=(0, 1)
            ))

            if output:
                console.print(f"[dim]预期输出:[/dim]")
                console.print(Panel(
                    output,
                    border_style="dim",
                    box=SIMPLE,
                    padding=(0, 1)
                ))
            console.print()

        console.print("[bold cyan]───────────────────────────────────────────────────────────────[/bold cyan]")

    def show_shortcuts(self):
        """Show shortcut tips."""
        self.console.print()
        self.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
        self.console.print(f"[bold white]  [GEAR] PPC10 快捷键参考[/bold white]")
        self.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
        self.console.print(f"  [{BrandColors.TEXT_SECONDARY}]{'─' * 56}[/{BrandColors.TEXT_SECONDARY}]")
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

        table1 = Table(show_header=True, box=ROUNDED, border_style=BrandColors.INFO, title=f"[bold {BrandColors.INFO}]  全局快捷键  [/bold {BrandColors.INFO}]")
        table1.add_column("快捷键", style=f"bold {BrandColors.ACCENT}", width=20)
        table1.add_column("功能", style="white", width=40)
        for shortcut, desc in global_shortcuts:
            table1.add_row(shortcut, desc)

        table2 = Table(show_header=True, box=ROUNDED, border_style=BrandColors.SUCCESS, title=f"[bold {BrandColors.SUCCESS}]  命令快捷方式  [/bold {BrandColors.SUCCESS}]")
        table2.add_column("命令", style=f"bold {BrandColors.SUCCESS}", width=20)
        table2.add_column("功能", style="white", width=40)
        for cmd, desc in command_shortcuts:
            table2.add_row(cmd, desc)

        table3 = Table(show_header=True, box=ROUNDED, border_style="magenta", title=f"[bold magenta]  交互模式快捷键  [/bold magenta]")
        table3.add_column("快捷键", style="bold magenta", width=20)
        table3.add_column("功能", style="white", width=40)
        for shortcut, desc in interactive_shortcuts:
            table3.add_row(shortcut, desc)

        columns = Columns([table1, table2, table3], equal=True, expand=True)
        self.console.print(columns)

        self.console.print()
        self.console.print(f"  [{BrandColors.TEXT_SECONDARY}]{'─' * 56}[/{BrandColors.TEXT_SECONDARY}]")
        self.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")

    def help_command_enhanced(
        self,
        command: str,
        description: str,
        usage: str,
        examples: List[Dict],
        options: List[Dict] = None,
        see_also: List[str] = None
    ):
        """Enhanced command help."""
        console.print(f"\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        console.print(f"[bold white]  [BOOK] ppc10 {command}[/bold white]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

        desc_panel = Panel(
            description,
            title="[bold yellow]描述[/bold yellow]",
            border_style="yellow",
            box=SIMPLE,
            padding=(0, 1)
        )
        console.print(desc_panel)

        usage_panel = Panel(
            f"[bold cyan]{usage}[/bold cyan]",
            title="[bold green]用法[/bold green]",
            border_style="green",
            box=SIMPLE,
            padding=(0, 1)
        )
        console.print(usage_panel)

        if options:
            console.print(f"\n[bold magenta]选项:[/bold magenta]")
            opt_table = Table(show_header=True, box=SIMPLE, border_style="dim")
            opt_table.add_column("选项", style="bold yellow", width=25)
            opt_table.add_column("说明", style="white", width=50)
            opt_table.add_column("默认值", style="dim", width=15)

            for opt in options:
                opt_name = opt.get('name', '')
                opt_desc = opt.get('desc', '')
                opt_default = opt.get('default', '')
                opt_table.add_row(opt_name, opt_desc, f"[dim]{opt_default}[/dim]")

            console.print(opt_table)

        if examples:
            console.print(f"\n[bold blue]示例:[/bold blue]")
            for i, ex in enumerate(examples, 1):
                if isinstance(ex, dict):
                    desc = ex.get('desc', '')
                    cmd = ex.get('cmd', '')
                    console.print(f"  [bold]{i}.[/bold] [dim]{desc}[/dim]")
                    console.print(Panel(
                        f"[bold green]$ {cmd}[/bold green]",
                        border_style="green",
                        box=SIMPLE,
                        padding=(0, 1)
                    ))
                else:
                    console.print(f"  [bold]{i}.[/bold] {ex}")

        if see_also:
            console.print(f"\n[bold cyan]相关命令:[/bold cyan]")
            see_also_str = "  ".join(f"[bold green]{cmd}[/bold green]" for cmd in see_also)
            console.print(f"  {see_also_str}")

        console.print("\n[bold cyan]───────────────────────────────────────────────────────────────[/bold cyan]")

    def help_index(self, categories: Dict[str, List[str]]):
        """Show help index."""
        self.console.print()
        self.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
        self.console.print(f"[bold white]  [BOOK] PPC10 命令索引[/bold white]")
        self.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
        self.console.print(f"  [{BrandColors.TEXT_SECONDARY}]PPC10 v{BrandAssets.VERSION} | 冰璃岩文本转语音工具[/{BrandColors.TEXT_SECONDARY}]")
        self.console.print(f"  [{BrandColors.TEXT_SECONDARY}]{'─' * 56}[/{BrandColors.TEXT_SECONDARY}]")
        self.console.print()

        category_groups = {
            "基础命令": [],
            "高级命令": [],
            "扩展命令": [],
        }

        category_icons = {
            "转换命令": "R",
            "配置命令": "G",
            "工具命令": "T",
            "信息命令": "i",
            "默认": "[FOLDER]",
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
            "基础命令": f"[{BrandColors.SUCCESS}]+[/{BrandColors.SUCCESS}]",
            "高级命令": f"[{BrandColors.ACCENT}]*[/{BrandColors.ACCENT}]",
            "扩展命令": f"[{BrandColors.INFO}]i[/{BrandColors.INFO}]",
        }

        for group_name, group_items in category_groups.items():
            if not group_items:
                continue
            group_icon = group_icons.get(group_name, "[FOLDER]")
            self.console.print(f"[bold {BrandColors.PRIMARY}]{group_icon} {group_name}[/bold {BrandColors.PRIMARY}]")
            self.console.print(f"  [{BrandColors.TEXT_SECONDARY}]{'─' * 40}[/{BrandColors.TEXT_SECONDARY}]")

            for category, commands_list in group_items:
                icon = category_icons.get(category, category_icons["默认"])
                self.console.print(f"  [bold]{icon} {category}:[/bold]")

                for cmd in commands_list:
                    self.console.print(f"    [green]•[/green] [bold cyan]{cmd}[/bold cyan]")
                self.console.print()

        self.console.print(f"  [{BrandColors.TEXT_SECONDARY}]{'─' * 56}[/{BrandColors.TEXT_SECONDARY}]")
        self.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
        self.console.print(f"[dim]PPC10 v{BrandAssets.VERSION} | 使用 [bold]ppc10 <command> --help[/bold] 查看命令详细帮助[/dim]")
        self.console.print(f"[dim]使用 [bold]ppc10 help[/bold] 进入交互式帮助浏览器[/dim]")

    def enhanced_stats_panel(self, stats: Dict[str, Any], title: str = "详细统计") -> Panel:
        """Create enhanced stats panel."""
        lines = []

        if "total" in stats:
            total = stats["total"]
            completed = stats.get("completed", 0)
            failed = stats.get("failed", 0)
            pending = stats.get("pending", total - completed - failed)

            progress_bar = self._create_mini_progress_bar(completed + failed, total)
            lines.append(f"[bold]任务统计[/bold]")
            lines.append(f"  总数：{total}")
            lines.append(f"  [{BrandColors.SUCCESS}]✓ 完成:[/{BrandColors.SUCCESS}] {completed}")
            lines.append(f"  [{BrandColors.ERROR}]✗ 失败:[/{BrandColors.ERROR}] {failed}")
            lines.append(f"  [{BrandColors.TEXT_SECONDARY}]○ 待处理:[/{BrandColors.TEXT_SECONDARY}] {pending}")
            lines.append("")

        if "success_rate" in stats:
            rate = stats["success_rate"]
            rate_color = BrandColors.SUCCESS if rate >= 90 else (BrandColors.WARNING if rate >= 70 else BrandColors.ERROR)
            retry_rate = stats.get("retry_rate", 0)
            quarantined_rate = stats.get("quarantined_rate", 0)

            lines.append(f"[bold]质量指标[/bold]")
            lines.append(f"  成功率：[{rate_color}]{rate:.1f}%[/{rate_color}]")
            lines.append(f"  重试率：[{BrandColors.WARNING}]{retry_rate:.1f}%[/{BrandColors.WARNING}]")
            lines.append(f"  隔离率：[{BrandColors.ACCENT}]{quarantined_rate:.1f}%[/{BrandColors.ACCENT}]")
            lines.append("")

        if "current_speed" in stats:
            current_speed = stats["current_speed"]
            average_speed = stats.get("average_speed", 0)
            p95_speed = stats.get("p95_speed", 0)

            lines.append(f"[bold]性能指标[/bold]")
            lines.append(f"  当前速度：[{BrandColors.INFO}]{current_speed:.2f}[/{BrandColors.INFO}] 任务/秒")
            lines.append(f"  平均速度：[{BrandColors.INFO}]{average_speed:.2f}[/{BrandColors.INFO}] 任务/秒")
            if p95_speed > 0:
                lines.append(f"  P95 速度：[{BrandColors.ACCENT}]{p95_speed:.2f}[/{BrandColors.ACCENT}] 任务/秒")
            lines.append("")

        if "elapsed" in stats:
            elapsed = stats["elapsed"]
            eta = stats.get("eta", 0)
            avg_duration = stats.get("avg_task_duration", 0)

            lines.append(f"[bold]时间统计[/bold]")
            lines.append(f"  已用时间：{self._format_duration(elapsed)}")
            lines.append(f"  预计剩余：{self._format_duration(eta)}")
            lines.append(f"  平均任务耗时：{avg_duration:.2f}秒")
            lines.append("")

        if stats.get("total_retries", 0) > 0 or stats.get("quarantined_count", 0) > 0 or stats.get("circuit_breaker_trips", 0) > 0:
            lines.append(f"[bold]可靠性[/bold]")
            if stats.get("total_retries", 0) > 0:
                lines.append(f"  总重试次数：[{BrandColors.WARNING}]{stats['total_retries']}[/{BrandColors.WARNING}]")
            if stats.get("quarantined_count", 0) > 0:
                lines.append(f"  隔离任务数：[{BrandColors.ACCENT}]{stats['quarantined_count']}[/{BrandColors.ACCENT}]")
            if stats.get("circuit_breaker_trips", 0) > 0:
                lines.append(f"  熔断器触发：[{BrandColors.ERROR}]{stats['circuit_breaker_trips']}[/{BrandColors.ERROR}]")
            lines.append("")

        if stats.get("error_type_counts"):
            lines.append(f"[bold]错误分类[/bold]")
            for error_type, count in sorted(stats["error_type_counts"].items(), key=lambda x: x[1], reverse=True)[:5]:
                lines.append(f"  {error_type}: {count}")

        return Panel(
            "\n".join(lines),
            title=f"[bold {BrandColors.PRIMARY}]{Icons.CHART} {title}[/bold {BrandColors.PRIMARY}]",
            border_style=BrandColors.PRIMARY,
            box=ROUNDED,
            padding=(1, 2)
        )

    def completion_report(
        self,
        stats: Dict[str, Any],
        executor_stats: Dict[str, Any] = None,
        title: str = "转换完成报告"
    ) -> None:
        """Show completion report."""
        self.console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
        self.console.print(f"[bold white]  {Icons.STAR} {title}[/bold white]")
        self.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n")

        total = stats.get("total", 0)
        completed = stats.get("completed", 0)
        failed = stats.get("failed", 0)
        success_rate = stats.get("success_rate", 0)

        if success_rate >= 90:
            result_icon = Icons.SUCCESS
            result_color = BrandColors.SUCCESS
            result_text = "优秀"
        elif success_rate >= 70:
            result_icon = Icons.WARNING
            result_color = BrandColors.WARNING
            result_text = "良好"
        else:
            result_icon = Icons.ERROR
            result_color = BrandColors.ERROR
            result_text = "需改进"

        self.console.print(f"[bold {result_color}]{result_icon} 总体评价：{result_text}[/bold {result_color}]")
        self.console.print()

        summary_table = Table(show_header=False, box=SIMPLE, border_style=BrandColors.PRIMARY)
        summary_table.add_column("指标", style="bold", width=20)
        summary_table.add_column("值", style="cyan", width=20)

        summary_table.add_row("总任务数", str(total))
        summary_table.add_row(f"[{BrandColors.SUCCESS}]成功[/{BrandColors.SUCCESS}]", f"[{BrandColors.SUCCESS}]{completed}[/{BrandColors.SUCCESS}]")
        summary_table.add_row(f"[{BrandColors.ERROR}]失败[/{BrandColors.ERROR}]", f"[{BrandColors.ERROR}]{failed}[/{BrandColors.ERROR}]")
        summary_table.add_row("成功率", f"[{result_color}]{success_rate:.1f}%[/{result_color}]")

        if "elapsed" in stats:
            summary_table.add_row("总用时", self._format_duration(stats["elapsed"]))

        if "average_speed" in stats:
            summary_table.add_row("平均速度", f"{stats['average_speed']:.2f} 任务/秒")

        self.console.print(summary_table)
        self.console.print()

        if stats.get("error_type_counts"):
            self.console.print(f"[bold {BrandColors.ERROR}]i 错误类型分布:[/bold {BrandColors.ERROR}]\n")

            error_table = Table(show_header=True, box=ROUNDED, border_style=BrandColors.ERROR)
            error_table.add_column("错误类型", style="bold yellow", width=30)
            error_table.add_column("数量", style="red", width=10, justify="right")
            error_table.add_column("占比", style="white", width=10, justify="right")

            total_errors = sum(stats["error_type_counts"].values())
            for error_type, count in sorted(stats["error_type_counts"].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_errors * 100) if total_errors > 0 else 0
                error_table.add_row(
                    error_type,
                    str(count),
                    f"{percentage:.1f}%"
                )

            self.console.print(error_table)
            self.console.print()

        if executor_stats:
            self.console.print(f"[bold {BrandColors.ACCENT}]G 执行器统计:[/bold {BrandColors.ACCENT}]\n")

            exec_table = Table(show_header=False, box=SIMPLE, border_style=BrandColors.ACCENT)
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
                    state_color = BrandColors.SUCCESS if cb_stats['current_state'] == 'CLOSED' else (BrandColors.WARNING if cb_stats['current_state'] == 'HALF_OPEN' else BrandColors.ERROR)
                    exec_table.add_row("熔断器", f"[{state_color}]状态：{cb_stats['current_state']}[/{state_color}]")
                if "total_calls" in cb_stats:
                    exec_table.add_row("", f"总调用：{cb_stats['total_calls']}, 失败：{cb_stats['failed_calls']}")

            self.console.print(exec_table)
            self.console.print()

        self.console.print(f"[bold {BrandColors.PRIMARY}]{'─' * 60}[/bold {BrandColors.PRIMARY}]")
        self.console.print(f"[dim]报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")

    def stats_panel(self, stats: Dict[str, Any], title: str = "实时统计") -> Panel:
        """Create real-time stats panel."""
        lines = []

        if "total" in stats and "completed" in stats:
            total = stats["total"]
            completed = stats["completed"]
            failed = stats.get("failed", 0)
            pending = total - completed - failed

            progress_bar = self._create_mini_progress_bar(completed + failed, total)
            lines.append(f"[bold]进度[/bold] {progress_bar} {completed + failed}/{total}")
            lines.append(f"  [{BrandColors.SUCCESS}]✓ 完成:[/{BrandColors.SUCCESS}] {completed}")
            lines.append(f"  [{BrandColors.ERROR}]✗ 失败:[/{BrandColors.ERROR}] {failed}")
            lines.append(f"  [{BrandColors.TEXT_SECONDARY}]○ 待处理:[/{BrandColors.TEXT_SECONDARY}] {pending}")

        if "success_rate" in stats:
            rate = stats["success_rate"]
            rate_color = BrandColors.SUCCESS if rate >= 90 else (BrandColors.WARNING if rate >= 70 else BrandColors.ERROR)
            lines.append(f"[bold]成功率[/bold] [{rate_color}]{rate:.1f}%[/{rate_color}]")

        if "current_speed" in stats or "speed" in stats:
            speed = stats.get("current_speed", stats.get("speed", 0))
            lines.append(f"[bold]当前速度[/bold] [{BrandColors.INFO}]{speed:.2f}[/{BrandColors.INFO}] 任务/秒")

        if "average_speed" in stats:
            avg_speed = stats["average_speed"]
            lines.append(f"[bold]平均速度[/bold] [{BrandColors.INFO}]{avg_speed:.2f}[/{BrandColors.INFO}] 任务/秒")

        if "p95_speed" in stats and stats["p95_speed"] > 0:
            p95_speed = stats["p95_speed"]
            lines.append(f"[bold]P95 速度[/bold] [{BrandColors.ACCENT}]{p95_speed:.2f}[/{BrandColors.ACCENT}] 任务/秒")

        if "elapsed" in stats:
            elapsed = stats["elapsed"]
            lines.append(f"[bold]已用时间[/bold] {self._format_duration(elapsed)}")

        if "eta" in stats:
            eta = stats["eta"]
            lines.append(f"[bold]预计剩余[/bold] {self._format_duration(eta)}")

        if "quarantined" in stats and stats["quarantined"] > 0:
            quarantined = stats["quarantined"]
            lines.append(f"[bold {BrandColors.WARNING}]! 隔离任务:[/{BrandColors.WARNING}] {quarantined}")

        if "circuit_breaker_trips" in stats and stats["circuit_breaker_trips"] > 0:
            trips = stats["circuit_breaker_trips"]
            lines.append(f"[bold {BrandColors.ERROR}]! 熔断器触发:[/{BrandColors.ERROR}] {trips}")

        return Panel(
            "\n".join(lines),
            title=f"[bold {BrandColors.PRIMARY}]{Icons.CHART} {title}[/bold {BrandColors.PRIMARY}]",
            border_style=BrandColors.PRIMARY,
            box=ROUNDED,
            padding=(1, 2)
        )

    def _create_mini_progress_bar(self, current: int, total: int, width: int = 20) -> str:
        """Create mini progress bar."""
        percent = current / total if total > 0 else 0
        filled = int(width * percent)
        bar = f"[{BrandColors.SUCCESS}]{'█' * filled}[/{BrandColors.SUCCESS}]{('░' * (width - filled))}"
        return f"[{bar}] {percent:.0%}"

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

    def decorate_status(self, status: str, text: str = None) -> str:
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

    def check_result_enhanced(
        self,
        checks: List[Dict],
        title: str = "检查结果",
        show_summary: bool = True
    ) -> None:
        """Enhanced check results."""
        table = Table(
            title=title,
            show_header=True,
            box=ROUNDED,
            border_style=BrandColors.PRIMARY,
        )
        table.add_column("项目", style="bold", width=20)
        table.add_column("状态", width=10)
        table.add_column("详情", width=40)

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

            table.add_row(name, status_text, detail)

        console.print(table)

        if show_summary:
            total = passed + failed
            if total > 0:
                pass_rate = (passed / total) * 100
                summary_color = "green" if pass_rate == 100 else ("yellow" if pass_rate >= 50 else "red")

                summary = Panel(
                    f"[bold]总计:[/bold] {total}  "
                    f"[green]通过:[/green] {passed}  "
                    f"[red]失败:[/red] {failed}  "
                    f"[bold {summary_color}]通过率:[/bold {summary_color}] {pass_rate:.1f}%",
                    title="[bold]汇总[/bold]",
                    border_style=summary_color,
                    box=SIMPLE,
                )
                console.print(summary)



class ParallelProgress:
    """Multi-task parallel progress display."""

    def __init__(self, console: Console, max_workers: int = 4):
        self.console = console
        self.max_workers = max_workers
        self._progress: Optional[Progress] = None
        self._tasks: Dict[str, int] = {}
        self._overall_task: Optional[int] = None
        self._total_tasks: int = 0
        self._completed_tasks: int = 0
        self._failed_tasks: int = 0
        self._start_time: Optional[float] = None

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
            console=self.console
        )
        self._progress.start()

        self._overall_task = self._progress.add_task(
            f"[{BrandColors.PRIMARY}]{description}[/{BrandColors.PRIMARY}]",
            total=total_tasks
        )

    def add_task(self, name: str, total: int = 100) -> None:
        """Add sub-task."""
        if self._progress is None:
            return

        task_id = self._progress.add_task(
            f"  [{BrandColors.TEXT_SECONDARY}]{name}[/{BrandColors.TEXT_SECONDARY}]",
            total=total
        )
        self._tasks[name] = task_id

    def update_task(self, name: str, advance: int = 1, description: str = None) -> None:
        """Update sub-task progress."""
        if self._progress is None or name not in self._tasks:
            return

        task_id = self._tasks[name]
        if description:
            self._progress.update(task_id, description=description, advance=advance)
        else:
            self._progress.advance(task_id, advance)

    def complete_task(self, name: str) -> None:
        """Mark task as completed."""
        if self._progress is None or name not in self._tasks:
            return

        task_id = self._tasks[name]
        self._progress.update(
            task_id,
            description=f"  [{BrandColors.SUCCESS}]+ {name}[/{BrandColors.SUCCESS}]",
            completed=self._progress.tasks[task_id].total
        )

        self._completed_tasks += 1
        if self._overall_task is not None:
            self._progress.advance(self._overall_task, 1)

    def fail_task(self, name: str, error: str) -> None:
        """Mark task as failed."""
        if self._progress is None or name not in self._tasks:
            return

        task_id = self._tasks[name]
        self._progress.update(
            task_id,
            description=f"  [{BrandColors.ERROR}]- {name}: {error}[/{BrandColors.ERROR}]"
        )

        self._failed_tasks += 1
        if self._overall_task is not None:
            self._progress.advance(self._overall_task, 1)

    def stop(self) -> None:
        """Stop progress display."""
        if self._progress:
            self._progress.stop()
            self._progress = None

    def get_stats(self) -> Dict[str, Any]:
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
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )

    return logging.getLogger("ppc10")


def config_wizard(console: Console = None, full: bool = False) -> Dict[str, Any]:
    """Interactive configuration wizard."""
    if console is None:
        console = Console()

    config = {}

    total_steps = 8 if full else 3
    current_step = 0

    def _step_header(step_num: int, step_title: str, step_icon: str = "⚙"):
        nonlocal current_step
        current_step = step_num
        progress_pct = int((step_num - 1) / total_steps * 100)
        bar_width = 30
        filled = int(bar_width * (step_num - 1) / total_steps)
        progress_bar = f"[{BrandColors.SUCCESS}]{'█' * filled}[/{BrandColors.SUCCESS}]{'░' * (bar_width - filled)}"
        console.print()
        console.print(f"  [{BrandColors.TEXT_SECONDARY}]步骤 {step_num}/{total_steps}[/{BrandColors.TEXT_SECONDARY}] [{progress_bar}] {progress_pct}%")
        console.print(f"[bold {BrandColors.ACCENT}]{step_icon} {step_title}[/bold {BrandColors.ACCENT}]")
        console.print(f"[dim]{'─' * 30}[/dim]")

    console.print()
    console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    console.print(f"[bold white]  {Icons.GEAR} PPC10 配置向导[/bold white]")
    console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    console.print(f"  [{BrandColors.TEXT_SECONDARY}]PPC10 v{BrandAssets.VERSION} | 冰璃岩文本转语音工具[/{BrandColors.TEXT_SECONDARY}]")
    console.print(f"  [{BrandColors.TEXT_SECONDARY}]{'─' * 56}[/{BrandColors.TEXT_SECONDARY}]")
    console.print()

    if full:
        console.print(f"[{BrandColors.TEXT_SECONDARY}]完整配置模式：请按照提示完成所有配置。[/{BrandColors.TEXT_SECONDARY}]")
    else:
        console.print(f"[{BrandColors.TEXT_SECONDARY}]快速配置模式：仅核心设置。使用 [bold]--full[/bold] 可配置所有项。[/{BrandColors.TEXT_SECONDARY}]")
    console.print()

    _step_header(1, "TTS 核心配置", "R")

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

    console.print(f"\n[{BrandColors.INFO}]可用语音选项:[/{BrandColors.INFO}]")
    for i, (voice_id, voice_desc) in enumerate(voice_options, 1):
        console.print(f"  [bold cyan]{i}.[/bold cyan] {voice_desc} [dim]({voice_id})[/dim]")

    voice_choice = Prompt.ask(
        f"\n[{BrandColors.PRIMARY}]请选择语音[/{BrandColors.PRIMARY}]",
        choices=[str(i) for i in range(1, len(voice_options) + 1)],
        default="2"
    )
    config["tts.voice"] = voice_options[int(voice_choice) - 1][0]

    console.print()
    concurrency = Prompt.ask(
        f"[{BrandColors.PRIMARY}]并发数 (1-64)[/{BrandColors.PRIMARY}]",
        default="8"
    )
    try:
        config["tts.concurrency"] = max(1, min(64, int(concurrency)))
    except ValueError:
        config["tts.concurrency"] = 8

    console.print()
    timeout_mode_options = [
        ("auto", "自动 - 根据文本长度动态调整"),
        ("fixed", "固定 - 使用固定超时值"),
        ("adaptive", "自适应 - 基于历史记录调整"),
    ]
    console.print(f"[{BrandColors.INFO}]超时模式选项:[/{BrandColors.INFO}]")
    for i, (mode_id, mode_desc) in enumerate(timeout_mode_options, 1):
        console.print(f"  [bold cyan]{i}.[/bold cyan] {mode_desc} [dim]({mode_id})[/dim]")

    timeout_choice = Prompt.ask(
        f"\n[{BrandColors.PRIMARY}]请选择超时模式[/{BrandColors.PRIMARY}]",
        choices=[str(i) for i in range(1, len(timeout_mode_options) + 1)],
        default="1"
    )
    config["tts.timeout_mode"] = timeout_mode_options[int(timeout_choice) - 1][0]

    console.print()
    timeout = Prompt.ask(
        f"[{BrandColors.PRIMARY}]固定超时时间 (秒, 0=自动推导)[/{BrandColors.PRIMARY}]",
        default="0"
    )
    try:
        config["tts.timeout"] = max(0, int(timeout))
    except ValueError:
        config["tts.timeout"] = 0

    if full:
        console.print()
        timeout_min = Prompt.ask(
            f"[{BrandColors.PRIMARY}]最小超时时间 (秒)[/{BrandColors.PRIMARY}]",
            default="45"
        )
        try:
            config["tts.timeout_min"] = max(10, min(450, int(timeout_min)))
        except ValueError:
            config["tts.timeout_min"] = 45

        console.print()
        timeout_max = Prompt.ask(
            f"[{BrandColors.PRIMARY}]最大超时时间 (秒)[/{BrandColors.PRIMARY}]",
            default="900"
        )
        try:
            config["tts.timeout_max"] = max(60, min(3600, int(timeout_max)))
        except ValueError:
            config["tts.timeout_max"] = 900

    if full:
        console.print()
        _step_header(2, "文本分段配置", "S")

        enable_segmentation = Confirm.ask(
            f"\n[{BrandColors.PRIMARY}]启用文本分段?[/{BrandColors.PRIMARY}]",
            default=True
        )
        config["tts.enable_segmentation"] = enable_segmentation

        console.print()
        max_seg_len = Prompt.ask(
            f"[{BrandColors.PRIMARY}]最大分段长度 (字符数)[/{BrandColors.PRIMARY}]",
            default="2500"
        )
        try:
            config["tts.max_segment_length"] = max(100, int(max_seg_len))
        except ValueError:
            config["tts.max_segment_length"] = 2500

        console.print()
        min_seg_len = Prompt.ask(
            f"[{BrandColors.PRIMARY}]最小分段长度 (字符数)[/{BrandColors.PRIMARY}]",
            default="100"
        )
        try:
            config["tts.min_segment_length"] = max(10, min(1000, int(min_seg_len)))
        except ValueError:
            config["tts.min_segment_length"] = 100

        console.print()
        silence_ms = Prompt.ask(
            f"[{BrandColors.PRIMARY}]分段间静音时长 (毫秒)[/{BrandColors.PRIMARY}]",
            default="100"
        )
        try:
            config["tts.segment_silence_ms"] = max(0, min(1000, int(silence_ms)))
        except ValueError:
            config["tts.segment_silence_ms"] = 100

    console.print()
    _step_header(2 if not full else 3, "可靠性配置", "R")

    console.print()
    max_retries = Prompt.ask(
        f"[{BrandColors.PRIMARY}]TTS 最大重试次数 (0-20)[/{BrandColors.PRIMARY}]",
        default="3"
    )
    try:
        retries_value = max(0, min(20, int(max_retries)))
        config["reliability.tts_retry.max_retries"] = retries_value
        config["tts.retries"] = retries_value
    except ValueError:
        config["reliability.tts_retry.max_retries"] = 3
        config["tts.retries"] = 3

    console.print()
    base_delay = Prompt.ask(
        f"[{BrandColors.PRIMARY}]重试基础延迟 (秒)[/{BrandColors.PRIMARY}]",
        default="1.0"
    )
    try:
        config["reliability.tts_retry.base_delay"] = max(0.1, min(60.0, float(base_delay)))
    except ValueError:
        config["reliability.tts_retry.base_delay"] = 1.0

    if full:
        console.print()
        max_delay = Prompt.ask(
            f"[{BrandColors.PRIMARY}]重试最大延迟 (秒)[/{BrandColors.PRIMARY}]",
            default="60.0"
        )
        try:
            config["reliability.tts_retry.max_delay"] = max(1.0, min(300.0, float(max_delay)))
        except ValueError:
            config["reliability.tts_retry.max_delay"] = 60.0

        console.print()
        circuit_threshold = Prompt.ask(
            f"[{BrandColors.PRIMARY}]熔断器失败阈值[/{BrandColors.PRIMARY}]",
            default="5"
        )
        try:
            config["reliability.tts_circuit.failure_threshold"] = max(1, min(20, int(circuit_threshold)))
        except ValueError:
            config["reliability.tts_circuit.failure_threshold"] = 5

    if full:
        console.print()
        _step_header(4, "性能配置", "P")

        console.print()
        rate_limit = Prompt.ask(
            f"[{BrandColors.PRIMARY}]每秒请求数限制[/{BrandColors.PRIMARY}]",
            default="100"
        )
        try:
            config["tts.rate_limit"] = max(1, int(rate_limit))
        except ValueError:
            config["tts.rate_limit"] = 100

        console.print()
        ema_alpha = Prompt.ask(
            f"[{BrandColors.PRIMARY}]EMA 平滑因子 (0.0-1.0)[/{BrandColors.PRIMARY}]",
            default="0.3"
        )
        try:
            config["tts.ema_alpha"] = max(0.0, min(1.0, float(ema_alpha)))
        except ValueError:
            config["tts.ema_alpha"] = 0.3

        console.print()
        buffer_size = Prompt.ask(
            f"[{BrandColors.PRIMARY}]缓冲区大小[/{BrandColors.PRIMARY}]",
            default="32"
        )
        try:
            config["tts.buffer_size"] = max(1, int(buffer_size))
        except ValueError:
            config["tts.buffer_size"] = 32

    if full:
        console.print()
        _step_header(5, "文本正则化配置", "N")

        console.print()
        enable_text_norm = Confirm.ask(
            f"[{BrandColors.PRIMARY}]启用文本正则化?[/{BrandColors.PRIMARY}]",
            default=True
        )
        config["tts.text_normalization.enable_text_normalization"] = enable_text_norm

        if enable_text_norm:
            console.print()
            enable_whitespace = Confirm.ask(
                f"[{BrandColors.PRIMARY}]启用空白字符规范化?[/{BrandColors.PRIMARY}]",
                default=True
            )
            config["tts.text_normalization.enable_whitespace_normalization"] = enable_whitespace

            console.print()
            enable_linebreak = Confirm.ask(
                f"[{BrandColors.PRIMARY}]启用换行符规范化?[/{BrandColors.PRIMARY}]",
                default=True
            )
            config["tts.text_normalization.enable_linebreak_normalization"] = enable_linebreak

            console.print()
            enable_punct = Confirm.ask(
                f"[{BrandColors.PRIMARY}]启用标点符号规范化?[/{BrandColors.PRIMARY}]",
                default=True
            )
            config["tts.text_normalization.enable_punctuation_normalization"] = enable_punct

            console.print()
            enable_trim = Confirm.ask(
                f"[{BrandColors.PRIMARY}]启用行首尾空白去除?[/{BrandColors.PRIMARY}]",
                default=True
            )
            config["tts.text_normalization.enable_trim_whitespace"] = enable_trim

    if full:
        console.print()
        _step_header(6, "章节分割配置", "C")

        console.print()
        split_presets = [
            ("chinese_novel", "中文小说"),
            ("english_novel", "英文小说"),
            ("default", "默认"),
        ]
        console.print(f"[{BrandColors.INFO}]章节预设选项:[/{BrandColors.INFO}]")
        for i, (preset_id, preset_desc) in enumerate(split_presets, 1):
            console.print(f"  [bold cyan]{i}.[/bold cyan] {preset_desc} [dim]({preset_id})[/dim]")

        split_choice = Prompt.ask(
            f"\n[{BrandColors.PRIMARY}]请选择章节预设[/{BrandColors.PRIMARY}]",
            choices=[str(i) for i in range(1, len(split_presets) + 1)],
            default="1"
        )
        config["split.preset"] = split_presets[int(split_choice) - 1][0]

        console.print()
        min_chap_len = Prompt.ask(
            f"[{BrandColors.PRIMARY}]最小章节长度 (字符数)[/{BrandColors.PRIMARY}]",
            default="100"
        )
        try:
            config["split.min_chapter_length"] = max(10, int(min_chap_len))
        except ValueError:
            config["split.min_chapter_length"] = 100

        console.print()
        add_sep = Confirm.ask(
            f"[{BrandColors.PRIMARY}]在章节名后添加等于号分隔符?[/{BrandColors.PRIMARY}]",
            default=True
        )
        config["split.add_title_separator"] = add_sep

    console.print()
    _step_header(3 if not full else 7, "界面配置", "U")

    console.print()
    verbose = Confirm.ask(
        f"[{BrandColors.PRIMARY}]启用详细输出模式?[/{BrandColors.PRIMARY}]",
        default=False
    )
    config["ui.verbose"] = verbose

    if full:
        console.print()
        show_progress = Confirm.ask(
            f"[{BrandColors.PRIMARY}]显示进度条?[/{BrandColors.PRIMARY}]",
            default=True
        )
        config["ui.show_progress"] = show_progress

        console.print()
        show_timestamps = Confirm.ask(
            f"[{BrandColors.PRIMARY}]显示时间戳?[/{BrandColors.PRIMARY}]",
            default=False
        )
        config["ui.show_timestamps"] = show_timestamps

    if full:
        console.print()
        _step_header(8, "功能开关", "F")

        console.print()
        smart_detect = Confirm.ask(
            f"[{BrandColors.PRIMARY}]启用智能检测?[/{BrandColors.PRIMARY}]",
            default=True
        )
        config["features.smart_detection"] = smart_detect

        console.print()
        merge_chapters = Confirm.ask(
            f"[{BrandColors.PRIMARY}]启用合并短章节?[/{BrandColors.PRIMARY}]",
            default=True
        )
        config["features.merge_short_chapters"] = merge_chapters

        console.print()
        auto_retry = Confirm.ask(
            f"[{BrandColors.PRIMARY}]启用自动重试?[/{BrandColors.PRIMARY}]",
            default=True
        )
        config["features.auto_retry"] = auto_retry

    console.print()
    console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    console.print(f"[bold white]  {Icons.SUCCESS} 配置预览[/bold white]")
    console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    console.print(f"  [{BrandColors.TEXT_SECONDARY}]步骤 {total_steps}/{total_steps}[/{BrandColors.TEXT_SECONDARY}] [{BrandColors.SUCCESS}]{'█' * 30}[/{BrandColors.SUCCESS}] 100%")
    console.print(f"  [{BrandColors.TEXT_SECONDARY}]{'─' * 56}[/{BrandColors.TEXT_SECONDARY}]")

    preview_table = Table(show_header=False, box=SIMPLE, border_style=BrandColors.SECONDARY)
    preview_table.add_column("配置项", style="bold")
    preview_table.add_column("值", style="cyan")

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

    for key, value in config.items():
        label = config_labels.get(key, key)
        if isinstance(value, bool):
            value_str = "是" if value else "否"
        else:
            value_str = str(value)
        preview_table.add_row(label, value_str)

    console.print(preview_table)
    console.print()

    if Confirm.ask(f"[{BrandColors.SUCCESS}]确认保存配置?[/{BrandColors.SUCCESS}]", default=True):
        console.print(f"\n[{BrandColors.SUCCESS}]+ 配置已保存！[/{BrandColors.SUCCESS}]")
        return config
    else:
        console.print(f"\n[{BrandColors.WARNING}]! 配置已取消[/{BrandColors.WARNING}]")
        return None
