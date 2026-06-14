"""CLI 主题与纯字符串 helper - PPC10.

Spec 10 之后, ``ui.py`` 仅承担两类职责:

1. **主题 / 颜色**: ``THEME`` + ``c(name, text)`` —— 业务代码颜色统一入口。
2. **纯字符串 helper**: ``format_duration`` / ``format_bytes`` / ``truncate``。

历史遗留的 ``CLIUI`` 类与 ``UIMode`` 仍保留(由
:mod:`src_m.cli.simple_progress` 使用),以保持向后兼容;新增代码请改用
:class:`src_m.cli.output.OutputFormatter`。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Any, Dict

from ..config.schema import UIMode, UIConfig  # noqa: F401 (向后兼容 re-export)

# 主题与 helper 收敛到 output,本模块只 re-export,避免循环。
from .output import THEME, c  # noqa: F401

# Path bootstrap (保持旧行为,即便模块自身不再使用 __version__ 也要 sys.path 就绪)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# 纯字符串 helper
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """把秒数格式化为 ``"1h 23m 45s"`` / ``"23m 45s"`` / ``"45s"``。"""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return str(seconds)
    if s < 0:
        s = 0.0
    total = int(s)
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {sec}s"
    if m > 0:
        return f"{m}m {sec}s"
    return f"{sec}s"


def format_bytes(n: float) -> str:
    """把字节数格式化为 ``"1.23 MB"``。"""
    try:
        v = float(n)
    except (TypeError, ValueError):
        return str(n)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024.0
        i += 1
    if i == 0:
        return f"{int(v)} {units[i]}"
    return f"{v:.2f} {units[i]}"


def truncate(text: str, max_len: int = 80) -> str:
    """超长字符串截断,末尾加 ``"…"``。"""
    if text is None:
        return ""
    s = str(text)
    if max_len <= 1 or len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


# ---------------------------------------------------------------------------
# 旧 CLIUI 类 —— 保留以兼容 simple_progress / 其它历史调用方
# ---------------------------------------------------------------------------


class CLIUI:
    """CLI UI manager (legacy).

    新代码请改用 :class:`src_m.cli.output.OutputFormatter`。
    本类仍可被 :mod:`src_m.cli.simple_progress` 使用以切换输出风格。
    """

    def __init__(self, config: Optional[UIConfig] = None):
        self.config = config or UIConfig()
        if self.config.mode == UIMode.SIMPLE:
            self.use_emoji = True
            self.show_timestamp = self.config.show_timestamps
            self.show_details = False
        elif self.config.mode == UIMode.CLASSIC:
            self.use_emoji = False
            self.show_timestamp = True
            self.show_details = False
        elif self.config.mode == UIMode.DEBUG:
            self.use_emoji = False
            self.show_timestamp = True
            self.show_details = True
        else:
            self.use_emoji = True
            self.show_timestamp = False
            self.show_details = False

        from rich.console import Console
        self.console = Console(no_color=self.config.no_color)
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging system (保留以兼容旧用法,内部实现简化)。"""
        log_file = Path(self.config.log_file) if self.config.log_file else None
        if self.config.mode == UIMode.DEBUG:
            handlers = [logging.StreamHandler(sys.stdout)]
            if log_file:
                handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                handlers=handlers,
            )
        elif self.config.mode == UIMode.CLASSIC:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                handlers=[logging.StreamHandler(sys.stdout)],
            )
        else:
            log_level = logging.DEBUG if self.config.verbose else logging.WARNING
            logging.basicConfig(
                level=log_level,
                format="%(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
            )
        self.logger = logging.getLogger("ppc10")

    def _format_time(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _emoji(self, emoji: str) -> str:
        return emoji if getattr(self.config, "use_emoji", True) else ""

    # 业务 print 方法全部保留为 no-op-ish 行为(旧代码可能仍依赖)。
    def info(self, message: str, **kwargs):
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(
                f"[dim]{self._format_time()}[/dim] │ [bold blue]INFO[/bold blue] │ {message}"
            )
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(message)
        else:
            self.console.print(f"[cyan]ℹ {message}[/cyan]")

    def success(self, message: str, **kwargs):
        emoji = self._emoji("✓")
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(
                f"[dim]{self._format_time()}[/dim] │ [bold green] OK [/bold green] │ {emoji} {message}"
            )
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"SUCCESS: {message}")
        else:
            self.console.print(f"[green]{emoji} {message}[/green]")

    def warning(self, message: str, **kwargs):
        emoji = self._emoji("⚠")
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(
                f"[dim]{self._format_time()}[/dim] │ [bold yellow]WARN[/bold yellow] │ {emoji} {message}"
            )
        elif self.config.mode == UIMode.DEBUG:
            self.logger.warning(f"WARNING: {message}")
        else:
            self.console.print(f"[yellow]{emoji} {message}[/yellow]")

    def error(self, message: str, **kwargs):
        emoji = self._emoji("✗")
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(
                f"[dim]{self._format_time()}[/dim] │ [bold red]ERR [/bold red] │ {emoji} {message}"
            )
        elif self.config.mode == UIMode.DEBUG:
            self.logger.error(f"ERROR: {message}")
        else:
            self.console.print(f"[red]{emoji} {message}[/red]")

    def debug(self, message: str, **kwargs):
        if self.config.mode == UIMode.DEBUG:
            self.logger.debug(f"DEBUG: {message}")

    def tts_start(self, input_path: str, output_path: str, voice: str, concurrency: int):
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(
                f"[dim]{self._format_time()}[/dim] │ [bold blue]INFO[/bold blue] │ 开始转换：{input_path} -> {output_path}"
            )
            self.console.print(
                f"           │       │ 语音：{voice}, 并发：{concurrency}"
            )
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"TTS_START: {input_path} -> {output_path}")
        else:
            self.console.print(f"[cyan]🎤 开始 TTS 转换[/cyan]")
            self.console.print(f"  输入：{input_path}")
            self.console.print(f"  输出：{output_path}")

    def tts_processing(self, file_path: str, attempt: int = 1, max_attempts: int = 3, timeout: int = 0):
        if self.config.mode == UIMode.CLASSIC:
            timeout_str = f", 超时 {timeout}s" if timeout > 0 else ""
            self.console.print(
                f"[dim]{self._format_time()}[/dim] │ [bold cyan]PROC[/bold cyan] │ 正在转换：{file_path} (尝试 {attempt}/{max_attempts}{timeout_str})"
            )
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(
                f"PROCESSING: {file_path} (attempt={attempt}/{max_attempts}, timeout={timeout})"
            )
        else:
            emoji = self._emoji("⚡")
            self.console.print(f"[cyan]{emoji} 处理：{Path(file_path).name}[/cyan]")

    def tts_success(self, file_path: str, duration: float, size: int):
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(
                f"[dim]{self._format_time()}[/dim] │ [bold green] OK [/bold green] │ ✓ 成功生成：{file_path} (用时 {duration:.2f}s, 大小 {size} 字节)"
            )
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"SUCCESS: {file_path} (duration={duration:.2f}s, size={size} bytes)")
        else:
            emoji = self._emoji("✅")
            self.console.print(
                f"[green]{emoji} {Path(file_path).name} ({duration:.2f}s, {size} 字节)[/green]"
            )

    def tts_failure(self, file_path: str, error: str, attempt: int = 1, max_attempts: int = 3):
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(
                f"[dim]{self._format_time()}[/dim] │ [bold yellow]WARN[/bold yellow] │ ✗ 转换失败 ({attempt}/{max_attempts}): {file_path} │ 错误：{error}"
            )
        elif self.config.mode == UIMode.DEBUG:
            self.logger.warning(f"FAILURE: {file_path} (attempt={attempt}/{max_attempts})")
            self.logger.error(f"  Error: {error}")
        else:
            emoji = self._emoji("❌")
            self.console.print(f"[red]{emoji} {Path(file_path).name}: {error}[/red]")

    def tts_complete(self, total: int, succeeded: int, failed: int, duration: float):
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(
                f"[dim]{self._format_time()}[/dim] │ [bold green] OK [/bold green] │ 🎉 批量转换完成！成功：{succeeded}, 失败：{failed}"
            )
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(
                f"COMPLETE: total={total}, succeeded={succeeded}, failed={failed}, duration={duration:.2f}s"
            )
        else:
            emoji = self._emoji("🎉")
            self.console.print(
                f"[green]{emoji} 转换完成！成功：{succeeded}/{total}, 失败：{failed}[/green]"
            )

    def create_progress(self, total: int = 100):
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            TimeElapsedColumn,
        )

        if self.config.mode == UIMode.DEBUG:
            return Progress(
                TextColumn("[progress.description]{task.description}"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console,
            )
        elif self.config.mode == UIMode.CLASSIC:
            return Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console,
            )
        else:
            return Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=40),
                TextColumn("[green]{task.completed}/{task.total}"),
                TextColumn("[yellow]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console,
            )

    def show_banner(self):
        from rich.panel import Panel
        from ppc10 import __version__
        banner = (
            "    ██████╗ ██████╗  ██████╗     ██╗ ██████╗ \n"
            "    ██╔══██╗██╔══██╗██╔════╝    ███║██╔═████╗\n"
            "    ██████╔╝██████╔╝██║         ╚██║██║██╔██║\n"
            "    ██╔═══╝ ██╔═══╝ ██║          ██║████╔╝██║\n"
            "    ██║     ██║     ╚██████╗     ██║╚██████╔╝\n"
            "    ╚═╝     ╚═╝      ╚═════╝     ╚═╝ ╚═════╝ "
        )

        if self.config.mode == UIMode.CLASSIC:
            self.console.print(
                f"[bold blue]PPC10[/bold blue] - [dim]冰璃岩文本转语音工具[/dim] v{__version__}"
            )
            self.console.print(f"[dim]{'━' * 60}[/dim]")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"PPC10 BANNER: Version {__version__}")
        else:
            gradient_colors = ["#4A90D9", "#3B8DD4", "#2C8ACF", "#1D87CA", "#2ECC71", "#27AE60"]
            logo_lines = banner.strip().split("\n")
            colored_lines = []
            for i, line in enumerate(logo_lines):
                color = gradient_colors[i % len(gradient_colors)]
                colored_lines.append(f"[{color}]{line}[/{color}]")
            colored_logo = "\n".join(colored_lines)
            self.console.print(
                Panel(
                    f"{colored_logo}\n"
                    f"[bold]冰璃岩 - 终极文本转语音工具[/bold]\n"
                    f"[dim]{'─' * 40}[/dim]\n"
                    f"版本：{__version__} | © 2026 BLY Team",
                    border_style="cyan",
                    expand=False,
                )
            )

    def show_stats(self, stats: Dict[str, Any]):
        from rich.table import Table
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"\n[bold blue]统计信息:[/bold blue]")
            self.console.print(f"[dim]{'─' * 40}[/dim]")
            for key, value in stats.items():
                self.console.print(f"  [bold]{key}:[/bold] {value}")
            self.console.print(f"[dim]{'─' * 40}[/dim]")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info("STATS:")
            for key, value in stats.items():
                self.logger.debug(f"  {key}: {value}")
        else:
            table = Table(title="📊 统计信息")
            table.add_column("指标", style="cyan")
            table.add_column("数值", style="green")
            for key, value in stats.items():
                table.add_row(key, str(value))
            self.console.print(table)

    def show_error_panel(self, title: str, message: str, error_type: str = "", suggestion: str = ""):
        from rich.panel import Panel
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"\n[bold red]错误：{title}[/bold red]")
            self.console.print(f"[dim]{'─' * 40}[/dim]")
            self.console.print(f"  {message}")
            if error_type:
                self.console.print(f"  [dim]类型：{error_type}[/dim]")
            if suggestion:
                self.console.print(f"  [yellow]建议：{suggestion}[/yellow]")
            self.console.print(f"[dim]{'─' * 40}[/dim]")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.error(f"ERROR_PANEL: {title}")
            self.logger.error(f"  Message: {message}")
        else:
            content = [message]
            if error_type:
                content.append(f"\n[red]类型：{error_type}[/red]")
            if suggestion:
                content.append(f"\n[yellow]建议：{suggestion}[/yellow]")
            self.console.print(
                Panel(
                    "\n".join(content),
                    title=f"[bold red]❌ {title}[/bold red]",
                    border_style="red",
                )
            )

    def show_success_panel(self, title: str, message: str, details: Optional[Dict] = None):
        from rich.panel import Panel
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"\n[bold green]成功：{title}[/bold green]")
            self.console.print(f"[dim]{'─' * 40}[/dim]")
            self.console.print(f"  {message}")
            if details:
                for key, value in details.items():
                    self.console.print(f"  [bold]{key}:[/bold] {value}")
            self.console.print(f"[dim]{'─' * 40}[/dim]")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"SUCCESS_PANEL: {title}")
        else:
            content = [message]
            if details:
                content.append("")
                for key, value in details.items():
                    content.append(f"[green]{key}:[/green] {value}")
            self.console.print(
                Panel(
                    "\n".join(content),
                    title=f"[bold green]✓ {title}[/bold green]",
                    border_style="green",
                )
            )

    def print(self, *args, **kwargs):
        self.console.print(*args, **kwargs)

    def log(self, level: str, message: str, **kwargs):
        if level.upper() == "DEBUG":
            self.debug(message, **kwargs)
        elif level.upper() == "INFO":
            self.info(message, **kwargs)
        elif level.upper() == "WARNING":
            self.warning(message, **kwargs)
        elif level.upper() == "ERROR":
            self.error(message, **kwargs)
        elif level.upper() == "SUCCESS":
            self.success(message, **kwargs)


# 全局 UI 实例(向后兼容)
_default_ui: Optional[CLIUI] = None


def get_ui(config: Optional[UIConfig] = None) -> CLIUI:
    """Get global UI instance."""
    global _default_ui
    if _default_ui is None:
        _default_ui = CLIUI(config)
    return _default_ui


def set_ui(config: UIConfig):
    """Set UI config."""
    global _default_ui
    _default_ui = CLIUI(config)
    return _default_ui


def set_ui_mode(mode: str, **kwargs):
    """Set UI mode (convenience method)."""
    global _default_ui
    ui_mode = UIMode(mode.lower())
    config = UIConfig(mode=ui_mode, **kwargs)
    _default_ui = CLIUI(config)
    return _default_ui


__all__ = [
    "THEME",
    "c",
    "format_duration",
    "format_bytes",
    "truncate",
    "CLIUI",
    "UIMode",
    "UIConfig",
    "get_ui",
    "set_ui",
    "set_ui_mode",
]
