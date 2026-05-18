"""CLI UI mode management - Three output modes: simple, classic, debug."""

import sys
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn
from rich.logging import RichHandler

from ..config.schema import UIMode, UIConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ppc9 import __version__


class CLIUI:
    """CLI UI manager."""

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

        self.console = Console(no_color=self.config.no_color)
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging system."""
        log_file = Path(self.config.log_file) if self.config.log_file else None

        if self.config.mode == UIMode.DEBUG:
            handlers = [logging.StreamHandler(sys.stdout)]
            if log_file:
                handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=handlers
            )
        elif self.config.mode == UIMode.CLASSIC:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[logging.StreamHandler(sys.stdout)]
            )
        else:
            log_level = logging.DEBUG if self.config.verbose else logging.WARNING
            logging.basicConfig(
                level=log_level,
                format='%(message)s',
                handlers=[logging.StreamHandler(sys.stdout)]
            )

        self.logger = logging.getLogger("ppc9")

    def _format_time(self) -> str:
        """Format timestamp."""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _emoji(self, emoji: str) -> str:
        """Return emoji based on mode."""
        return emoji if self.config.use_emoji else ""

    def info(self, message: str, **kwargs):
        """Info message."""
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"[dim]{self._format_time()}[/dim] │ [bold blue]INFO[/bold blue] │ {message}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(message)
            if kwargs:
                for key, value in kwargs.items():
                    self.logger.debug(f"  {key}: {value}")
        else:
            self.console.print(f"[cyan]ℹ {message}[/cyan]")

    def success(self, message: str, **kwargs):
        """Success message."""
        emoji = self._emoji("✓")
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"[dim]{self._format_time()}[/dim] │ [bold green] OK [/bold green] │ {emoji} {message}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"SUCCESS: {message}")
            if kwargs:
                for key, value in kwargs.items():
                    self.logger.debug(f"  {key}: {value}")
        else:
            self.console.print(f"[green]{emoji} {message}[/green]")

    def warning(self, message: str, **kwargs):
        """Warning message."""
        emoji = self._emoji("⚠")
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"[dim]{self._format_time()}[/dim] │ [bold yellow]WARN[/bold yellow] │ {emoji} {message}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.warning(f"WARNING: {message}")
            if kwargs:
                for key, value in kwargs.items():
                    self.logger.debug(f"  {key}: {value}")
        else:
            self.console.print(f"[yellow]{emoji} {message}[/yellow]")

    def error(self, message: str, **kwargs):
        """Error message."""
        emoji = self._emoji("✗")
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"[dim]{self._format_time()}[/dim] │ [bold red]ERR [/bold red] │ {emoji} {message}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.error(f"ERROR: {message}")
            if kwargs:
                for key, value in kwargs.items():
                    self.logger.debug(f"  {key}: {value}")
        else:
            self.console.print(f"[red]{emoji} {message}[/red]")

    def debug(self, message: str, **kwargs):
        """Debug message (debug mode only)."""
        if self.config.mode == UIMode.DEBUG:
            self.logger.debug(f"DEBUG: {message}")
            if kwargs:
                for key, value in kwargs.items():
                    self.logger.debug(f"  {key}: {value}")

    def tts_start(self, input_path: str, output_path: str, voice: str, concurrency: int):
        """TTS conversion start."""
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"[dim]{self._format_time()}[/dim] │ [bold blue]INFO[/bold blue] │ 开始转换：{input_path} -> {output_path}")
            self.console.print(f"           │       │ 语音：{voice}, 并发：{concurrency}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"TTS_START: {input_path} -> {output_path}")
            self.logger.debug(f"  voice: {voice}")
            self.logger.debug(f"  concurrency: {concurrency}")
            self.logger.debug(f"  input_size: {Path(input_path).stat().st_size if Path(input_path).exists() else 'N/A'}")
        else:
            self.console.print(f"[cyan]🎤 开始 TTS 转换[/cyan]")
            self.console.print(f"  输入：{input_path}")
            self.console.print(f"  输出：{output_path}")

    def tts_processing(self, file_path: str, attempt: int = 1, max_attempts: int = 3, timeout: int = 0):
        """Processing file."""
        if self.config.mode == UIMode.CLASSIC:
            timeout_str = f", 超时 {timeout}s" if timeout > 0 else ""
            self.console.print(f"[dim]{self._format_time()}[/dim] │ [bold cyan]PROC[/bold cyan] │ 正在转换：{file_path} (尝试 {attempt}/{max_attempts}{timeout_str})")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"PROCESSING: {file_path} (attempt={attempt}/{max_attempts}, timeout={timeout})")
        else:
            emoji = self._emoji("⚡")
            self.console.print(f"[cyan]{emoji} 处理：{Path(file_path).name}[/cyan]")

    def tts_success(self, file_path: str, duration: float, size: int):
        """Processing success."""
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"[dim]{self._format_time()}[/dim] │ [bold green] OK [/bold green] │ ✓ 成功生成：{file_path} (用时 {duration:.2f}s, 大小 {size} 字节)")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"SUCCESS: {file_path} (duration={duration:.2f}s, size={size} bytes)")
        else:
            emoji = self._emoji("✅")
            self.console.print(f"[green]{emoji} {Path(file_path).name} ({duration:.2f}s, {size} 字节)[/green]")

    def tts_failure(self, file_path: str, error: str, attempt: int = 1, max_attempts: int = 3):
        """Processing failure."""
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"[dim]{self._format_time()}[/dim] │ [bold yellow]WARN[/bold yellow] │ ✗ 转换失败 ({attempt}/{max_attempts}): {file_path} │ 错误：{error}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.warning(f"FAILURE: {file_path} (attempt={attempt}/{max_attempts})")
            self.logger.error(f"  Error: {error}")
        else:
            emoji = self._emoji("❌")
            self.console.print(f"[red]{emoji} {Path(file_path).name}: {error}[/red]")

    def tts_complete(self, total: int, succeeded: int, failed: int, duration: float):
        """Batch conversion complete."""
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"[dim]{self._format_time()}[/dim] │ [bold green] OK [/bold green] │ 🎉 批量转换完成！成功：{succeeded}, 失败：{failed}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"COMPLETE: total={total}, succeeded={succeeded}, failed={failed}, duration={duration:.2f}s")
            self.logger.debug(f"  success_rate: {succeeded/total*100 if total > 0 else 0:.2f}%")
            self.logger.debug(f"  avg_speed: {total/duration if duration > 0 else 0:.2f} tasks/s")
        else:
            emoji = self._emoji("🎉")
            self.console.print(f"[green]{emoji} 转换完成！成功：{succeeded}/{total}, 失败：{failed}[/green]")

    def create_progress(self, total: int = 100) -> Progress:
        """Create progress bar."""
        if self.config.mode == UIMode.DEBUG:
            return Progress(
                TextColumn("[progress.description]{task.description}"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console
            )
        elif self.config.mode == UIMode.CLASSIC:
            return Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            )
        else:
            return Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=40),
                TextColumn("[green]{task.completed}/{task.total}"),
                TextColumn("[yellow]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            )

    def show_banner(self):
        """Show banner."""
        banner = """
    ██████╗ ██████╗  ██████╗ █████╗ 
    ██╔══██╗██╔══██╗██╔════╝██╔══██╗
    ██████╔╝██████╔╝██║     █████╔╝
    ██╔═══╝ ██╔═══╝ ██║     ██╔══██╗
    ██║     ██║     ██████╗ ███████╗
    ╚═╝     ╚═╝      ╚═════╝ ╚════╝ 
                                    ██╗   ██╗
                                    ╚██╗ ██╔╝
                                     ╚████╔╝ 
                                      ╚██╔╝  
                                       ╚═╝   
        """

        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"[bold blue]PPC9[/bold blue] - [dim]冰璃岩文本转语音工具[/dim] v{__version__}")
            self.console.print(f"[dim]{'━' * 60}[/dim]")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"PPC9 BANNER: Version {__version__}")
            self.logger.debug("Mode: DEBUG")
        else:
            gradient_colors = ["#4A90D9", "#3B8DD4", "#2C8ACF", "#1D87CA", "#2ECC71", "#27AE60"]
            logo_lines = banner.strip().split('\n')
            colored_lines = []
            for i, line in enumerate(logo_lines):
                color = gradient_colors[i % len(gradient_colors)]
                colored_lines.append(f"[{color}]{line}[/{color}]")
            colored_logo = "\n".join(colored_lines)
            self.console.print(Panel(
                f"{colored_logo}\n"
                f"[bold]冰璃岩 - 终极文本转语音工具[/bold]\n"
                f"[dim]{'─' * 40}[/dim]\n"
                f"版本：{__version__} | © 2026 BLY Team",
                border_style="cyan",
                expand=False
            ))

    def show_stats(self, stats: Dict[str, Any]):
        """Show statistics."""
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
        """Show error panel."""
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
            if error_type:
                self.logger.error(f"  Type: {error_type}")
            if suggestion:
                self.logger.error(f"  Suggestion: {suggestion}")
        else:
            content = [message]
            if error_type:
                content.append(f"\n[red]类型：{error_type}[/red]")
            if suggestion:
                content.append(f"\n[yellow]建议：{suggestion}[/yellow]")

            self.console.print(Panel(
                "\n".join(content),
                title=f"[bold red]❌ {title}[/bold red]",
                border_style="red"
            ))

    def show_success_panel(self, title: str, message: str, details: Optional[Dict] = None):
        """Show success panel."""
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
            self.logger.info(f"  Message: {message}")
            if details:
                for key, value in details.items():
                    self.logger.debug(f"  {key}: {value}")
        else:
            content = [message]
            if details:
                content.append("")
                for key, value in details.items():
                    content.append(f"[green]{key}:[/green] {value}")

            self.console.print(Panel(
                "\n".join(content),
                title=f"[bold green]✓ {title}[/bold green]",
                border_style="green"
            ))

    def print(self, *args, **kwargs):
        """Direct print."""
        self.console.print(*args, **kwargs)

    def log(self, level: str, message: str, **kwargs):
        """Generic log method."""
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


# Global UI instance

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
