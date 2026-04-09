"""输出格式化 - 冰璃岩开发组 (BLY Team)
使用 Rich 库实现简洁、友好的输出
"""

import sys
import os
from typing import Optional, Any, Callable, Dict, List
from datetime import datetime
from dataclasses import dataclass

# 从主模块导入版本号，确保版本号统一管理
# 使用 sys.path 导入以避免相对导入问题
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ppc8 import __version__

# Windows 终端兼容：强制使用 UTF-8 编码
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # 重新配置标准输出为 UTF-8
    import io
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
import logging

# Windows 终端兼容：强制使用 UTF-8 编码
console = Console(file=sys.stdout, legacy_windows=False)


class BrandAssets:
    """品牌资源 - 冰璃岩开发组 (BLY Team)"""
    LOGO_ASCII = """
██████╗ ██████╗  ██████╗ █████╗ 
██╔══██╗██╔══██╗██╔════╝██╔══██╗
██████╔╝██████╔╝██║     ╚█████╔╝
██╔═══╝ ██╔═══╝ ██║     ██╔══██╗
██║     ██║     ╚██████╗╚█████╔╝
╚═╝     ╚═╝      ╚═════╝ ╚════╝ 
                                """
    # 版本号从主模块导入，统一管理
    VERSION = __version__
    TAGLINE = "冰璃岩 - 终极文本转语音工具"
    COPYRIGHT = "© 2026 BLY Team. All rights reserved."


class BrandColors:
    """品牌色彩 - 冰璃岩开发组 (BLY Team)"""
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


class ProgressStyles:
    """进度条样式 - 冰璃岩开发组 (BLY Team)"""
    DEFAULT = "cyan"
    SUCCESS = "green"
    WARNING = "yellow"
    ERROR = "red"
    PULSE = "magenta"
    
    BAR_STYLES = {
        "default": "█▓▒░ ",
        "smooth": "━━━╸ ",
        "blocks": "█▉▊▋▌▍▎▏ ",
    }
    
    ANIMATION_PATTERNS = {
        "pulse": ["◐", "◓", "◑", "◒"],
        "wave": ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂"],
        "bounce": ["⠁", "⠃", "⠇", "⡇", "⡏", "⡟", "⡿", "⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"],
    }


class Icons:
    """图标定义 - 冰璃岩开发组 (BLY Team)"""
    SUCCESS = "+"
    ERROR = "-"
    WARNING = "!"
    INFO = "i"
    ROCKET = "[ROCKET]"
    GEAR = "⚙"
    MICROPHONE = "[MIC]"
    SOUND = "[SOUND]"
    FILE = "[FILE]"
    FOLDER = "[FOLDER]"
    CHART = "[CHART]"
    CLOCK = "[TIME]"
    STAR = "[STAR]"
    BOOK = "[BOOK]"
    LINK = "[LINK]"


class StatusIcons:
    """状态图标 - 冰璃岩开发组 (BLY Team)"""
    RUNNING = ("◐", "cyan")
    COMPLETED = ("+", "green")
    FAILED = ("-", "red")
    PENDING = ("o", "dim")
    WARNING = ("!", "yellow")
    SKIPPED = ("->", "dim")
    INFO = ("i", "blue")
    SUCCESS = ("+", "green")


class OutputStyle:
    """输出样式 - 冰璃岩开发组 (BLY Team)"""

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
    """错误修复建议 - 冰璃岩开发组 (BLY Team)"""
    
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
        """获取错误修复建议 - 冰璃岩开发组 (BLY Team)
        
        参数:
            error_type: 错误类型名称
        返回:
            修复建议列表
        """
        return cls.SUGGESTIONS.get(error_type, cls.SUGGESTIONS["default"])
    
    @classmethod
    def add_suggestion(cls, error_type: str, suggestions: List[str]):
        """添加自定义错误建议 - 冰璃岩开发组 (BLY Team)
        
        参数:
            error_type: 错误类型名称
            suggestions: 建议列表
        """
        if error_type in cls.SUGGESTIONS:
            cls.SUGGESTIONS[error_type].extend(suggestions)
        else:
            cls.SUGGESTIONS[error_type] = suggestions


@dataclass
class TaskStatus:
    """任务状态 - 冰璃岩开发组 (BLY Team)"""
    name: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    retries: int = 0
    error: Optional[str] = None


@dataclass
class RetryInfo:
    """重试信息 - 冰璃岩开发组 (BLY Team)"""
    attempt: int
    max_attempts: int
    delay: float
    error: str
    will_retry: bool


class OutputFormatter:
    """输出格式化器 - 冰璃岩开发组 (BLY Team)"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.console = Console()
        self._progress: Optional[Progress] = None
        self._live: Optional[Live] = None
        self._task_statuses: dict = {}

    def set_verbose(self, verbose: bool):
        """设置详细模式"""
        self.verbose = verbose

    def info(self, message: str, **kwargs):
        """输出信息"""
        if self.verbose:
            self._log("INFO", message)
        else:
            console.print(message, style=OutputStyle.INFO)

    def success(self, message: str):
        """输出成功"""
        console.print(f"+ {message}", style=OutputStyle.SUCCESS)

    def error(self, message: str):
        """输出错误"""
        console.print(f"- {message}", style=OutputStyle.ERROR)

    def warning(self, message: str):
        """输出警告"""
        console.print(f"! {message}", style=OutputStyle.WARNING)

    def debug(self, message: str):
        """输出调试信息"""
        if self.verbose:
            self._log("DEBUG", message)

    def _log(self, level: str, message: str):
        """带时间戳的日志输出"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"{timestamp} | {level:7s} | {message}")

    def title(self, text: str):
        """输出标题"""
        console.print(f"\n{text}", style=OutputStyle.TITLE)
        console.print("=" * len(text))

    def panel(
        self,
        content: str,
        title: str = None,
        style: str = "blue"
    ):
        """输出面板 - 冰璃岩开发组 (BLY Team)"""
        panel = Panel(
            content,
            title=title,
            style=style,
            expand=False
        )
        console.print(panel)
    
    def error_panel(
        self,
        message: str,
        title: str = "错误",
        error_type: str = None,
        suggestion: str = None,
        details: str = None
    ) -> None:
        """美化的错误面板 - 冰璃岩开发组 (BLY Team)
        
        参数:
            message: 错误消息
            title: 面板标题
            error_type: 错误类型
            suggestion: 修复建议
            details: 详细信息
        """
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
        """美化的成功面板 - 冰璃岩开发组 (BLY Team)
        
        参数:
            message: 成功消息
            title: 面板标题
            details: 详细信息字典
        """
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
        """美化的警告面板 - 冰璃岩开发组 (BLY Team)
        
        参数:
            message: 警告消息
            title: 面板标题
            suggestion: 修复建议
        """
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
        """可折叠的堆栈追踪 - 冰璃岩开发组 (BLY Team)
        
        参数:
            exception: 异常对象
            expanded: 是否默认展开
            max_lines: 最大显示行数
        """
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
        """输出表格 - 冰璃岩开发组 (BLY Team)"""
        table = Table(title=title)

        for col in columns:
            table.add_column(col["header"], **col.get("options", {}))

        for row in rows:
            table.add_row(*row)

        console.print(table)

    def progress_start(self, total: int, description: str = "Processing"):
        """开始进度条 - 冰璃岩开发组 (BLY Team)"""
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
        """更新进度"""
        if self._progress:
            self._progress.advance(task_id, advance)

    def progress_stop(self):
        """停止进度条"""
        if self._progress:
            self._progress.stop()
            self._progress = None

    def progress_update_description(self, task_id: int, description: str):
        """更新进度描述"""
        if self._progress:
            self._progress.update(task_id, description=description)

    def compact_progress(
        self,
        current: int,
        total: int,
        filename: str,
        suffix: str = ""
    ):
        """简洁进度显示"""
        percent = current / total if total > 0 else 0
        bar_length = 20
        filled = int(bar_length * percent)
        bar = "█" * filled + "░" * (bar_length - filled)

        status = f"[{current}/{total}] {bar} {percent:.0%}"
        if suffix:
            status += f" {suffix}"

        console.print(f"\r{status} {filename}", end="", flush=True)

    def final_progress(self, current: int, total: int, duration: float):
        """最终进度显示"""
        percent = current / total if total > 0 else 0
        bar_length = 20
        filled = int(bar_length * percent)
        bar = "█" * filled + "░" * (bar_length - filled)

        console.print(f"\r[{current}/{total}] {bar} {percent:.0%} 完成: {current}/{total} 用时: {duration:.1f}s")

    def stat(self, key: str, value: Any):
        """输出统计"""
        console.print(f"  {key}: {value}")

    def stats(self, stats: dict, title: str = "统计"):
        """输出统计信息"""
        self.panel("\n".join(f"{k}: {v}" for k, v in stats.items()), title)

    def config_show(self, config: dict):
        """显示配置"""
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
        """显示命令帮助 - 冰璃岩开发组 (BLY Team)"""
        content = f"[bold]描述[/bold]\n{description}\n\n"
        content += f"[bold]用法[/bold]\n{usage}\n\n"
        if examples:
            content += f"[bold]示例[/bold]\n"
            for example in examples:
                content += f"  {example}\n"

        self.panel(content, f"ppc8 {command}", "green")

    def check_result(self, checks: list):
        """显示检查结果"""
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
        """显示重试状态"""
        if info.will_retry:
            msg = f"R 第 {info.attempt}/{info.max_attempts} 次尝试失败：{info.error}，{info.delay:.1f}s 后重试"
            console.print(msg, style=OutputStyle.RETRY)
        else:
            msg = f"- 已重试 {info.attempt} 次，最终失败：{info.error}"
            console.print(msg, style=OutputStyle.ERROR)

    def task_status(self, task: TaskStatus):
        """显示单个任务状态"""
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

    def batch_summary(
        self,
        total: int,
        succeeded: int,
        failed: int,
        duration: float,
        retries: int = 0
    ):
        """显示批量处理汇总"""
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

    def create_advanced_progress(
        self,
        description: str = "处理中",
        show_speed: bool = True
    ) -> Progress:
        """创建高级进度条"""
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
        """开始实时状态显示"""
        self._live = Live(console=console, refresh_per_second=4)
        self._live.start()
        return self._live

    def live_status_stop(self):
        """停止实时状态显示"""
        if self._live:
            self._live.stop()
            self._live = None

    def update_task_status(self, task_id: str, status: TaskStatus):
        """更新任务状态（用于实时显示）"""
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
        """显示品牌横幅"""
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
        """显示欢迎信息"""
        welcome_content = f"""[{BrandColors.PRIMARY}]欢迎使用 PPC8![/{BrandColors.PRIMARY}]

[{BrandColors.TEXT_SECONDARY}]冰璃岩项目开发组 - 一个功能强大的文本转语音工具，支持多种语音引擎和批量处理。[/{BrandColors.TEXT_SECONDARY}]"""
        self.console.print(Panel(
            welcome_content,
            title=f"[{BrandColors.ACCENT}][ROCKET] 快速开始 [/{BrandColors.ACCENT}]",
            border_style=BrandColors.PRIMARY,
            expand=False
        ))
        self.console.print()
        commands_table = Table(show_header=False, box=None, padding=(0, 2))
        commands_table.add_column("命令", style=f"bold {BrandColors.PRIMARY}")
        commands_table.add_column("说明", style=BrandColors.TEXT_SECONDARY)
        commands_table.add_row(f"{Icons.SOUND} ppc8 convert", "转换文本为语音")
        commands_table.add_row(f"{Icons.FOLDER} ppc8 batch", "批量处理文件")
        commands_table.add_row(f"{Icons.GEAR} ppc8 config", "配置设置")
        commands_table.add_row(f"{Icons.INFO} ppc8 --help", "查看帮助信息")
        self.console.print(Panel(
            commands_table,
            title=f"[{BrandColors.SECONDARY}][BOOK] 常用命令 [/{BrandColors.SECONDARY}]",
            border_style=BrandColors.SECONDARY,
            expand=False
        ))
        self.console.print()

    def interactive_help(self, commands: Dict[str, Dict]):
        """交互式帮助浏览器 - 冰璃岩开发组 (BLY Team)
        
        参数:
            commands: 命令字典
        """
        self.title("[BOOK] PPC8 交互式帮助浏览器")
        
        command_list = list(commands.keys())
        current_idx = 0
        search_filter = ""
        
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
            console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
            console.print("[bold white]  [BOOK] PPC8 交互式帮助浏览器[/bold white]")
            console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
            
            if search_filter:
                console.print(f"\n[dim]搜索: [bold yellow]{search_filter}[/bold yellow][/dim]")
            
            console.print("\n[bold]可用命令:[/bold]")
            for i, cmd in enumerate(filtered_commands):
                if i == current_idx:
                    console.print(f"  [bold green]▶ {cmd}[/bold green] - {commands[cmd].get('desc', '无描述')}")
                else:
                    console.print(f"    [dim]{cmd}[/dim] - {commands[cmd].get('desc', '无描述')}")
            
            console.print("\n[bold cyan]───────────────────────────────────────────────────────────────[/bold cyan]")
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
        """显示命令详细信息 - 冰璃岩开发组 (BLY Team)"""
        console.clear()
        console.print(f"\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        console.print(f"[bold white]  命令详情: [bold green]{command}[/bold green][/bold white]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")
        
        console.print(f"[bold]T 描述:[/bold]")
        console.print(f"  {info.get('desc', '无描述')}\n")
        
        console.print(f"[bold]T 用法:[/bold]")
        console.print(f"  [cyan]{info.get('usage', '无用法说明')}[/cyan]\n")
        
        examples = info.get('examples', [])
        if examples:
            console.print(f"[bold]i 示例:[/bold]")
            for ex in examples:
                if isinstance(ex, dict):
                    console.print(f"  [green]•[/green] [dim]{ex.get('desc', '')}[/dim]")
                    console.print(f"    [bold]$ {ex.get('cmd', '')}[/bold]")
                else:
                    console.print(f"  [green]•[/green] {ex}")
            console.print()
        
        options = info.get('options', [])
        if options:
            console.print(f"[bold]G 选项:[/bold]")
            for opt in options:
                if isinstance(opt, dict):
                    opt_name = opt.get('name', '')
                    opt_desc = opt.get('desc', '')
                    console.print(f"  [yellow]{opt_name}[/yellow] - {opt_desc}")
            console.print()
        
        see_also = info.get('see_also', [])
        if see_also:
            console.print(f"[bold]L 相关命令:[/bold]")
            console.print(f"  {', '.join(see_also)}\n")
        
        console.print("[bold cyan]───────────────────────────────────────────────────────────────[/bold cyan]")
        Prompt.ask("\n[dim]按 Enter 返回[/dim]")

    def command_examples(self, command: str, examples: List[Dict]):
        """显示命令示例 - 冰璃岩开发组 (BLY Team)
        
        参数:
            command: 命令名称
            examples: 示例列表
        """
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
        """显示快捷键提示"""
        console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        console.print("[bold white]  K  PPC8 快捷键参考[/bold white]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")
        
        global_shortcuts = [
            ("Ctrl + C", "中断当前操作"),
            ("Ctrl + D", "退出程序"),
            ("Tab", "自动补全命令"),
            ("↑ / ↓", "浏览历史命令"),
            ("Ctrl + L", "清屏"),
            ("Ctrl + R", "搜索历史命令"),
        ]
        
        command_shortcuts = [
            ("ppc8 --help", "显示帮助信息"),
            ("ppc8 --version", "显示版本号"),
            ("ppc8 -v", "详细输出模式"),
            ("ppc8 convert -h", "convert 命令帮助"),
            ("ppc8 config show", "显示当前配置"),
        ]
        
        interactive_shortcuts = [
            ("↑ / ↓ / j / k", "上下导航"),
            ("Enter", "选择/确认"),
            ("/ 或 s", "搜索过滤"),
            ("q", "退出/返回"),
            ("数字键", "快速选择"),
        ]
        
        table = Table(show_header=True, box=ROUNDED, border_style="cyan")
        table.add_column("快捷键", style="bold yellow", width=20)
        table.add_column("功能", style="white", width=40)
        
        console.print("[bold]L 全局快捷键:[/bold]")
        for shortcut, desc in global_shortcuts:
            table.add_row(shortcut, desc)
        console.print(table)
        console.print()
        
        table2 = Table(show_header=True, box=ROUNDED, border_style="green")
        table2.add_column("命令", style="bold green", width=20)
        table2.add_column("功能", style="white", width=40)
        
        console.print("[bold]P 命令快捷方式:[/bold]")
        for cmd, desc in command_shortcuts:
            table2.add_row(cmd, desc)
        console.print(table2)
        console.print()
        
        table3 = Table(show_header=True, box=ROUNDED, border_style="magenta")
        table3.add_column("快捷键", style="bold magenta", width=20)
        table3.add_column("功能", style="white", width=40)
        
        console.print("[bold]K 交互模式快捷键:[/bold]")
        for shortcut, desc in interactive_shortcuts:
            table3.add_row(shortcut, desc)
        console.print(table3)
        
        console.print("\n[bold cyan]───────────────────────────────────────────────────────────────[/bold cyan]")

    def help_command_enhanced(
        self, 
        command: str, 
        description: str, 
        usage: str, 
        examples: List[Dict],
        options: List[Dict] = None,
        see_also: List[str] = None
    ):
        """增强的命令帮助显示 - 冰璃岩开发组 (BLY Team)
        
        参数:
            command: 命令名称
            description: 详细描述
            usage: 用法说明
            examples: 示例列表
            options: 选项列表
            see_also: 相关命令
        """
        console.print(f"\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        console.print(f"[bold white]  [BOOK] ppc8 {command}[/bold white]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")
        
        desc_panel = Panel(
            description,
            title="[bold yellow]T 描述[/bold yellow]",
            border_style="yellow",
            box=SIMPLE,
            padding=(0, 1)
        )
        console.print(desc_panel)
        
        usage_panel = Panel(
            f"[bold cyan]{usage}[/bold cyan]",
            title="[bold green]T 用法[/bold green]",
            border_style="green",
            box=SIMPLE,
            padding=(0, 1)
        )
        console.print(usage_panel)
        
        if options:
            console.print(f"\n[bold magenta]G 选项:[/bold magenta]")
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
            console.print(f"\n[bold blue]i 示例:[/bold blue]")
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
            console.print(f"\n[bold cyan]L 相关命令:[/bold cyan]")
            see_also_str = "  ".join(f"[bold green]{cmd}[/bold green]" for cmd in see_also)
            console.print(f"  {see_also_str}")
        
        console.print("\n[bold cyan]───────────────────────────────────────────────────────────────[/bold cyan]")

    def help_index(self, categories: Dict[str, List[str]]):
        """显示帮助索引 - 冰璃岩开发组 (BLY Team)
        
        参数:
            categories: 分类字典
        """
        console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        console.print("[bold white]  [BOOK] PPC8 命令索引[/bold white]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")
        
        category_icons = {
            "转换命令": "R",
            "配置命令": "G",
            "工具命令": "T",
            "信息命令": "i",
            "默认": "[FOLDER]"
        }
        
        for category, commands_list in categories.items():
            icon = category_icons.get(category, category_icons["默认"])
            console.print(f"[bold]{icon} {category}:[/bold]")
            
            for cmd in commands_list:
                console.print(f"  [green]•[/green] [bold cyan]{cmd}[/bold cyan]")
            console.print()
        
        console.print("[bold cyan]───────────────────────────────────────────────────────────────[/bold cyan]")
        console.print("[dim]使用 [bold]ppc8 <command> --help[/bold] 查看命令详细帮助[/dim]")
        console.print("[dim]使用 [bold]ppc8 help[/bold] 进入交互式帮助浏览器[/dim]")

    def enhanced_stats_panel(
        self,
        stats: Dict[str, Any],
        title: str = "详细统计"
    ) -> Panel:
        """创建增强统计面板 - 冰璃岩开发组 (BLY Team)
        
        参数:
            stats: 详细统计数据
            title: 面板标题
        返回:
            Panel 对象
        """
        lines = []
        
        if "total" in stats:
            total = stats["total"]
            completed = stats.get("completed", 0)
            failed = stats.get("failed", 0)
            pending = stats.get("pending", total - completed - failed)
            
            progress_bar = self._create_mini_progress_bar(completed + failed, total)
            lines.append(f"[bold]任务统计[/bold]")
            lines.append(f"  总数：{total}")
            lines.append(f"  [{BrandColors.SUCCESS}]+ 完成:[/{BrandColors.SUCCESS}] {completed}")
            lines.append(f"  [{BrandColors.ERROR}]- 失败:[/{BrandColors.ERROR}] {failed}")
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
        """显示完成报告 - 冰璃岩开发组 (BLY Team)
        
        参数:
            stats: 转换统计数据
            executor_stats: 执行器统计数据
            title: 报告标题
        """
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
        summary_table.add_row(f"[{BrandColors.SUCCESS}]成功[{BrandColors.SUCCESS}]", f"[{BrandColors.SUCCESS}]{completed}[/{BrandColors.SUCCESS}]")
        summary_table.add_row(f"[{BrandColors.ERROR}]失败[{BrandColors.ERROR}]", f"[{BrandColors.ERROR}]{failed}[/{BrandColors.ERROR}]")
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
        """创建实时统计面板 - 冰璃岩开发组 (BLY Team)
        
        参数:
            stats: 统计数据
            title: 面板标题
        返回:
            Panel 对象
        """
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
        """创建迷你进度条"""
        if total == 0:
            percent = 0
        else:
            percent = current / total
        
        filled = int(width * percent)
        bar = f"[{BrandColors.SUCCESS}]{'█' * filled}[/{BrandColors.SUCCESS}]{('░' * (width - filled))}"
        return f"[{bar}] {percent:.0%}"
    
    def _format_duration(self, seconds: float) -> str:
        """格式化持续时间"""
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
    
    def animated_progress(
        self,
        total: int,
        description: str = "处理中",
        animation_style: str = "pulse"
    ) -> Progress:
        """创建带动画效果的进度条
        
        参数:
            total: 总任务数
            description: 描述文本
            animation_style: 动画样式
            
        返回:
            Progress 对象
        """
        pattern = ProgressStyles.ANIMATION_PATTERNS.get(animation_style, ProgressStyles.ANIMATION_PATTERNS["pulse"])
        
        class AnimatedSpinnerColumn(SpinnerColumn):
            """自定义动画列"""
            def __init__(self, pattern: List[str], *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._pattern = pattern
                self._frame = 0
            
            def render(self, task):
                if task.finished:
                    return Text("+", style=BrandColors.SUCCESS)
                frame = self._pattern[self._frame % len(self._pattern)]
                self._frame += 1
                return Text(frame, style=BrandColors.ACCENT)
        
        progress = Progress(
            AnimatedSpinnerColumn(pattern),
            TextColumn(f"[{BrandColors.PRIMARY}]{{task.description}}[/{BrandColors.PRIMARY}]"),
            BarColumn(
                complete_style=BrandColors.SUCCESS,
                finished_style=BrandColors.SUCCESS,
                pulse_style=BrandColors.ACCENT
            ),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console
        )
        
        progress.add_task(description, total=total)
        return progress
    
    def dashboard(
        self,
        title: str,
        sections: Dict[str, Dict[str, Any]],
        refresh: bool = False
    ) -> None:
        """显示仪表板 - 冰璃岩开发组 (BLY Team)
        
        参数:
            title: 仪表板标题
            sections: 各区域数据
            refresh: 是否刷新显示
        """
        if refresh:
            self.console.print("\033[H\033[J", end="")
        
        self.console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
        self.console.print(f"[bold white]  {Icons.CHART} {title}[/bold white]")
        self.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n")
        
        panels = []
        
        for section_name, section_data in sections.items():
            panel_content = []
            
            if section_name == "进度":
                current = section_data.get("current", 0)
                total = section_data.get("total", 100)
                status = section_data.get("status", "pending")
                
                percent = current / total if total > 0 else 0
                bar = self._create_mini_progress_bar(current, total, 30)
                panel_content.append(bar)
                
                status_icons = {
                    "pending": "[dim]o[/{BrandColors.TEXT_SECONDARY}]",
                    "running": f"[{BrandColors.INFO}]◐ 运行中[/{BrandColors.INFO}]",
                    "completed": f"[{BrandColors.SUCCESS}]+ 已完成[/{BrandColors.SUCCESS}]",
                    "failed": f"[{BrandColors.ERROR}]- 失败[/{BrandColors.ERROR}]",
                }
                panel_content.append(status_icons.get(status, "未知状态"))
            
            elif section_name == "统计":
                success = section_data.get("success", 0)
                failed = section_data.get("failed", 0)
                speed = section_data.get("speed", "-")
                
                panel_content.append(f"[{BrandColors.SUCCESS}]+ 成功:[/{BrandColors.SUCCESS}] {success}")
                panel_content.append(f"[{BrandColors.ERROR}]- 失败:[/{BrandColors.ERROR}] {failed}")
                panel_content.append(f"[{BrandColors.INFO}]P 速度:[/{BrandColors.INFO}] {speed}")
            
            elif section_name == "资源":
                cpu = section_data.get("cpu", "-")
                memory = section_data.get("memory", "-")
                
                panel_content.append(f"[{BrandColors.ACCENT}]CPU:[/{BrandColors.ACCENT}] {cpu}")
                panel_content.append(f"[{BrandColors.ACCENT}]内存:[/{BrandColors.ACCENT}] {memory}")
            
            else:
                for key, value in section_data.items():
                    panel_content.append(f"[bold]{key}:[/bold] {value}")
            
            section_panel = Panel(
                "\n".join(panel_content),
                title=f"[bold]{section_name}[/bold]",
                border_style=BrandColors.SECONDARY,
                box=SIMPLE,
                padding=(0, 1)
            )
            panels.append(section_panel)
        
        columns = Columns(panels, equal=True, expand=True)
        self.console.print(columns)
        self.console.print(f"\n[bold {BrandColors.PRIMARY}]{'─' * 60}[/bold {BrandColors.PRIMARY}]")

    def decorate_status(self, status: str, text: str = None) -> str:
        """装饰状态文本 - 冰璃岩开发组 (BLY Team)
        
        参数:
            status: 状态类型
            text: 可选的附加文本
        返回:
            装饰后的文本
        """
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

    def table_enhanced(
        self,
        title: str,
        columns: List[Dict],
        rows: List[List],
        alternate_rows: bool = True,
        show_borders: bool = True,
        style: str = "rounded"
    ) -> None:
        """增强表格显示 - 冰璃岩开发组 (BLY Team)
        
        参数:
            title: 表格标题
            columns: 列定义
            rows: 行数据
            alternate_rows: 是否使用交替行颜色
            show_borders: 是否显示边框
            style: 表格样式
        """
        from rich.box import ROUNDED, SIMPLE, SIMPLE_HEAVY
        
        box_styles = {
            "rounded": ROUNDED,
            "simple": SIMPLE,
            "none": None,
        }
        
        box = box_styles.get(style, ROUNDED)
        
        table = Table(
            title=title,
            box=box if show_borders else None,
            show_header=True,
            header_style="bold",
            border_style=BrandColors.PRIMARY,
            row_styles=["", "dim"] if alternate_rows else None,
        )
        
        for col in columns:
            col_options = {}
            if "style" in col:
                col_options["style"] = col["style"]
            if "width" in col:
                col_options["width"] = col["width"]
            if "justify" in col:
                col_options["justify"] = col["justify"]
            if "no_wrap" in col:
                col_options["no_wrap"] = col["no_wrap"]
            if "overflow" in col:
                col_options["overflow"] = col["overflow"]
                
            table.add_column(col["header"], **col_options)
        
        for row in rows:
            table.add_row(*[str(cell) if cell is not None else "" for cell in row])
        
        console.print(table)

    def responsive_table(
        self,
        title: str,
        columns: List[Dict],
        rows: List[List],
        min_width: int = 40
    ) -> None:
        """响应式表格 - 冰璃岩开发组 (BLY Team)
        
        根据终端宽度自动调整列宽和显示方式
        
        参数:
            title: 表格标题
            columns: 列定义
            rows: 行数据
            min_width: 最小宽度
        """
        terminal_width = console.width
        available_width = max(terminal_width - 10, min_width)
        
        total_fixed_width = 0
        flexible_columns = []
        
        for col in columns:
            if "width" in col:
                total_fixed_width += col["width"]
            else:
                flexible_columns.append(col)
        
        remaining_width = available_width - total_fixed_width - (len(columns) * 3)
        
        if flexible_columns:
            width_per_flex = max(remaining_width // len(flexible_columns), 10)
            for col in flexible_columns:
                col["width"] = width_per_flex
        
        if terminal_width < 80:
            self._render_compact_table(title, columns, rows)
        else:
            self.table_enhanced(
                title=title,
                columns=columns,
                rows=rows,
                alternate_rows=True,
                show_borders=True,
                style="rounded"
            )

    def _render_compact_table(self, title: str, columns: List[Dict], rows: List[List]) -> None:
        """渲染紧凑型表格（用于窄终端）"""
        console.print(f"\n[bold]{title}[/bold]")
        console.print("[dim]" + "─" * 40 + "[/dim]")
        
        for row_idx, row in enumerate(rows):
            if row_idx > 0:
                console.print("[dim]" + "─" * 40 + "[/dim]")
            
            for col_idx, col in enumerate(columns):
                if col_idx < len(row):
                    header = col.get("header", "")
                    style = col.get("style", None)
                    value = row[col_idx]
                    
                    if style:
                        console.print(f"  [bold]{header}:[/bold] [{style}]{value}[/{style}]")
                    else:
                        console.print(f"  [bold]{header}:[/bold] {value}")

    def check_result_enhanced(
        self,
        checks: List[Dict],
        title: str = "检查结果",
        show_summary: bool = True
    ) -> None:
        """增强的检查结果显示
        
        参数:
            checks: 检查项列表
            title: 表格标题
            show_summary: 是否显示汇总
        """
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

    def beautiful_list(
        self,
        items: List[Dict],
        style: str = "bullet"
    ) -> None:
        """美化列表显示 - 冰璃岩开发组 (BLY Team)
        
        参数:
            items: 列表项
            style: 列表样式
        """
        if style == "bullet":
            self._render_bullet_list(items)
        elif style == "numbered":
            self._render_numbered_list(items)
        elif style == "cards":
            self._render_cards_list(items)
        else:
            self._render_bullet_list(items)

    def _render_bullet_list(self, items: List[Dict]) -> None:
        """渲染项目符号列表"""
        for item in items:
            text = item.get("text", "")
            item_style = item.get("style", None)
            icon = item.get("icon", "•")
            indent = item.get("indent", 0)
            
            indent_str = "  " * indent
            
            if item_style:
                console.print(f"{indent_str}[{item_style}]{icon}[/{item_style}] {text}")
            else:
                console.print(f"{indent_str}{icon} {text}")

    def _render_numbered_list(self, items: List[Dict]) -> None:
        """渲染编号列表"""
        for idx, item in enumerate(items, 1):
            text = item.get("text", "")
            item_style = item.get("style", None)
            indent = item.get("indent", 0)
            
            indent_str = "  " * indent
            
            if item_style:
                console.print(f"{indent_str}[bold cyan]{idx}.[/bold cyan] [{item_style}]{text}[/{item_style}]")
            else:
                console.print(f"{indent_str}[bold cyan]{idx}.[/bold cyan] {text}")

    def _render_cards_list(self, items: List[Dict]) -> None:
        """渲染卡片式列表"""
        for item in items:
            text = item.get("text", "")
            item_style = item.get("style", None)
            icon = item.get("icon", "•")
            detail = item.get("detail", None)
            
            content = f"[bold]{icon} {text}[/bold]"
            if detail:
                content += f"\n[dim]{detail}[/dim]"
            
            border_style = item_style if item_style else BrandColors.PRIMARY
            
            panel = Panel(
                content,
                border_style=border_style,
                box=SIMPLE,
                padding=(0, 1),
            )
            console.print(panel)


class ParallelProgress:
    """多任务并行进度显示 - 冰璃岩开发组 (BLY Team)"""
    
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
        """开始并行进度 - 冰璃岩开发组 (BLY Team)
        
        参数:
            total_tasks: 总任务数
            description: 整体描述
        """
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
        """添加子任务 - 冰璃岩开发组 (BLY Team)
        
        参数:
            name: 任务名称
            total: 任务总进度单位
        """
        if self._progress is None:
            return
        
        task_id = self._progress.add_task(
            f"  [{BrandColors.TEXT_SECONDARY}]{name}[/{BrandColors.TEXT_SECONDARY}]",
            total=total
        )
        self._tasks[name] = task_id
    
    def update_task(self, name: str, advance: int = 1, description: str = None) -> None:
        """更新子任务进度 - 冰璃岩开发组 (BLY Team)
        
        参数:
            name: 任务名称
            advance: 前进步数
            description: 新描述
        """
        if self._progress is None or name not in self._tasks:
            return
        
        task_id = self._tasks[name]
        if description:
            self._progress.update(task_id, description=description, advance=advance)
        else:
            self._progress.advance(task_id, advance)
    
    def complete_task(self, name: str) -> None:
        """标记任务完成 - 冰璃岩开发组 (BLY Team)
        
        参数:
            name: 任务名称
        """
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
        """标记任务失败 - 冰璃岩开发组 (BLY Team)
        
        参数:
            name: 任务名称
            error: 错误信息
        """
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
        """停止进度显示"""
        if self._progress:
            self._progress.stop()
            self._progress = None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
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
    """设置日志"""
    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )

    return logging.getLogger("ppc8")


def config_wizard(console: Console = None, full: bool = False) -> Dict[str, Any]:
    """交互式配置向导 - 冰璃岩开发组 (BLY Team)

    参数:
        console: Console 实例
        full: 是否显示所有配置项
    返回:
        配置字典
    """
    if console is None:
        console = Console()

    config = {}

    console.print()
    console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    console.print(f"[bold white]  {Icons.GEAR} PPC8 配置向导[/bold white]")
    console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    console.print()

    if full:
        console.print(f"[{BrandColors.TEXT_SECONDARY}]完整配置模式：请按照提示完成所有配置。[/ {BrandColors.TEXT_SECONDARY}]")
    else:
        console.print(f"[{BrandColors.TEXT_SECONDARY}]快速配置模式：仅核心设置。使用 [bold]--full[/bold] 可配置所有项。[/ {BrandColors.TEXT_SECONDARY}]")
    console.print()

    # ========== TTS 核心配置 ==========
    console.print(f"[bold {BrandColors.ACCENT}]T TTS 核心配置[/bold {BrandColors.ACCENT}]")
    console.print("[dim]─" * 30 + "[/dim]")

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
        f"\n[{BrandColors.PRIMARY}]请选择语音[/ {BrandColors.PRIMARY}]",
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
        f"\n[{BrandColors.PRIMARY}]请选择超时模式[/ {BrandColors.PRIMARY}]",
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

    # ========== 分段配置 ==========
    if full:
        console.print()
        console.print(f"[bold {BrandColors.ACCENT}]S 文本分段配置[/bold {BrandColors.ACCENT}]")
        console.print("[dim]─" * 30 + "[/dim]")

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

    # ========== 可靠性配置 ==========
    console.print()
    console.print(f"[bold {BrandColors.ACCENT}]R 可靠性配置[/bold {BrandColors.ACCENT}]")
    console.print("[dim]─" * 30 + "[/dim]")

    console.print()
    max_retries = Prompt.ask(
        f"[{BrandColors.PRIMARY}]TTS 最大重试次数 (0-20)[/{BrandColors.PRIMARY}]",
        default="3"
    )
    try:
        retries_value = max(0, min(20, int(max_retries)))
        config["reliability.tts_retry.max_retries"] = retries_value
        config["tts.retries"] = retries_value  # 同步写入 tts.retries
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

    # ========== 性能配置 ==========
    if full:
        console.print()
        console.print(f"[bold {BrandColors.ACCENT}]P 性能配置[/bold {BrandColors.ACCENT}]")
        console.print("[dim]─" * 30 + "[/dim]")

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

    # ========== 文本正则化配置 ==========
    if full:
        console.print()
        console.print(f"[bold {BrandColors.ACCENT}]N 文本正则化配置[/bold {BrandColors.ACCENT}]")
        console.print("[dim]─" * 30 + "[/dim]")

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

    # ========== 章节分割配置 ==========
    if full:
        console.print()
        console.print(f"[bold {BrandColors.ACCENT}]C 章节分割配置[/bold {BrandColors.ACCENT}]")
        console.print("[dim]─" * 30 + "[/dim]")

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
            f"\n[{BrandColors.PRIMARY}]请选择章节预设[/ {BrandColors.PRIMARY}]",
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

    # ========== UI 配置 ==========
    console.print()
    console.print(f"[bold {BrandColors.ACCENT}]U 界面配置[/bold {BrandColors.ACCENT}]")
    console.print("[dim]─" * 30 + "[/dim]")

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

    # ========== 功能开关 ==========
    if full:
        console.print()
        console.print(f"[bold {BrandColors.ACCENT}]F 功能开关[/bold {BrandColors.ACCENT}]")
        console.print("[dim]─" * 30 + "[/dim]")

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

    # ========== 配置预览 ==========
    console.print()
    console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    console.print(f"[bold white]  {Icons.SUCCESS} 配置预览[/bold white]")
    console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")

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

    if Confirm.ask(f"[{BrandColors.SUCCESS}]确认保存配置?[/ {BrandColors.SUCCESS}]", default=True):
        console.print(f"\n[{BrandColors.SUCCESS}]+ 配置已保存！[/ {BrandColors.SUCCESS}]")
        return config
    else:
        console.print(f"\n[{BrandColors.WARNING}]! 配置已取消[/ {BrandColors.WARNING}]")
        return None
