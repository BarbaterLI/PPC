"""Output formatting module - Clean, friendly terminal output using Rich."""

import contextlib
import io
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
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
from rich.table import Table

from src.cli.design import atoms, layouts
from src.cli.design.renderers import RENDERERS
from src.cli.design.tokens import COLORS, get_icon, get_style
from src.cli.design.tokens import (
    set_no_color as _tokens_set_no_color,
    set_no_emoji as _tokens_set_no_emoji,
)
from src.utils.elevation import set_console_factory

from .branding import BrandingMixin, StatusIcons
from .help_browser import HelpBrowserMixin
from .panels import PanelsMixin
from .wizard import WizardMixin


def setup_console() -> None:
    """显式初始化控制台：模块搜索路径 + stdout/stderr UTF-8 包装（幂等）。"""
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 任何平台下 stdout/stderr 编码非 UTF-8 时都强制包装为 UTF-8：
    # - Windows 传统控制台默认 GBK/cp936；
    # - Linux 的 LC_ALL=C / POSIX locale 下 stdout 编码为 ASCII/ANSI_X3.4-1968，
    #   会导致 Rich 与中文输出崩溃或乱码。
    # 仅在编码确实非 UTF-8 时包装，避免对正常环境产生任何开销。
    if sys.stdout.encoding != "utf-8" or sys.stderr.encoding != "utf-8":
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        stdout_buffer = getattr(sys.stdout, "buffer", None)
        stderr_buffer = getattr(sys.stderr, "buffer", None)
        if sys.stdout.encoding != "utf-8" and stdout_buffer is not None:
            sys.stdout = io.TextIOWrapper(stdout_buffer, encoding="utf-8", line_buffering=True)
        if sys.stderr.encoding != "utf-8" and stderr_buffer is not None:
            sys.stderr = io.TextIOWrapper(stderr_buffer, encoding="utf-8", line_buffering=True)

    # 向 utils.elevation 注入全局 console 工厂，解耦 utils → cli 的层级倒置。
    set_console_factory(lambda: console)


# 全局 console:不显式绑定 file,让 Rich 在写入时读取 sys.stdout。
# 这样既能跟随 setup_console() 的 Windows UTF-8 包装,又能与 pytest capsys 协作。
console = Console(legacy_windows=False)


# 全局 --no-color 标记(由 typer_app 在解析后写入)
no_color: bool = False


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


class OutputFormatter(BrandingMixin, PanelsMixin, WizardMixin, HelpBrowserMixin):
    """Output formatter with Rich console support."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        # 复用全局 console(Spec unify-log-and-cli-ui Task 4):确保所有输出
        # 走同一个 Console 实例,set_no_color / RichHandler 等只需配置一处。
        self.console = console
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
        self._renderers = {
            "human": RENDERERS["human"](self.console),
            "json": RENDERERS["json"](),
            "quiet": RENDERERS["quiet"](),
        }

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
        self._renderers[self.mode].render(component)

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

    def error(self, exc) -> None:
        """统一错误渲染入口。

        支持两种调用方式:
        1. ``error(CLIError(...))`` —— 输出错误代码、消息与 Hint。
        2. ``error("xxx")`` / ``error(Exception(...))`` —— 兼容旧用法,渲染单行错误。

        ``--verbose`` 时追加 stack;``--json`` 时输出单行 JSON。
        """
        # 延迟 import 避免循环
        from ..core.exceptions import CLIError

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
        """原始错误消息(走 design tokens 主题),不渲染 traceback。

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

    def detail(self, message: str, **kwargs):
        """Detail message - verbose-aware info output.

        与 :meth:`info` 不同,仅在 ``--verbose`` 开启时渲染;
        与 :meth:`debug` 不同,使用普通样式(无 DEBUG 前缀),适合
        配置回显、分任务进度等"非默认但正常"的信息输出。
        """
        if not self.verbose:
            return
        self._render(atoms.Message("detail", message, timestamp=self._timestamp() if self.timestamps else None))

    def _log(self, level: str, message: str):
        """Log with timestamp。"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"{timestamp} | {level:7s} | {message}")

    def title(self, text: str):
        """Output title."""
        console.print(f"\n{text}", style=get_style("header"))
        console.print("=" * len(text))

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

        self._render(atoms.Panel(title=f"ppc {command}", content=content, style="success"))

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
            console.print(msg, style=get_style("retry"))
        else:
            msg = f"{get_icon('error')} 已重试 {info.attempt} 次，最终失败：{info.error}"
            console.print(msg, style=get_style("error"))

    def task_status(self, task: TaskStatus):
        """Show single task status."""
        status_icons = {
            "pending": get_icon("pending"),
            "running": get_icon("running"),
            "completed": get_icon("completed"),
            "failed": get_icon("error"),
        }
        status_styles = {
            "pending": get_style("info"),
            "running": get_style("task_running"),
            "completed": get_style("task_completed"),
            "failed": get_style("task_failed"),
        }

        icon = status_icons.get(task.status, "?")
        style = status_styles.get(task.status, get_style("info"))

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
            BarColumn(complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        self._progress.start()

        self._overall_task = self._progress.add_task(
            f"[{COLORS.primary}]{description}[/{COLORS.primary}]", total=total_tasks
        )

    def add_task(self, name: str, total: int = 100) -> None:
        """Add sub-task."""
        if self._progress is None:
            return

        task_id = self._progress.add_task(
            f"  [{COLORS.muted}]{name}[/{COLORS.muted}]", total=total
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
            description=f"  [{COLORS.success}]+ {name}[/{COLORS.success}]",
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
        self._progress.update(task_id, description=f"  [{COLORS.error}]- {name}: {error}[/{COLORS.error}]")

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


def setup_logging(verbose: bool = False, quiet: bool = False, json_mode: bool = False) -> logging.Logger:
    """初始化日志系统,挂载 RichHandler 到根 logger。

    Args:
        verbose: 启用 DEBUG 级别日志。
        quiet: 仅输出 WARNING 及以上级别日志。
        json_mode: 当前仅作为占位参数(quiet 优先级最高,日志走全局 console)。

    Returns:
        已配置的 ``ppc`` logger。
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # RichHandler 输出到 stderr,不干扰 JSON stdout
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_path=verbose,
        show_time=True,
        tracebacks_show_locals=verbose,
    )
    handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    # 清除已有 handlers(避免重复)
    root.handlers.clear()
    root.addHandler(handler)

    logger = logging.getLogger("ppc")
    logger.setLevel(level)
    return logger
