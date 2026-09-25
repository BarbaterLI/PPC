"""Higher-level CLI layout components.

A Layout composes multiple :mod:`src.cli.design.atoms` into a screen-like
output. Each layout exposes ``to_components()`` for renderer-agnostic output
and ``to_rich()`` for convenient combined Rich rendering.
"""

from __future__ import annotations

from typing import Any

from rich.console import Group

from src.cli.design import atoms
from src.cli.design.tokens import COLORS, get_icon
from src.utils.format import format_duration

Atom = atoms.Message | atoms.Panel | atoms.Table | atoms.StatGrid | atoms.ProgressBar | atoms.Trace | atoms.CommandHelp


class BaseLayout:
    """组合 ``to_components()`` 的原子并统一聚合成 Rich Group。"""

    def to_components(self) -> list[Atom]:
        raise NotImplementedError

    def to_rich(self) -> Group:
        return Group(*[c.to_rich() for c in self.to_components()])



class WelcomeLayout(BaseLayout):
    """Brand welcome screen: banner, version/tagline, commands and tips."""

    def __init__(
        self,
        version: str,
        tagline: str,
        commands: list[tuple[str, str]],
        tips: list[str],
    ) -> None:
        self.version = version
        self.tagline = tagline
        self.commands = commands
        self.tips = tips

    def to_components(self) -> list[Atom]:
        """Return ordered atoms for the welcome screen."""
        components: list[Atom] = []

        # Brand banner panel
        banner_lines = [
            f"[{COLORS.primary}]PPC_NEXT[/{COLORS.primary}]",
            "",
            f"[{COLORS.accent}]版本 {self.version}[/{COLORS.accent}]",
            f"[{COLORS.secondary}]{self.tagline}[/{COLORS.secondary}]",
        ]
        components.append(
            atoms.Panel(
                title=f"[{COLORS.accent}]{get_icon('rocket')} 欢迎使用 PPC_NEXT[/{COLORS.accent}]",
                content=banner_lines,
                style="primary",
            )
        )

        # Command grid
        command_rows = [[cmd, desc] for cmd, desc in self.commands]
        components.append(
            atoms.Table(
                headers=["命令", "说明"],
                rows=command_rows,
                title=f"{get_icon('book')} 常用命令",
            )
        )

        # Quick tips panel
        tip_lines = [f"  {get_icon('info')} {tip}" for tip in self.tips]
        components.append(
            atoms.Panel(
                title=f"[{COLORS.accent}]{get_icon('star')} 快速入门[/{COLORS.accent}]",
                content=tip_lines,
                style="accent",
            )
        )

        return components

    


class CommandHelpLayout(BaseLayout):
    """Command help detail panel: wraps :class:`atoms.CommandHelp` for renderer-agnostic output."""

    def __init__(
        self,
        command: str,
        description: str,
        usage: str,
        examples: list[dict[str, str]] | None = None,
        options: list[dict[str, str]] | None = None,
        see_also: list[str] | None = None,
    ) -> None:
        self.command = command
        self.description = description
        self.usage = usage
        self.examples = examples or []
        self.options = options or []
        self.see_also = see_also or []

    def to_components(self) -> list[Atom]:
        """Return the command help as a single atom."""
        return [
            atoms.CommandHelp(
                command=self.command,
                description=self.description,
                usage=self.usage,
                examples=self.examples,
                options=self.options,
                see_also=self.see_also,
            )
        ]

    


class TaskDashboardLayout(BaseLayout):
    """Live task dashboard: header, progress bar and stat grid."""

    def __init__(
        self,
        total: int,
        completed: int,
        failed: int,
        current_task: str | None,
        speed: float,
        elapsed: float,
        eta: float,
    ) -> None:
        self.total = total
        self.completed = completed
        self.failed = failed
        self.current_task = current_task
        self.speed = speed
        self.elapsed = elapsed
        self.eta = eta

    def to_components(self) -> list[Atom]:
        """Return ordered atoms for the dashboard."""
        components: list[Atom] = []

        current = self.completed + self.failed
        progress_atom = atoms.ProgressBar(current, self.total, width=20)

        header_text = f"{get_icon('chart')} 实时任务看板"
        if self.current_task:
            header_text += f" - {self.current_task}"
        components.append(atoms.Message("info", header_text))
        components.append(progress_atom)

        items: dict[str, Any] = {
            "总任务": self.total,
            "已完成": self.completed,
            "失败": self.failed,
            "进度": f"{current}/{self.total}",
            "速度": f"{self.speed:.2f} 任务/秒",
            "已用时间": format_duration(self.elapsed),
            "预计剩余": format_duration(self.eta),
        }
        if self.current_task:
            items["当前任务"] = self.current_task

        components.append(atoms.StatGrid(items=items, title="任务统计"))
        return components

    


class CompletionReportLayout(BaseLayout):
    """Completion summary with optional error distribution table."""

    def __init__(
        self,
        total: int,
        completed: int,
        failed: int,
        elapsed: float,
        error_type_counts: dict[str, int] | None = None,
        retries: int = 0,
    ) -> None:
        self.total = total
        self.completed = completed
        self.failed = failed
        self.elapsed = elapsed
        self.error_type_counts = error_type_counts or {}
        self.retries = retries

    @property
    def success_rate(self) -> float:
        """Compute success rate as a percentage."""
        if self.total <= 0:
            return 0.0
        return (self.completed / self.total) * 100

    def to_components(self) -> list[Atom]:
        """Return ordered atoms for the completion report."""
        components: list[Atom] = []

        rate = self.success_rate
        if rate >= 90:
            result_text = "优秀"
            result_style = "success"
        elif rate >= 70:
            result_text = "良好"
            result_style = "warning"
        else:
            result_text = "需改进"
            result_style = "error"

        summary_lines = [
            f"{get_icon(result_style)} 总体评价：{result_text} ({rate:.1f}%)",
            "",
            f"总任务数：{self.total}",
            f"成功：{self.completed}",
            f"失败：{self.failed}",
            f"总用时：{format_duration(self.elapsed)}",
        ]
        if self.retries > 0:
            summary_lines.append(f"总重试次数：{self.retries}")

        components.append(
            atoms.Panel(
                title=f"{get_icon('star')} 处理结果汇总",
                content=summary_lines,
                style=result_style,
            )
        )

        if self.error_type_counts:
            total_errors = sum(self.error_type_counts.values())
            rows = []
            for error_type, count in sorted(self.error_type_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_errors * 100) if total_errors > 0 else 0
                rows.append([error_type, count, f"{percentage:.1f}%"])
            components.append(
                atoms.Table(
                    headers=["错误类型", "数量", "占比"],
                    rows=rows,
                    title=f"{get_icon('info')} 错误类型分布",
                )
            )

        return components

    


class ConfigPreviewLayout(BaseLayout):
    """Key/value config preview with friendly labels."""

    def __init__(
        self,
        config: dict[str, Any],
        labels: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.labels = labels or {}

    def _format_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "是" if value else "否"
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        if isinstance(value, dict):
            return ", ".join(f"{k}: {v}" for k, v in value.items())
        return str(value)

    def to_components(self) -> list[Atom]:
        """Return ordered atoms for the config preview."""
        rows = []
        for key, value in self.config.items():
            label = self.labels.get(key, key)
            rows.append([label, self._format_value(value)])

        return [atoms.Table(headers=["配置项", "值"], rows=rows, title="当前配置")]

    


class ErrorLayout(BaseLayout):
    """Error panel with optional hint and suggestions."""

    def __init__(
        self,
        code: str,
        message: str,
        hint: str | None = None,
        suggestions: list[str] | None = None,
        verbose: bool = False,
        cause: Exception | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.hint = hint
        self.suggestions = suggestions or []
        self.verbose = verbose
        self.cause = cause

    def to_components(self) -> list[Atom]:
        """Return ordered atoms for the error screen."""
        components: list[Atom] = []

        content_lines = [
            f"[bold {COLORS.error}]{get_icon('error')} {self.code}[/bold {COLORS.error}]",
            "",
            self.message,
        ]

        if self.hint:
            content_lines.append("")
            content_lines.append(f"{get_icon('info')} 提示：{self.hint}")

        if self.suggestions:
            content_lines.append("")
            content_lines.append(f"{get_icon('info')} 修复建议：")
            for suggestion in self.suggestions:
                content_lines.append(f"  + {suggestion}")

        components.append(
            atoms.Panel(
                title=f"[{COLORS.error}]{get_icon('error')} 错误[/{COLORS.error}]",
                content=content_lines,
                style="error",
            )
        )

        if self.verbose and self.cause is not None:
            components.append(atoms.Trace(self.cause, expanded=True))

        return components

    


class StepLayout(BaseLayout):
    """Wizard step header with progress bar and title."""

    def __init__(
        self,
        step: int,
        total: int,
        title: str,
        icon: str = "",
    ) -> None:
        self.step = step
        self.total = total
        self.title = title
        self.icon = icon

    def to_components(self) -> list[Atom]:
        """Return ordered atoms for the step header."""
        if self.total <= 0:
            progress = atoms.ProgressBar(0, 0, width=30)
        else:
            progress = atoms.ProgressBar(self.step, self.total, width=30)

        label = f"{self.icon} {self.title}" if self.icon else self.title
        message = atoms.Message(
            "info",
            f"步骤 {self.step}/{self.total}：{label}",
        )
        return [message, progress]

    
