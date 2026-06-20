"""Atomic CLI 组件。

每个 Atom 都提供 ``to_rich()`` 与 ``to_plain()`` 两个出口，分别用于 Rich 渲染与纯文本输出，
视觉常量统一从 ``src.cli.design.tokens`` 读取。
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

from rich.panel import Panel as RichPanel
from rich.table import Table as RichTable
from rich.text import Text
from rich.traceback import Traceback

from src.cli.design.tokens import SPACING, get_icon, get_style


@dataclass
class Message:
    """单行状态消息。"""

    level: str
    text: str
    timestamp: str | None = None

    def to_plain(self) -> str:
        """返回纯文本格式：可选时间戳 + 图标 + 消息内容。"""
        icon = get_icon(self.level) if self.level in ("info", "success", "warning", "error") else ""
        parts = [p for p in (self.timestamp, icon, self.text) if p]
        return " ".join(parts)

    def to_rich(self) -> Text:
        """返回带样式的 Rich Text。"""
        icon = get_icon(self.level) if self.level in ("info", "success", "warning", "error") else ""
        parts = [p for p in (self.timestamp, icon, self.text) if p]
        text = Text(" ".join(parts))
        text.stylize(get_style(self.level))
        return text


@dataclass
class Panel:
    """带标题的卡片容器。"""

    title: str | None
    content: str | list[str]
    style: str = "info"

    def _content_text(self) -> str:
        if isinstance(self.content, list):
            return "\n".join(self.content)
        return self.content

    def to_plain(self) -> str:
        """返回简单的纯文本面板表示。"""
        lines = []
        if self.title:
            lines.append(f"--- {self.title} ---")
        lines.append(self._content_text())
        return "\n".join(lines)

    def to_rich(self) -> RichPanel:
        """返回 Rich Panel。"""
        return RichPanel(
            self._content_text(),
            title=self.title,
            border_style=get_style(self.style),
            padding=(SPACING.sm, SPACING.md),
        )


@dataclass
class Table:
    """基于 Rich 的表格组件。"""

    headers: list[str]
    rows: list[list[Any]]
    title: str | None = None

    def to_plain(self) -> str:
        """返回左对齐的简易文本表格。"""
        widths = [len(str(h)) for h in self.headers]
        for row in self.rows:
            for i, cell in enumerate(row):
                if i >= len(widths):
                    widths.append(0)
                widths[i] = max(widths[i], len(str(cell)))

        def fmt_row(values: list[Any]) -> str:
            cells = [str(v).ljust(widths[i]) for i, v in enumerate(values)]
            return " | ".join(cells)

        lines = []
        if self.title:
            lines.append(self.title)
        lines.append(fmt_row(self.headers))
        lines.append("-" * (sum(widths) + 3 * (len(widths) - 1)))
        for row in self.rows:
            lines.append(fmt_row(row))
        return "\n".join(lines)

    def to_rich(self) -> RichTable:
        """返回 Rich Table。"""
        table = RichTable(title=self.title, header_style=get_style("bold"))
        for header in self.headers:
            table.add_column(header)
        for row in self.rows:
            table.add_row(*[str(cell) for cell in row])
        return table


@dataclass
class StatGrid:
    """两列键值统计网格。"""

    items: dict[str, Any]
    title: str | None = None

    def to_plain(self) -> str:
        """返回 ``Key: Value`` 形式的纯文本。"""
        lines = []
        if self.title:
            lines.append(self.title)
        for key, value in self.items.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def to_rich(self) -> RichTable:
        """返回两列 Rich Table。"""
        table = RichTable(title=self.title, show_header=False, box=None)
        table.add_column("Key", style=get_style("muted"))
        table.add_column("Value", style=get_style("bold"))
        for key, value in self.items.items():
            table.add_row(str(key), str(value))
        return table


@dataclass
class ProgressBar:
    """纯文本进度条。"""

    current: int
    total: int
    width: int = 20

    def to_plain(self) -> str:
        """返回形如 ``[████░░░░░░] 40%`` 的进度条字符串。"""
        ratio = 0.0 if self.total <= 0 else min(1.0, max(0.0, self.current / self.total))
        filled = int(ratio * self.width)
        empty = self.width - filled
        bar = "█" * filled + "░" * empty
        percent = int(ratio * 100)
        return f"[{bar}] {percent}%"

    def to_rich(self) -> Text:
        """返回带样式的进度条 Rich Text。"""
        plain = self.to_plain()
        text = Text(plain)
        text.stylize(get_style("secondary"))
        return text


@dataclass
class Trace:
    """可折叠的异常跟踪信息。"""

    exception: Exception
    expanded: bool = False
    max_lines: int = 20

    def to_plain(self) -> str:
        """返回截断后的标准 traceback 字符串。"""
        tb_lines = traceback.format_exception(type(self.exception), self.exception, self.exception.__traceback__)
        text = "".join(tb_lines)
        lines = text.splitlines()
        if len(lines) > self.max_lines:
            lines = lines[: self.max_lines] + ["... (truncated)"]
        return "\n".join(lines)

    def to_rich(self) -> Traceback:
        """返回 Rich Traceback。"""
        return Traceback(
            self.exception,  # type: ignore[arg-type]  # Rich Traceback 接受 Exception 实例
            show_locals=self.expanded,
            width=88,
            max_frames=self.max_lines,
        )


@dataclass
class CommandHelp:
    """命令帮助面板。"""

    command: str
    description: str
    usage: str
    examples: list[dict[str, str]] = field(default_factory=list)
    options: list[dict[str, str]] = field(default_factory=list)
    see_also: list[str] = field(default_factory=list)

    def _build_lines(self) -> list[str]:
        lines: list[str] = []
        lines.append(f"命令: {self.command}")
        lines.append("")
        lines.append(self.description)
        lines.append("")
        lines.append(f"用法: {self.usage}")

        if self.options:
            lines.append("")
            lines.append("选项:")
            for opt in self.options:
                name = opt.get("name", "")
                desc = opt.get("description", "")
                lines.append(f"  {name:<20} {desc}")

        if self.examples:
            lines.append("")
            lines.append("示例:")
            for ex in self.examples:
                cmd = ex.get("command", "")
                desc = ex.get("description", "")
                lines.append(f"  {cmd}")
                if desc:
                    lines.append(f"    {desc}")

        if self.see_also:
            lines.append("")
            lines.append(f"参见: {', '.join(self.see_also)}")

        return lines

    def to_plain(self) -> str:
        """返回纯文本帮助信息。"""
        return "\n".join(self._build_lines())

    def to_rich(self) -> RichPanel:
        """返回 Rich 帮助面板。"""
        content = "\n".join(self._build_lines())
        return RichPanel(
            content,
            title=f"帮助: {self.command}",
            border_style=get_style("info"),
            padding=(SPACING.sm, SPACING.md),
        )
