"""面板族渲染 - 成功/警告/错误面板与统计面板。"""

from datetime import datetime
from typing import Any

from rich.box import ROUNDED, SIMPLE
from rich.panel import Panel
from rich.table import Table

from src.cli.design import atoms, layouts
from src.cli.design.tokens import COLORS, get_icon
from src.utils.format import format_duration


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


class PanelsMixin:
    """面板族 mixin：panel / error_panel / success_panel / warning_panel / 统计面板。"""

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
        details: dict[str, Any] | None = None,
    ) -> None:
        """Styled warning panel."""
        content_parts = []
        content_parts.append(f"[bold {COLORS.warning}]{get_icon('warning')} {message}[/bold {COLORS.warning}]")

        if suggestion:
            content_parts.append(f"\n[bold {COLORS.accent}]{get_icon('info')} 建议:[/bold {COLORS.accent}]")
            content_parts.append(f"  [green]+[/green] {suggestion}")

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

        # 额外信息走 _render() 分派(Spec unify-log-and-cli-ui Task 7):
        # 错误位置与消息用 error 级别(quiet 模式下仍输出到 stderr),
        # 建议用 info / success 级别(对应原 COLORS.info / green 视觉)。
        self._render(atoms.Message(level="error", text=f"错误位置: {error_type}"))
        if error_msg:
            self._render(atoms.Message(level="error", text=error_msg))

        suggestions = ErrorSuggestions.get_suggestions(error_type)
        if suggestions:
            self._render(atoms.Message(level="info", text="可能的解决方案:"))
            for sug in suggestions[:3]:
                self._render(atoms.Message(level="success", text=sug))

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
            lines.append(f"  已用时间：{format_duration(elapsed)}")
            lines.append(f"  预计剩余：{format_duration(eta)}")
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
