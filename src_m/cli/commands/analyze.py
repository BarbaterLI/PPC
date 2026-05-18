"""分析命令 - 深度系统分析与健康评分。

提供性能、配置、错误模式、依赖、网络、资源和代码质量分析，
支持自动修复、历史对比、持续监控和 JSON/HTML 导出。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.box import SIMPLE

from ..output import OutputFormatter, Icons, BrandColors
from ...analysis.engine import AnalysisEngine
from ...analysis.models import (
    AnalysisCategory,
    AnalysisIssue,
    HealthReport,
    Severity,
)
from ...analysis.analyzers import (
    PerformanceAnalyzer,
    ConfigAnalyzer,
    ErrorPatternAnalyzer,
    DependencyAnalyzer,
    NetworkAnalyzer,
    ResourceAnalyzer,
    CodeQualityAnalyzer,
)
from ...analysis.history import AnalysisHistoryManager
from ...analysis.diff import AnalysisDiffer, compute_diff
from ...analysis.html_report import HTMLReportGenerator


class AnalyzeIcons:
    """Analysis icons - Windows terminal compatible."""
    PERFORMANCE = "[PERF]"
    CONFIG = "[CFG]"
    ERRORS = "[ERR]"
    HEALTH = "[HLT]"
    FIX = "[FIX]"
    DEPENDENCY = "[DEP]"
    NETWORK = "[NET]"
    RESOURCE = "[RES]"
    QUALITY = "[QLT]"


SEVERITY_COLORS = {
    Severity.CRITICAL: BrandColors.ERROR,
    Severity.HIGH: BrandColors.ERROR,
    Severity.MEDIUM: BrandColors.WARNING,
    Severity.LOW: BrandColors.INFO,
    Severity.INFO: BrandColors.TEXT_SECONDARY,
}

CATEGORY_LABELS = {
    AnalysisCategory.PERFORMANCE: "性能",
    AnalysisCategory.MEMORY: "内存",
    AnalysisCategory.CONFIGURATION: "配置",
    AnalysisCategory.RELIABILITY: "可靠性",
    AnalysisCategory.DEPENDENCY: "依赖",
    AnalysisCategory.NETWORK: "网络",
    AnalysisCategory.RESOURCE: "资源",
    AnalysisCategory.CODE_QUALITY: "代码质量",
}

CATEGORY_ICONS = {
    AnalysisCategory.PERFORMANCE: AnalyzeIcons.PERFORMANCE,
    AnalysisCategory.MEMORY: AnalyzeIcons.PERFORMANCE,
    AnalysisCategory.CONFIGURATION: AnalyzeIcons.CONFIG,
    AnalysisCategory.RELIABILITY: AnalyzeIcons.ERRORS,
    AnalysisCategory.DEPENDENCY: AnalyzeIcons.DEPENDENCY,
    AnalysisCategory.NETWORK: AnalyzeIcons.NETWORK,
    AnalysisCategory.RESOURCE: AnalyzeIcons.RESOURCE,
    AnalysisCategory.CODE_QUALITY: AnalyzeIcons.QUALITY,
}


def _score_label(score: int) -> tuple[str, str]:
    if score >= 90:
        return ("优秀", BrandColors.SUCCESS)
    elif score >= 70:
        return ("良好", BrandColors.WARNING)
    elif score >= 50:
        return ("一般", BrandColors.ACCENT)
    else:
        return ("需改进", BrandColors.ERROR)


def _severity_icon(severity: Severity) -> str:
    if severity == Severity.CRITICAL:
        return "-"
    elif severity == Severity.HIGH:
        return "!"
    elif severity == Severity.MEDIUM:
        return "i"
    elif severity == Severity.LOW:
        return "i"
    return "i"


async def _run_analysis(
    performance: bool,
    config: bool,
    errors: bool,
    dependency: bool = False,
    network: bool = False,
    resource: bool = False,
    quality: bool = False,
) -> HealthReport:
    engine = AnalysisEngine()

    if performance:
        engine.register(PerformanceAnalyzer())
    if config:
        engine.register(ConfigAnalyzer())
    if errors:
        engine.register(ErrorPatternAnalyzer())
    if dependency:
        engine.register(DependencyAnalyzer())
    if network:
        engine.register(NetworkAnalyzer())
    if resource:
        engine.register(ResourceAnalyzer())
    if quality:
        engine.register(CodeQualityAnalyzer())

    return await engine.run()


def _display_issues(
    output: OutputFormatter,
    issues: List[AnalysisIssue],
) -> None:
    if not issues:
        output.success("未发现异常")
        return

    table = Table(
        show_header=True,
        box=SIMPLE,
        border_style=BrandColors.PRIMARY,
    )
    table.add_column("级别", width=8)
    table.add_column("类别", width=10)
    table.add_column("位置", width=20)
    table.add_column("描述", width=40)
    table.add_column("建议", width=30)

    for issue in issues:
        color = SEVERITY_COLORS.get(issue.severity, BrandColors.TEXT_SECONDARY)
        icon = _severity_icon(issue.severity)
        severity_text = f"[{color}]{icon} {issue.severity.value.upper()}[/{color}]"
        category_text = CATEGORY_LABELS.get(issue.category, issue.category.value)
        location_text = issue.location or "-"
        desc_text = issue.description
        suggestion_text = issue.suggestion or "-"

        table.add_row(
            severity_text,
            category_text,
            location_text,
            desc_text,
            suggestion_text,
        )

    output.console.print(table)


def _display_health_score(
    output: OutputFormatter,
    score: int,
) -> None:
    label, color = _score_label(score)
    score_bar_width = 40
    filled = int(score_bar_width * (score / 100))
    bar = f"[{'█' * filled}{'░' * (score_bar_width - filled)}]"

    panel_content = (
        f"[bold {color}]{AnalyzeIcons.HEALTH} 健康评分: {score}/100[/bold {color}]\n"
        f"[{color}]{bar}[/{color}]\n"
        f"[bold {color}]评价: {label}[/bold {color}]"
    )

    panel = Panel(
        panel_content,
        title="[bold]健康度汇总[/bold]",
        border_style=color,
        box=SIMPLE,
    )
    output.console.print(panel)


def _apply_fixes(
    output: OutputFormatter,
    issues: List[AnalysisIssue],
) -> int:
    fixed = 0
    for issue in issues:
        if issue.category != AnalysisCategory.CONFIGURATION:
            continue
        if not issue.suggestion:
            continue

        try:
            from ...config.manager import ConfigManager
            config_manager = ConfigManager()

            if issue.location == "tts.voice" and "tts.voice" in issue.suggestion:
                continue

            if "timeout_min" in (issue.location or "") and "timeout_max" in (issue.location or ""):
                try:
                    config_manager.update_config("tts.timeout_min", 45, source="auto_fix")
                    config_manager.update_config("tts.timeout_max", 900, source="auto_fix")
                    output.success(f"已修复: {issue.description}")
                    fixed += 1
                except Exception as e:
                    output.error(f"修复失败: {e}")
                continue

            if "concurrency" in (issue.location or "") and "timeout" in (issue.location or ""):
                try:
                    config_manager.update_config("tts.timeout", 120, source="auto_fix")
                    output.success(f"已修复: 超时时间调整为 120s")
                    fixed += 1
                except Exception as e:
                    output.error(f"修复失败: {e}")
                continue

            if "retries" in (issue.location or "") and "auto_retry" in (issue.location or ""):
                try:
                    config_manager.update_config("tts.retries", 3, source="auto_fix")
                    output.success(f"已修复: retries 设置为 3")
                    fixed += 1
                except Exception as e:
                    output.error(f"修复失败: {e}")
                continue

            if "rate_limit" in (issue.location or "") and "buffer_size" in (issue.location or ""):
                try:
                    config_manager.update_config("tts.buffer_size", 32, source="auto_fix")
                    output.success(f"已修复: buffer_size 设置为 32")
                    fixed += 1
                except Exception as e:
                    output.error(f"修复失败: {e}")
                continue

            if "tts.retries" in (issue.location or "") and "reliability.tts_retry.max_retries" in (issue.location or ""):
                try:
                    tts_retries = config_manager.get("tts.retries")
                    config_manager.update_config("reliability.tts_retry.max_retries", tts_retries, source="auto_fix")
                    output.success(f"已修复: 统一重试次数为 {tts_retries}")
                    fixed += 1
                except Exception as e:
                    output.error(f"修复失败: {e}")
                continue

        except Exception as e:
            output.error(f"修复失败: {e}")

    return fixed


def _display_diff(
    output: OutputFormatter,
    current: HealthReport,
    previous: HealthReport,
) -> None:
    diff = compute_diff(current, previous)

    score_label_current, score_color_current = _score_label(current.score)
    score_label_prev, _ = _score_label(previous.score)

    panel_content = (
        f"[bold]较上次分析: "
        f"[{'green' if diff.score_diff >= 0 else 'red'}]{'+' if diff.score_diff >= 0 else ''}{diff.score_diff} 分[/]"
        f" (当前 {current.score}/100 - {score_label_current}, "
        f"上次 {previous.score}/100 - {score_label_prev})[/bold]\n\n"
    )

    if diff.new_issues:
        panel_content += (
            f"[bold {BrandColors.ERROR}]  ╳ 新增 ({len(diff.new_issues)} 项)[/bold {BrandColors.ERROR}]\n"
        )
        for issue in diff.new_issues:
            panel_content += f"    - {issue.description}\n"

    if diff.fixed_issues:
        panel_content += (
            f"\n[bold {BrandColors.SUCCESS}]  ✓ 已修复 ({len(diff.fixed_issues)} 项)[/bold {BrandColors.SUCCESS}]\n"
        )
        for issue in diff.fixed_issues:
            panel_content += f"    - {issue.description}\n"

    if diff.persistent_issues:
        panel_content += (
            f"\n[bold {BrandColors.WARNING}]  → 持续存在 ({len(diff.persistent_issues)} 项)[/bold {BrandColors.WARNING}]\n"
        )
        for issue in diff.persistent_issues:
            panel_content += f"    - {issue.description}\n"

    panel = Panel(
        panel_content,
        title="[bold]差异对比[/bold]",
        border_style=BrandColors.ACCENT,
        box=SIMPLE,
    )
    output.console.print(panel)


def _export_html_report(
    output: OutputFormatter,
    report: HealthReport,
    export_html: str,
    previous: Optional[HealthReport] = None,
) -> None:
    try:
        export_path = Path(export_html)
        if not export_path.suffix:
            export_path = export_path.with_suffix(".html")

        generator = HTMLReportGenerator()

        diff = None
        if previous is not None:
            diff = compute_diff(report, previous)

        history_mgr = AnalysisHistoryManager()
        history = history_mgr.list_reports(limit=10)

        generator.generate(report, export_path, diff=diff, history=history)

        output.success_panel(
            f"HTML 报告已导出: {export_path}",
            title="导出成功",
            details={
                "文件路径": str(export_path),
                "问题总数": str(len(report.issues)),
                "健康评分": f"{report.score}/100",
            },
        )
    except Exception as e:
        output.error_panel(
            f"HTML 导出失败: {e}",
            title="导出错误",
            error_type=type(e).__name__,
            suggestion="检查文件路径是否正确且有写入权限",
        )


def _display_results(
    output: OutputFormatter,
    report: HealthReport,
    categories_info: List[str],
) -> None:
    output.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    output.console.print(f"[bold white]  {Icons.CHART} 分析结果[/bold white]")
    output.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n")

    _display_health_score(output, report.score)
    output.console.print()

    if report.issues:
        grouped: Dict[AnalysisCategory, List[AnalysisIssue]] = {}
        for issue in report.issues:
            grouped.setdefault(issue.category, []).append(issue)

        for category, issues in grouped.items():
            icon = CATEGORY_ICONS.get(category, "[ANL]")
            label = CATEGORY_LABELS.get(category, category.value)
            output.console.print(f"\n[bold {BrandColors.PRIMARY}]{icon} {label} ({len(issues)} 项)[/bold {BrandColors.PRIMARY}]")
            output.console.print(f"[dim]{'─' * 50}[/dim]")
            _display_issues(output, issues)
    else:
        output.success_panel("所有检查项均正常，系统状态良好", title="分析完成")


def _apply_fixes_interactive(
    output: OutputFormatter,
    report: HealthReport,
) -> None:
    if not report.issues:
        return

    output.console.print(f"\n[bold {BrandColors.ACCENT}]{'─' * 60}[/bold {BrandColors.ACCENT}]")
    output.console.print(f"[bold white]  {AnalyzeIcons.FIX} 自动修复[/bold white]")
    output.console.print(f"[bold {BrandColors.ACCENT}]{'─' * 60}[/bold {BrandColors.ACCENT}]\n")

    try:
        if output.console.is_terminal:
            if Confirm.ask(
                f"[{BrandColors.INFO}]是否执行自动修复？[/{BrandColors.INFO}]",
                default=False,
            ):
                fixed_count = _apply_fixes(output, report.issues)
                output.console.print(
                    f"\n[bold {BrandColors.SUCCESS}]完成修复: {fixed_count} 项[/bold {BrandColors.SUCCESS}]"
                )
            else:
                output.console.print("[dim]已跳过自动修复[/dim]")
        else:
            fixed_count = _apply_fixes(output, report.issues)
            output.console.print(
                f"\n[bold {BrandColors.SUCCESS}]完成修复: {fixed_count} 项[/bold {BrandColors.SUCCESS}]"
            )
    except Exception as e:
        output.error(f"自动修复失败: {e}")


def _export_json_report(
    output: OutputFormatter,
    report: HealthReport,
    export: str,
) -> None:
    try:
        export_path = Path(export)
        if not export_path.suffix:
            export_path = export_path.with_suffix(".json")

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        output.success_panel(
            f"报告已导出: {export_path}",
            title="导出成功",
            details={
                "文件路径": str(export_path),
                "问题总数": str(len(report.issues)),
                "健康评分": f"{report.score}/100",
            },
        )
    except Exception as e:
        output.error_panel(
            f"导出失败: {e}",
            title="导出错误",
            error_type=type(e).__name__,
            suggestion="检查文件路径是否正确且有写入权限",
        )


def _save_history(report: HealthReport) -> None:
    try:
        history_mgr = AnalysisHistoryManager()
        history_mgr.save_report(report)
    except Exception:
        pass


def _handle_diff_mode(
    output: OutputFormatter,
    performance: bool,
    config: bool,
    errors: bool,
    dependency: bool,
    network: bool,
    resource: bool,
    quality: bool,
    fix: bool,
    export: Optional[str],
    export_html: Optional[str],
) -> None:
    history_mgr = AnalysisHistoryManager()
    previous = history_mgr.get_latest_report()

    try:
        report = asyncio.run(_run_analysis(performance, config, errors, dependency, network, resource, quality))
    except Exception as e:
        output.error_panel(
            f"分析执行失败: {e}",
            title="分析错误",
            error_type=type(e).__name__,
            suggestion="检查系统状态后重试",
        )
        return

    _save_history(report)

    if previous is not None:
        _display_diff(output, report, previous)
    else:
        output.console.print("[dim]未找到历史记录，跳过对比[/dim]")

    categories = []
    if performance:
        categories.append(f"{AnalyzeIcons.PERFORMANCE} 性能")
    if config:
        categories.append(f"{AnalyzeIcons.CONFIG} 配置")
    if errors:
        categories.append(f"{AnalyzeIcons.ERRORS} 可靠性")
    if dependency:
        categories.append(f"{AnalyzeIcons.DEPENDENCY} 依赖")
    if network:
        categories.append(f"{AnalyzeIcons.NETWORK} 网络")
    if resource:
        categories.append(f"{AnalyzeIcons.RESOURCE} 资源")
    if quality:
        categories.append(f"{AnalyzeIcons.QUALITY} 代码质量")

    _display_results(output, report, categories)

    if fix:
        _apply_fixes_interactive(output, report)

    if export:
        _export_json_report(output, report, export)

    if export_html:
        _export_html_report(output, report, export_html, previous=previous)


def _handle_watch_mode(
    output: OutputFormatter,
    interval: int,
    performance: bool,
    config: bool,
    errors: bool,
    dependency: bool,
    network: bool,
    resource: bool,
    quality: bool,
    export_html: Optional[str],
) -> None:
    history_mgr = AnalysisHistoryManager()

    watch_message = (
        f"[bold {BrandColors.ACCENT}]监控模式已启动[/bold {BrandColors.ACCENT}] "
        f"[dim](间隔: {interval}s)[/dim]\n"
        f"[dim]按 Ctrl+C 停止监控[/dim]"
    )
    output.console.print(Panel(watch_message, border_style=BrandColors.ACCENT, box=SIMPLE))
    output.console.print()

    first_run = True
    previous_report: Optional[HealthReport] = None

    def _make_watch_display(
        current_report: HealthReport,
        prev: Optional[HealthReport],
        cycle: int,
    ) -> Panel:
        label, color = _score_label(current_report.score)

        lines = [
            f"[bold]监控周期 #{cycle}[/bold]",
            f"[{color}]健康评分: {current_report.score}/100 ({label})[/{color}]",
            f"问题总数: {len(current_report.issues)}",
        ]

        if prev is not None:
            diff = compute_diff(current_report, prev)
            diff_sign = "+" if diff.score_diff >= 0 else ""
            lines.append(
                f"较上次变化: [{'green' if diff.score_diff >= 0 else 'red'}]"
                f"{diff_sign}{diff.score_diff}[/] 分"
            )
            if diff.new_issues:
                lines.append(f"[red]新增: {len(diff.new_issues)} 项[/red]")
            if diff.fixed_issues:
                lines.append(f"[green]修复: {len(diff.fixed_issues)} 项[/green]")

        return Panel(
            "\n".join(lines),
            title="[bold]实时监控[/bold]",
            border_style=color,
            box=SIMPLE,
        )

    try:
        cycle = 0
        while True:
            try:
                report = asyncio.run(
                    _run_analysis(performance, config, errors, dependency, network, resource, quality)
                )
            except Exception as e:
                output.console.print(f"[red]分析失败: {e}[/red]")
                if first_run:
                    return
                break

            _save_history(report)
            cycle += 1

            if first_run:
                if export_html:
                    _export_html_report(output, report, export_html, previous=None)
                output.console.print(f"[dim]初始分析完成，开始持续监控...[/dim]\n")

            display_panel = _make_watch_display(report, previous_report, cycle)
            output.console.clear()
            output.console.print(display_panel)

            previous_report = report
            first_run = False

            asyncio.run(asyncio.sleep(interval))

    except KeyboardInterrupt:
        output.console.print(f"\n[dim]监控已停止[/dim]")


def _handle_history(
    output: OutputFormatter,
    args: List[str],
) -> None:
    history_mgr = AnalysisHistoryManager()

    if not args or args[0] == "list":
        records = history_mgr.list_reports(limit=30)
        if not records:
            output.console.print("[dim]暂无历史记录[/dim]")
            return

        table = Table(
            show_header=True,
            box=SIMPLE,
            border_style=BrandColors.PRIMARY,
        )
        table.add_column("ID", width=16)
        table.add_column("时间", width=20)
        table.add_column("评分", width=8)
        table.add_column("问题数", width=8)

        for record in records:
            record_id = record.get("id", "-")
            timestamp = record.get("timestamp", "-")
            score = record.get("score", "-")
            issue_count = record.get("issue_count", "-")
            table.add_row(str(record_id), str(timestamp), str(score), str(issue_count))

        output.console.print(table)

    elif args[0] == "show" and len(args) > 1:
        report_id = args[1]
        report = history_mgr.get_report(report_id)
        if report is None:
            output.error(f"未找到记录: {report_id}")
            return

        _display_results(output, report, [])
        output.console.print()
        _export_html_report(output, report, f"analysis_history_{report_id}.html", previous=None)
    else:
        output.error("用法: ppc9 analyze history list|show <id>")


def handle_analyze(
    performance: bool = False,
    config: bool = False,
    errors: bool = False,
    dependency: bool = False,
    network: bool = False,
    resource: bool = False,
    quality: bool = False,
    fix: bool = False,
    export: Optional[str] = None,
    diff: bool = False,
    watch: bool = False,
    interval: int = 60,
    export_html: Optional[str] = None,
    action: Optional[str] = None,
) -> None:
    output = OutputFormatter(verbose=False)

    if action == "history":
        _handle_history(output, [])
        return

    run_all = not any([performance, config, errors, dependency, network, resource, quality])
    if run_all:
        performance = True
        config = True
        errors = True
        dependency = True
        network = True
        resource = True
        quality = True

    is_windows = sys.platform == "win32"
    gear_icon = "⚙" if not is_windows else "[GEAR]"

    output.title(f"{gear_icon} PPC9 系统分析")

    categories = []
    if performance:
        categories.append(f"{AnalyzeIcons.PERFORMANCE} 性能")
    if config:
        categories.append(f"{AnalyzeIcons.CONFIG} 配置")
    if errors:
        categories.append(f"{AnalyzeIcons.ERRORS} 可靠性")
    if dependency:
        categories.append(f"{AnalyzeIcons.DEPENDENCY} 依赖")
    if network:
        categories.append(f"{AnalyzeIcons.NETWORK} 网络")
    if resource:
        categories.append(f"{AnalyzeIcons.RESOURCE} 资源")
    if quality:
        categories.append(f"{AnalyzeIcons.QUALITY} 代码质量")

    output.console.print(f"[dim]分析模块: {' | '.join(categories)}[/dim]\n")

    if watch:
        _handle_watch_mode(
            output,
            interval,
            performance,
            config,
            errors,
            dependency,
            network,
            resource,
            quality,
            export_html,
        )
        return

    if diff:
        _handle_diff_mode(
            output,
            performance,
            config,
            errors,
            dependency,
            network,
            resource,
            quality,
            fix,
            export,
            export_html,
        )
        return

    report = asyncio.run(_run_analysis(performance, config, errors, dependency, network, resource, quality))

    _save_history(report)

    _display_results(output, report, categories)

    if fix:
        _apply_fixes_interactive(output, report)

    if export:
        _export_json_report(output, report, export)

    if export_html:
        _export_html_report(output, report, export_html, previous=None)
