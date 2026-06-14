"""分析命令 - 统一系统健康检查与深度分析。

提供两种模式：
- 默认（不带 --deep）：执行轻量级系统健康检查
  （系统环境、依赖、网络、文件系统、系统资源、配置验证），
  支持 --fix 交互式一键修复和 --export 导出 JSON 报告。
- 深度模式（--deep）：对性能瓶颈、配置冲突、错误模式、依赖、网络、
  资源和代码质量进行深度分析，生成健康评分和修复建议。
两种模式可同时启用（既运行健康检查又运行深度分析）。
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.box import SIMPLE
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from ..output import OutputFormatter, Icons, BrandColors
from ...config.manager import get_default_config_dir
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
from ...analysis.diff import compute_diff
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

    SUCCESS = "+"
    ERROR = "-"
    WARNING = "!"
    INFO = "i"

    SYSTEM_ENV = "[ENV]"
    DEPENDENCIES = "[DEP]"
    NETWORK_CHECK = "[NET]"
    FILESYSTEM = "[DIR]"
    SYSTEM_RESOURCES = "[RES]"
    CONFIG_CHECK = "[CFG]"

    PYTHON = "PY"
    OS = "OS"
    ARCH = "ARC"
    VENV = "VEN"
    PACKAGE = "PKG"
    TTS_SERVICE = "TTS"
    API = "API"
    CONFIG_DIR = "CFGD"
    CONFIG_FILE = "CFGF"
    OUTPUT_DIR = "OUTD"
    DISK = "DSK"
    PERMISSION = "PRM"
    CPU = "CPU"
    MEMORY = "MEM"
    CPU_USAGE = "CPUU"
    TTS_VOICE = "TTSS"
    CONCURRENCY = "CONC"
    RETRY = "RET"
    TEXT_NORM = "TXN"


class HealthCheckCategory:
    """健康检查分类定义。"""
    SYSTEM_ENV = "system_env"
    DEPENDENCIES = "dependencies"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    SYSTEM_RESOURCES = "system_resources"
    CONFIG = "config"


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


# -----------------------------------------------------------------------------
# 深度分析（原有）显示与修复逻辑
# -----------------------------------------------------------------------------

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
    output.console.print(f"[bold white]  {Icons.CHART} 深度分析结果[/bold white]")
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
    report_or_results: Any,
    export: str,
) -> None:
    try:
        export_path = Path(export)
        if not export_path.suffix:
            export_path = export_path.with_suffix(".json")

        if isinstance(report_or_results, HealthReport):
            data = report_or_results.to_dict()
            extra = {
                "问题总数": str(len(report_or_results.issues)),
                "健康评分": f"{report_or_results.score}/100",
            }
        else:
            data = report_or_results
            summary = data.get("summary", {}) if isinstance(data, dict) else {}
            extra = {
                "检查项数": str(summary.get("total", "-")),
                "通过率": f"{summary.get('pass_rate', 0):.1f}%" if isinstance(summary, dict) else "-",
            }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        output.success_panel(
            f"报告已导出: {export_path}",
            title="导出成功",
            details={
                "文件路径": str(export_path),
                **extra,
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


# -----------------------------------------------------------------------------
# 健康检查（来自 check.py）逻辑
# -----------------------------------------------------------------------------

class _SystemChecker:
    """系统检查器 - 执行系统诊断检查（轻量级健康检查模式）。"""

    def __init__(self, output: OutputFormatter):
        self.output = output
        self.results: Dict[str, List[Dict]] = {
            HealthCheckCategory.SYSTEM_ENV: [],
            HealthCheckCategory.DEPENDENCIES: [],
            HealthCheckCategory.NETWORK: [],
            HealthCheckCategory.FILESYSTEM: [],
            HealthCheckCategory.SYSTEM_RESOURCES: [],
            HealthCheckCategory.CONFIG: [],
        }
        self.fix_suggestions: Dict[str, List[str]] = {}

    def add_result(
        self,
        category: str,
        name: str,
        status: bool,
        detail: str,
        icon: str = "",
        suggestion: Optional[str] = None,
    ):
        self.results[category].append({
            "name": name,
            "status": status,
            "detail": detail,
            "icon": icon,
        })

        if not status and suggestion:
            key = f"{category}:{name}"
            self.fix_suggestions[key] = suggestion

    def check_system_environment(self):
        self.output.title(f"{Icons.GEAR} 系统环境检查")

        python_version = platform.python_version()
        python_ok = sys.version_info >= (3, 8)
        self.add_result(
            HealthCheckCategory.SYSTEM_ENV,
            "Python 版本",
            python_ok,
            f"{python_version} (要求：3.8+)",
            AnalyzeIcons.PYTHON,
            "请升级 Python 到 3.8 或更高版本",
        )

        os_info = f"{platform.system()} {platform.release()}"
        self.add_result(
            HealthCheckCategory.SYSTEM_ENV,
            "操作系统",
            True,
            os_info,
            AnalyzeIcons.OS,
        )

        arch = platform.machine()
        self.add_result(
            HealthCheckCategory.SYSTEM_ENV,
            "系统架构",
            True,
            f"{arch} ({platform.architecture()[0]})",
            AnalyzeIcons.ARCH,
        )

        in_venv = sys.prefix != sys.base_prefix
        venv_status = "已激活" if in_venv else "未激活"
        self.add_result(
            HealthCheckCategory.SYSTEM_ENV,
            "虚拟环境",
            True,
            venv_status,
            AnalyzeIcons.VENV,
            None,
        )

    def check_dependencies(self):
        self.output.title(f"{Icons.BOOK} 依赖包检查")

        required_deps = {
            "typer": ("Typer", AnalyzeIcons.PACKAGE, "命令行框架"),
            "rich": ("Rich", AnalyzeIcons.PACKAGE, "终端美化库"),
            "edge_tts": ("Edge TTS", AnalyzeIcons.PACKAGE, "TTS 引擎"),
            "pydub": ("PyDub", AnalyzeIcons.PACKAGE, "音频处理"),
        }

        for pkg_name, (display_name, icon, desc) in required_deps.items():
            try:
                import importlib
                module = importlib.import_module(pkg_name)

                try:
                    version = getattr(module, "__version__", "unknown")
                    if version == "unknown":
                        from importlib.metadata import version as _pkg_version
                        version = _pkg_version(pkg_name)
                except Exception:
                    version = "已安装"

                self.add_result(
                    HealthCheckCategory.DEPENDENCIES,
                    display_name,
                    True,
                    f"{version} - {desc}",
                    icon,
                )
            except ImportError:
                self.add_result(
                    HealthCheckCategory.DEPENDENCIES,
                    display_name,
                    False,
                    "未安装",
                    icon,
                    f"运行 pip install {pkg_name} 安装",
                )

        optional_deps = {
            "psutil": ("PSUtil", AnalyzeIcons.PACKAGE, "系统资源监控"),
        }

        self.output.info("\n可选依赖:")
        for pkg_name, (display_name, icon, desc) in optional_deps.items():
            try:
                import importlib
                module = importlib.import_module(pkg_name)

                try:
                    version = getattr(module, "__version__", "unknown")
                    if version == "unknown":
                        from importlib.metadata import version as _pkg_version
                        version = _pkg_version(pkg_name)
                except Exception:
                    version = "已安装"

                self.add_result(
                    HealthCheckCategory.DEPENDENCIES,
                    display_name,
                    True,
                    f"{version} - {desc}",
                    icon,
                )
            except ImportError:
                self.add_result(
                    HealthCheckCategory.DEPENDENCIES,
                    display_name,
                    False,
                    "未安装（可选）",
                    icon,
                    f"运行 pip install {pkg_name} 安装（可选）",
                )

    async def _check_tts_service(self) -> int:
        try:
            import edge_tts
            voices = await edge_tts.list_voices()
            return len(voices)
        except Exception:
            return 0

    def check_network_connectivity(self):
        self.output.title(f"{Icons.LINK} 网络连通性检查")

        try:
            voice_count = asyncio.run(self._check_tts_service())
            tts_ok = voice_count > 0
            self.add_result(
                HealthCheckCategory.NETWORK,
                "TTS 服务",
                tts_ok,
                f"{'正常' if tts_ok else '异常'} - {voice_count} 个语音" if tts_ok else "无法连接",
                AnalyzeIcons.TTS_SERVICE,
                "检查网络连接或代理设置" if not tts_ok else None,
            )
        except Exception as e:
            self.add_result(
                HealthCheckCategory.NETWORK,
                "TTS 服务",
                False,
                f"检查失败：{str(e)}",
                AnalyzeIcons.TTS_SERVICE,
                "检查网络连接或稍后重试",
            )

        self.add_result(
            HealthCheckCategory.NETWORK,
            "API 端点",
            True,
            "可达（模拟检查）",
            AnalyzeIcons.API,
        )

    def _get_config_dir(self) -> Path:
        return get_default_config_dir()

    def check_filesystem(self):
        self.output.title(f"{Icons.FOLDER} 文件系统检查")

        config_dir = self._get_config_dir()
        config_exists = config_dir.exists()
        self.add_result(
            HealthCheckCategory.FILESYSTEM,
            "配置目录",
            config_exists,
            str(config_dir),
            AnalyzeIcons.CONFIG_DIR,
            "可运行 'ppc10 config init' 创建配置目录" if not config_exists else None,
        )

        config_file = config_dir / "config.yml"
        config_file_exists = config_file.exists()
        self.add_result(
            HealthCheckCategory.FILESYSTEM,
            "配置文件",
            config_file_exists,
            str(config_file),
            AnalyzeIcons.CONFIG_FILE,
            "可运行 'ppc10 config init' 创建配置文件" if not config_file_exists else None,
        )

        output_dir = Path.cwd() / "output"
        output_exists = output_dir.exists()
        self.add_result(
            HealthCheckCategory.FILESYSTEM,
            "输出目录",
            output_exists,
            str(output_dir),
            AnalyzeIcons.OUTPUT_DIR,
            f"运行 mkdir {output_dir} 创建目录" if not output_exists else None,
        )

        try:
            import shutil
            total, used, free = shutil.disk_usage(str(Path.home()))
            free_gb = free / (1024 ** 3)
            disk_ok = free_gb > 1.0
            self.add_result(
                HealthCheckCategory.FILESYSTEM,
                "磁盘空间",
                disk_ok,
                f"可用：{free_gb:.2f} GB",
                AnalyzeIcons.DISK,
                "清理磁盘空间以确保正常运行" if not disk_ok else None,
            )
        except Exception as e:
            self.add_result(
                HealthCheckCategory.FILESYSTEM,
                "磁盘空间",
                False,
                f"检查失败：{str(e)}",
                AnalyzeIcons.DISK,
            )

        try:
            config_dir.mkdir(parents=True, exist_ok=True)
            test_file = config_dir / ".permission_test"
            test_file.touch(exist_ok=True)
            test_file.unlink()
            permission_ok = True
        except PermissionError:
            permission_ok = False
        except Exception:
            permission_ok = False

        self.add_result(
            HealthCheckCategory.FILESYSTEM,
            "目录权限",
            permission_ok,
            "正常" if permission_ok else "权限不足",
            AnalyzeIcons.PERMISSION,
            "以管理员权限运行或修改目录权限" if not permission_ok else None,
        )

    def check_system_resources(self):
        self.output.title(f"{Icons.CHART} 系统资源检查")

        cpu_count = os.cpu_count() or 1
        self.add_result(
            HealthCheckCategory.SYSTEM_RESOURCES,
            "CPU 核心",
            True,
            f"{cpu_count} 核心",
            AnalyzeIcons.CPU,
        )

        try:
            import psutil
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024 ** 3)
            available_gb = memory.available / (1024 ** 3)
            memory_ok = available_gb > 1.0
            self.add_result(
                HealthCheckCategory.SYSTEM_RESOURCES,
                "系统内存",
                memory_ok,
                f"总计：{total_gb:.2f} GB, 可用：{available_gb:.2f} GB",
                AnalyzeIcons.MEMORY,
                "关闭不必要的程序释放内存" if not memory_ok else None,
            )
        except ImportError:
            self.add_result(
                HealthCheckCategory.SYSTEM_RESOURCES,
                "系统内存",
                False,
                "无法检测（未安装 psutil）",
                AnalyzeIcons.MEMORY,
                "运行 pip install psutil 安装",
            )

        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.5)
            cpu_ok = cpu_percent < 90
            self.add_result(
                HealthCheckCategory.SYSTEM_RESOURCES,
                "CPU 使用率",
                cpu_ok,
                f"{cpu_percent:.1f}%",
                AnalyzeIcons.CPU_USAGE,
                "关闭高负载程序" if not cpu_ok else None,
            )
        except ImportError:
            pass

    def check_config(self):
        self.output.title(f"{Icons.GEAR} 配置验证")

        try:
            from ...config.manager import ConfigManager
            config_manager = ConfigManager()

            try:
                config = config_manager.get_config()
                config_ok = True
                config_detail = f"版本：{config.version}"
            except Exception as e:
                config_ok = False
                config_detail = f"加载失败：{str(e)}"

            self.add_result(
                HealthCheckCategory.CONFIG,
                "配置加载",
                config_ok,
                config_detail,
                AnalyzeIcons.SUCCESS,
                "运行 'ppc10 config init' 初始化配置" if not config_ok else None,
            )

            if config_ok:
                tts_voice = config.tts.voice
                voice_ok = bool(tts_voice)
                self.add_result(
                    HealthCheckCategory.CONFIG,
                    "TTS 语音",
                    voice_ok,
                    tts_voice if voice_ok else "未设置",
                    AnalyzeIcons.TTS_VOICE,
                    "运行 'ppc10 config set tts.voice <语音名>' 设置" if not voice_ok else None,
                )

                concurrency = config.tts.concurrency
                concurrency_ok = 1 <= concurrency <= 10
                self.add_result(
                    HealthCheckCategory.CONFIG,
                    "并发数",
                    concurrency_ok,
                    str(concurrency),
                    AnalyzeIcons.CONCURRENCY,
                    "并发数应在 1-10 之间" if not concurrency_ok else None,
                )

                try:
                    tts_retries = config.tts.retries
                    reliability_retries = config.reliability.tts_retry.max_retries
                    retries_ok = tts_retries >= 0 and reliability_retries >= 0
                    self.add_result(
                        HealthCheckCategory.CONFIG,
                        "TTS 重试次数",
                        retries_ok,
                        f"TTS={tts_retries}, 可靠性={reliability_retries}",
                        AnalyzeIcons.RETRY,
                    )
                except AttributeError as e:
                    self.add_result(
                        HealthCheckCategory.CONFIG,
                        "TTS 重试次数",
                        False,
                        f"配置错误：{str(e)}",
                        AnalyzeIcons.RETRY,
                    )

                text_norm = config.tts.text_normalization
                norm_enabled = text_norm.enable_text_normalization
                self.add_result(
                    HealthCheckCategory.CONFIG,
                    "文本规范化",
                    True,
                    f"{'启用' if norm_enabled else '禁用'}",
                    AnalyzeIcons.TEXT_NORM,
                )

        except ImportError:
            self.add_result(
                HealthCheckCategory.CONFIG,
                "配置模块",
                False,
                "无法导入配置模块",
                AnalyzeIcons.ERROR,
            )
        except Exception as e:
            self.add_result(
                HealthCheckCategory.CONFIG,
                "配置检查",
                False,
                f"检查失败：{str(e)}",
                AnalyzeIcons.ERROR,
                "查看详细日志获取更多信息",
            )

    def get_all_results(self) -> Dict[str, Any]:
        all_checks = []
        for category_results in self.results.values():
            all_checks.extend(category_results)

        total = len(all_checks)
        passed = sum(1 for c in all_checks if c["status"])
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": round(pass_rate, 2),
            },
            "categories": {
                name: results for name, results in self.results.items()
            },
            "suggestions": self.fix_suggestions,
        }


def _run_health_check(
    output: OutputFormatter,
    full: bool = False,
) -> Dict[str, Any]:
    """运行轻量级健康检查，返回结果字典。"""
    is_windows = sys.platform == "win32"
    gear_icon = "⚙" if not is_windows else "[GEAR]"

    output.title(f"{gear_icon} PPC10 系统健康检查")

    checker = _SystemChecker(output)
    checker.check_system_environment()
    checker.check_dependencies()
    checker.check_network_connectivity()
    checker.check_filesystem()
    checker.check_system_resources()
    checker.check_config()

    results = checker.get_all_results()

    output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    output.console.print(f"[bold white]  {Icons.CHART} 检查结果汇总[/bold white]")
    output.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n")

    for category_name, category_results in checker.results.items():
        if not category_results:
            continue

        category_labels = {
            HealthCheckCategory.SYSTEM_ENV: "系统环境",
            HealthCheckCategory.DEPENDENCIES: "依赖包",
            HealthCheckCategory.NETWORK: "网络连通性",
            HealthCheckCategory.FILESYSTEM: "文件系统",
            HealthCheckCategory.SYSTEM_RESOURCES: "系统资源",
            HealthCheckCategory.CONFIG: "配置验证",
        }

        category_icons = {
            HealthCheckCategory.SYSTEM_ENV: AnalyzeIcons.SYSTEM_ENV,
            HealthCheckCategory.DEPENDENCIES: AnalyzeIcons.DEPENDENCIES,
            HealthCheckCategory.NETWORK: AnalyzeIcons.NETWORK_CHECK,
            HealthCheckCategory.FILESYSTEM: AnalyzeIcons.FILESYSTEM,
            HealthCheckCategory.SYSTEM_RESOURCES: AnalyzeIcons.SYSTEM_RESOURCES,
            HealthCheckCategory.CONFIG: AnalyzeIcons.CONFIG_CHECK,
        }

        checks = [
            {
                "name": item["name"],
                "status": item["status"],
                "detail": item["detail"],
                "icon": item["icon"],
            }
            for item in category_results
        ]

        label = category_labels.get(category_name, category_name)
        icon = category_icons.get(category_name, "")
        output.check_result_enhanced(checks, title=f"{icon} {label}", show_summary=False)
        output.console.print()

    summary = results["summary"]
    pass_rate = summary["pass_rate"]

    if pass_rate == 100:
        summary_color = BrandColors.SUCCESS
        summary_icon = Icons.SUCCESS
        summary_text = "优秀"
    elif pass_rate >= 70:
        summary_color = BrandColors.WARNING
        summary_icon = Icons.WARNING
        summary_text = "良好"
    else:
        summary_color = BrandColors.ERROR
        summary_icon = Icons.ERROR
        summary_text = "需改进"

    summary_panel = Panel(
        f"[bold]总计:[/bold] {summary['total']}  "
        f"[{BrandColors.SUCCESS}]通过:[/{BrandColors.SUCCESS}] {summary['passed']}  "
        f"[{BrandColors.ERROR}]失败:[/{BrandColors.ERROR}] {summary['failed']}  "
        f"[bold {summary_color}]通过率:[/bold {summary_color}] {pass_rate:.1f}%  "
        f"[bold {summary_color}]{summary_icon} 评价：{summary_text}[/bold {summary_color}]",
        title="[bold]检查汇总[/bold]",
        border_style=summary_color,
        box=SIMPLE,
    )
    output.console.print(summary_panel)

    if checker.fix_suggestions:
        output.console.print(f"\n[bold {BrandColors.ACCENT}]{'─' * 60}[/bold {BrandColors.ACCENT}]")
        output.console.print(f"[bold white]  {Icons.INFO} 修复建议[/bold white]")
        output.console.print(f"[bold {BrandColors.ACCENT}]{'─' * 60}[/bold {BrandColors.ACCENT}]\n")

        for key, suggestion in checker.fix_suggestions.items():
            category, name = key.split(":", 1)
            output.console.print(f"[yellow]{AnalyzeIcons.WARNING} {name}:[/yellow]")
            output.console.print(f"  [green]→ {suggestion}[/green]\n")

    return {
        "results": results,
        "checker": checker,
    }


def _apply_health_fixes(
    output: OutputFormatter,
    checker: _SystemChecker,
) -> int:
    """交互式应用健康检查发现的可一键修复项。"""
    try:
        if output.console.is_terminal and not Confirm.ask(
            f"\n[{BrandColors.INFO}]是否执行一键修复？[/{BrandColors.INFO}]",
            default=False,
        ):
            output.console.print("[dim]已跳过自动修复[/dim]")
            return 0
    except Exception:
        pass

    output.console.print(f"\n[bold {BrandColors.PRIMARY}]执行修复...[/bold {BrandColors.PRIMARY}]\n")

    fixed_count = 0
    for key, suggestion in checker.fix_suggestions.items():
        category, name = key.split(":", 1)

        if name == "配置目录":
            config_dir = checker._get_config_dir()
            try:
                config_dir.mkdir(parents=True, exist_ok=True)
                output.success(f"已创建配置目录：{config_dir}")
                fixed_count += 1
            except Exception as e:
                output.error(f"创建配置目录失败：{e}")

        elif name == "配置文件":
            config_dir = checker._get_config_dir()
            config_file = config_dir / "config.yml"
            try:
                config_dir.mkdir(parents=True, exist_ok=True)
                from ...config.presets import COMMENTED_YAML_TEMPLATE
                with open(config_file, "w", encoding="utf-8") as f:
                    f.write(COMMENTED_YAML_TEMPLATE)
                output.success(f"已创建配置文件：{config_file}")
                fixed_count += 1
            except Exception as e:
                output.error(f"创建配置文件失败：{e}")

        elif name == "输出目录":
            output_dir = Path.cwd() / "output"
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                output.success(f"已创建输出目录：{output_dir}")
                fixed_count += 1
            except Exception as e:
                output.error(f"创建输出目录失败：{e}")

    output.console.print(
        f"\n[bold {BrandColors.SUCCESS}]完成修复：{fixed_count} 项[/bold {BrandColors.SUCCESS}]"
    )
    return fixed_count


def handle_analyze(
    deep: bool = False,
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
    full: bool = False,
) -> None:
    """统一处理 analyze 命令 - 健康检查 + 深度分析。

    默认（不传 --deep）运行轻量级系统健康检查；
    --deep 启用深度分析；两者可叠加。
    """
    output = OutputFormatter(verbose=False)

    deep_flags_set = any(
        [performance, config, errors, dependency, network, resource, quality, diff, watch]
    )
    run_deep = deep or deep_flags_set

    if not run_deep:
        info = _run_health_check(output, full=full)
        results = info["results"]
        checker = info["checker"]

        if fix:
            _apply_health_fixes(output, checker)

        if export:
            _export_json_report(output, results, export)

        return

    run_all = not any(
        [performance, config, errors, dependency, network, resource, quality]
    )
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

    output.title(f"{gear_icon} PPC10 系统深度分析")

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

    report = asyncio.run(
        _run_analysis(performance, config, errors, dependency, network, resource, quality)
    )

    _save_history(report)

    _display_results(output, report, categories)

    if fix:
        _apply_fixes_interactive(output, report)

    if export:
        _export_json_report(output, report, export)

    if export_html:
        _export_html_report(output, report, export_html, previous=None)
