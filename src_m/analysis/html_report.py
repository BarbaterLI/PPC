"""HTML report generator for PPC10 analysis results.

Produces a single self-contained HTML file with an interactive dark-theme
dashboard showing health scores, issues grouped by category, optional diff
data, and optional history trends.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .diff import DiffResult
from .models import AnalysisCategory, AnalysisIssue, HealthReport, Severity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_COLORS: Dict[Severity, str] = {
    Severity.CRITICAL: "#ff4444",
    Severity.HIGH: "#ff8800",
    Severity.MEDIUM: "#ffcc00",
    Severity.LOW: "#88ccff",
    Severity.INFO: "#aaaaaa",
}

SEVERITY_LABELS: Dict[Severity, str] = {
    Severity.CRITICAL: "严重",
    Severity.HIGH: "高",
    Severity.MEDIUM: "中",
    Severity.LOW: "低",
    Severity.INFO: "信息",
}

CATEGORY_LABELS: Dict[AnalysisCategory, str] = {
    AnalysisCategory.PERFORMANCE: "性能",
    AnalysisCategory.MEMORY: "内存",
    AnalysisCategory.CONFIGURATION: "配置",
    AnalysisCategory.RELIABILITY: "可靠性",
    AnalysisCategory.DEPENDENCY: "依赖",
    AnalysisCategory.NETWORK: "网络",
    AnalysisCategory.RESOURCE: "资源",
    AnalysisCategory.CODE_QUALITY: "代码质量",
    AnalysisCategory.SECURITY: "安全",
    AnalysisCategory.UNKNOWN: "未知",
}

BRAND = "PPC10 Analysis Report"

# ---------------------------------------------------------------------------
# CSS template (inline, single-file)
# ---------------------------------------------------------------------------

_CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;background:#0d1117;color:#e6edf3;line-height:1.6;padding:24px}
.container{max-width:960px;margin:0 auto}
header{text-align:center;padding:32px 16px 24px;border-bottom:1px solid #21262d;margin-bottom:32px}
header h1{font-size:28px;font-weight:700;letter-spacing:-.5px}
header h1 span{color:#58a6ff}
header .subtitle{color:#8b949e;font-size:14px;margin-top:4px}
.score-section{margin-bottom:32px}
.score-card{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:28px;text-align:center}
.score-value{font-size:56px;font-weight:800;line-height:1}
.score-label{font-size:14px;color:#8b949e;margin-top:4px}
.score-bar-wrap{background:#21262d;border-radius:8px;height:16px;margin:20px 0 8px;overflow:hidden;position:relative}
.score-bar-fill{height:100%;border-radius:8px;transition:width 1.2s cubic-bezier(.22,1,.36,1);position:relative;min-width:0}
@keyframes shimmer{0%{background-position:200% 0}to{background-position:-200% 0}}
.score-bar-fill.animating::after{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:linear-gradient(90deg,transparent 0,rgba(255,255,255,.15) 50%,transparent 100%);background-size:200% 100%;animation:shimmer 2s ease-in-out infinite}
.score-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;margin-top:20px}
.stat-item{background:#0d1117;border-radius:8px;padding:12px;text-align:center}
.stat-item .stat-value{font-size:22px;font-weight:700}
.stat-item .stat-label{font-size:12px;color:#8b949e;margin-top:2px}
.section-title{font-size:18px;font-weight:600;margin:32px 0 16px;display:flex;align-items:center;gap:8px}
.section-title svg{flex-shrink:0}
.issue-category{margin-bottom:20px}
.issue-category-header{display:flex;align-items:center;gap:10px;padding:10px 16px;background:#161b22;border:1px solid #21262d;border-radius:8px 8px 0 0;font-weight:600;font-size:15px;cursor:default}
.issue-category-header .count-badge{margin-left:auto;background:#30363d;color:#e6edf3;font-size:12px;font-weight:600;padding:2px 10px;border-radius:10px}
.issue-table{width:100%;border-collapse:collapse;border:1px solid #21262d;border-top:none;border-radius:0 0 8px 8px;overflow:hidden}
.issue-table th,.issue-table td{padding:10px 16px;text-align:left;border-bottom:1px solid #21262d;font-size:13px}
.issue-table th{background:#0d1117;color:#8b949e;font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:.5px}
.issue-table tr:last-child td{border-bottom:none}
.issue-table tr:hover td{background:#1c2128}
.severity-badge{display:inline-block;padding:2px 10px;border-radius:10px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.3px;color:#fff}
.diff-section{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:20px;margin-bottom:24px}
.diff-header{display:flex;align-items:center;gap:12px;margin-bottom:16px}
.diff-score{font-size:32px;font-weight:800}
.diff-score.positive{color:#3fb950}
.diff-score.negative{color:#f85149}
.diff-score.neutral{color:#8b949e}
.diff-summary{color:#8b949e;font-size:13px;padding:8px 12px;background:#0d1117;border-radius:6px;margin-bottom:12px}
.diff-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:600px){.diff-grid{grid-template-columns:1fr}}
.diff-list{background:#0d1117;border-radius:8px;padding:12px}
.diff-list h4{font-size:13px;font-weight:600;margin-bottom:8px;display:flex;align-items:center;gap:6px}
.diff-list ul{list-style:none;padding:0}
.diff-list li{padding:6px 0;font-size:13px;border-bottom:1px solid #21262d;display:flex;align-items:flex-start;gap:6px}
.diff-list li:last-child{border-bottom:none}
.diff-list li .sev-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-top:4px;flex-shrink:0}
.trend-section{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:20px;margin-bottom:24px}
.trend-section h3{font-size:15px;font-weight:600;margin-bottom:16px}
.trend-list{display:flex;flex-direction:column;gap:6px}
.trend-item{display:flex;align-items:center;gap:12px;padding:8px 12px;background:#0d1117;border-radius:6px;font-size:13px}
.trend-item .trend-time{color:#8b949e;min-width:160px}
.trend-item .trend-score{font-weight:700;min-width:40px;text-align:right}
.trend-bar-wrap{flex:1;background:#21262d;border-radius:4px;height:8px;overflow:hidden}
.trend-bar-fill{height:100%;border-radius:4px;transition:width .6s ease}
footer{text-align:center;padding:24px 0;color:#484f58;font-size:12px;border-top:1px solid #21262d;margin-top:40px}
.no-issues{text-align:center;padding:40px 16px;color:#8b949e}
.no-issues svg{margin-bottom:12px}
.metrics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0}
.metrics-card{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px;text-align:center}
.metrics-card .m-value{font-size:20px;font-weight:700;color:#58a6ff}
.metrics-card .m-label{font-size:11px;color:#8b949e;margin-top:2px}
"""

# ---------------------------------------------------------------------------
# SVG icons (inline)
# ---------------------------------------------------------------------------

_ICONS = {
    "health": (
        '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#3fb950" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>'
        '</svg>'
    ),
    "issues": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f0883e" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>'
        '<line x1="12" y1="16" x2="12.01" y2="16"/>'
        '</svg>'
    ),
    "trend": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8b949e" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>'
        '<polyline points="17 6 23 6 23 12"/>'
        '</svg>'
    ),
    "diff": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8b949e" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/>'
        '</svg>'
    ),
    "check": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3fb950" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
        '<polyline points="22 4 12 14.01 9 11.01"/>'
        '</svg>'
    ),
    "clock": (
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#8b949e" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'
        '</svg>'
    ),
    "category": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>'
        '<rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>'
        '</svg>'
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(text: object) -> str:
    """Return text safe for embedding in HTML."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _severity_color(severity: Severity) -> str:
    return SEVERITY_COLORS.get(severity, "#aaaaaa")


def _score_color(score: int) -> str:
    if score >= 80:
        return "#3fb950"
    if score >= 60:
        return "#d29922"
    if score >= 40:
        return "#f0883e"
    return "#f85149"


def _format_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _severity_badge(severity: Severity) -> str:
    color = _severity_color(severity)
    label = SEVERITY_LABELS.get(severity, severity.value)
    return f'<span class="severity-badge" style="background:{color}">{_esc(label)}</span>'


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_score_section(report: HealthReport) -> str:
    score = report.score
    color = _score_color(score)
    total_issues = len(report.issues)
    critical_count = report.critical_count()
    high_count = report.high_count()
    now_str = _format_dt(report.timestamp)

    return f"""\
<div class="score-section">
  <div class="score-card">
    <div class="score-value" style="color:{color}">{score}</div>
    <div class="score-label">健康评分</div>
    <div class="score-bar-wrap">
      <div class="score-bar-fill animating" style="width:{score}%;background:{color}"></div>
    </div>
    <div class="score-stats">
      <div class="stat-item">
        <div class="stat-value" style="color:{color}">{total_issues}</div>
        <div class="stat-label">问题总数</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" style="color:#ff4444">{critical_count}</div>
        <div class="stat-label">严重</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" style="color:#ff8800">{high_count}</div>
        <div class="stat-label">高</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" style="color:#8b949e">{_esc(now_str)}</div>
        <div class="stat-label">分析时间</div>
      </div>
    </div>
  </div>
</div>"""


def _build_diff_section(diff: DiffResult) -> str:
    score_diff = diff.score_diff
    if score_diff > 0:
        diff_class = "positive"
        prefix = "+"
    elif score_diff < 0:
        diff_class = "negative"
        prefix = ""
    else:
        diff_class = "neutral"
        prefix = ""

    rows_new = ""
    for issue in diff.new_issues:
        color = _severity_color(issue.severity)
        rows_new += (
            f'<li><span class="sev-dot" style="background:{color}"></span>'
            f'<span>{_esc(issue.description)}</span></li>'
        )

    rows_fixed = ""
    for issue in diff.fixed_issues:
        color = _severity_color(issue.severity)
        rows_fixed += (
            f'<li><span class="sev-dot" style="background:{color}"></span>'
            f'<span>{_esc(issue.description)}</span></li>'
        )

    if not rows_new:
        rows_new = '<li style="color:#8b949e">无新增问题</li>'
    if not rows_fixed:
        rows_fixed = '<li style="color:#8b949e">无已修复问题</li>'

    return f"""\
<div class="diff-section">
  <div class="diff-header">
    {_ICONS["diff"]}
    <span class="diff-score {diff_class}">{prefix}{score_diff}</span>
    <span style="color:#8b949e;font-size:14px">较上次分数</span>
  </div>
  {f'<div class="diff-summary">{_esc(diff.summary)}</div>' if diff.summary else ''}
  <div class="diff-grid">
    <div class="diff-list">
      <h4>{_ICONS["issues"]} 新增问题 ({len(diff.new_issues)})</h4>
      <ul>{rows_new}</ul>
    </div>
    <div class="diff-list">
      <h4>{_ICONS["check"]} 已修复 ({len(diff.fixed_issues)})</h4>
      <ul>{rows_fixed}</ul>
    </div>
  </div>
</div>"""


def _build_trend_section(history: List[Dict[str, Any]]) -> str:
    items_html = ""
    for entry in history:
        score = entry.get("score", 0)
        ts = entry.get("timestamp", "")
        if isinstance(score, int):
            color = _score_color(score)
        else:
            color = "#8b949e"
        try:
            dt = datetime.fromisoformat(ts) if isinstance(ts, str) else datetime.min
            time_str = _format_dt(dt)
        except (ValueError, TypeError):
            time_str = str(ts)
        items_html += f"""\
    <div class="trend-item">
      <span class="trend-time">{_esc(time_str)}</span>
      <span class="trend-score" style="color:{color}">{score}</span>
      <div class="trend-bar-wrap">
        <div class="trend-bar-fill" style="width:{score}%;background:{color}"></div>
      </div>
    </div>"""

    return f"""\
<div class="trend-section">
  <h3>{_ICONS["trend"]} 历史趋势 ({len(history)} 条记录)</h3>
  <div class="trend-list">{items_html}</div>
</div>"""


def _build_issues_section(report: HealthReport) -> str:
    if not report.issues:
        return f"""\
<div class="no-issues">
  {_ICONS["check"]}
  <p>未发现任何问题</p>
</div>"""

    grouped: Dict[AnalysisCategory, List[AnalysisIssue]] = report.issues_by_category()
    sections = ""

    for category in AnalysisCategory:
        issues = grouped.get(category)
        if not issues:
            continue
        label = CATEGORY_LABELS.get(category, category.value)
        rows = ""
        for issue in issues:
            badge = _severity_badge(issue.severity)
            suggestion = _esc(issue.suggestion) if issue.suggestion else '<span style="color:#484f58">无建议</span>'
            location = _esc(issue.location) if issue.location else '<span style="color:#484f58">-</span>'
            rows += f"""\
          <tr>
            <td>{badge}</td>
            <td style="color:#8b949e;font-family:monospace;font-size:12px">{location}</td>
            <td>{_esc(issue.description)}</td>
            <td>{suggestion}</td>
          </tr>"""

        sections += f"""\
    <div class="issue-category">
      <div class="issue-category-header">
        {_ICONS["category"]}
        {_esc(label)}
        <span class="count-badge">{len(issues)} 项</span>
      </div>
      <table class="issue-table">
        <thead>
          <tr>
            <th style="width:70px">级别</th>
            <th style="width:130px">位置</th>
            <th>描述</th>
            <th>建议</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>"""

    return sections


def _build_metrics_section(report: HealthReport) -> str:
    if not report.metrics:
        return ""
    cards = ""
    for key, value in report.metrics.items():
        if isinstance(value, float):
            display = f"{value:.2f}"
        else:
            display = str(value)
        cards += f"""\
    <div class="metrics-card">
      <div class="m-value">{_esc(display)}</div>
      <div class="m-label">{_esc(key)}</div>
    </div>"""
    return f"""\
<div class="section-title">{_ICONS["health"]} 指标</div>
<div class="metrics-grid">{cards}</div>"""


def _build_summary_section(report: HealthReport) -> str:
    parts = []
    if report.component:
        parts.append(f"组件: {_esc(report.component)}")
    parts.append(f"评分: {report.score}/100")
    parts.append(f"问题: {len(report.issues)} 个")
    parts.append(f"时间: {_format_dt(report.timestamp)}")
    summary_text = f" — {_esc(report.summary)}" if report.summary else ""
    return f"""\
<div style="background:#161b22;border:1px solid #21262d;border-radius:12px;padding:20px;margin-bottom:24px">
  <div style="display:flex;flex-wrap:wrap;gap:16px;align-items:center">
    <span style="font-size:14px;color:#8b949e">{' · '.join(parts)}{summary_text}</span>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class HTMLReportGenerator:
    """Generates a self-contained interactive HTML analysis report."""

    def __init__(self) -> None:
        pass

    def generate(
        self,
        report: HealthReport,
        output_path: Union[str, Path],
        diff: Optional[DiffResult] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Path:
        """Generate an HTML report and write it to *output_path*.

        Args:
            report: The health report to render.
            output_path: Destination file path.
            diff: Optional diff result for score change visualisation.
            history: Optional list of historical report dicts (each must
                     contain at least ``score`` and ``timestamp`` keys).

        Returns:
            The absolute path of the generated HTML file.
        """
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        score_section = _build_score_section(report)
        summary_section = _build_summary_section(report)
        issues_section = _build_issues_section(report)
        metrics_section = _build_metrics_section(report)

        diff_section = ""
        if diff is not None:
            diff_section = _build_diff_section(diff)

        trend_section = ""
        if history:
            trend_section = _build_trend_section(history)

        component_info = ""
        if report.component:
            component_info = f" &mdash; {_esc(report.component)}"

        html = f"""\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{BRAND}{component_info}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
  <header>
    <h1><span>PPC10</span> 分析报告</h1>
    <div class="subtitle">生成于{_format_dt(report.timestamp)}{component_info}</div>
  </header>
  {score_section}
  {summary_section}
  {diff_section}
  {trend_section}
  {metrics_section}
  <div class="section-title">{_ICONS["issues"]} 问题明细</div>
  {issues_section}
  <footer>PPC10 Analysis Engine &mdash; 自包含 HTML 报告</footer>
</div>
</body>
</html>"""

        path.write_text(html, encoding="utf-8")
        return path
