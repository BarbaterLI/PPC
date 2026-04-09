"""split 指令实现 - 冰璃岩开发组 (BLY Team)

将大型文本文件分割成多个章节文件，支持自定义规则和预设配置。
"""

import json
import sys
from pathlib import Path
from typing import Optional, List

from ...config import ConfigManager, get_preset, CustomRule
from ...executor import SplitterExecutor
from ..output import OutputFormatter, BrandColors, Icons
from rich.panel import Panel
from rich.box import ROUNDED


def parse_custom_rules(rules_input: str) -> List[CustomRule]:
    """解析自定义规则输入（JSON字符串或文件路径）"""
    if not rules_input:
        return []

    rules_input = rules_input.strip()

    if rules_input.startswith('['):
        try:
            rules_data = json.loads(rules_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析失败: {e}")
    else:
        path = Path(rules_input)
        if not path.exists():
            raise ValueError(f"规则文件不存在: {path}")
        try:
            with path.open('r', encoding='utf-8') as f:
                rules_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON文件解析失败: {e}")

    if not isinstance(rules_data, list):
        raise ValueError("规则必须是JSON数组格式")

    rules = []
    for i, rule_data in enumerate(rules_data):
        try:
            rule = CustomRule(**rule_data)
            rules.append(rule)
        except Exception as e:
            raise ValueError(f"规则[{i}]配置错误: {e}")

    return rules


def handle_split(
    input_file: Path,
    output_dir: Path,
    preset: str,
    custom_rules: Optional[str] = None,
    add_title_separator: Optional[bool] = None
):
    """处理 split 命令
    
    参数:
        input_file: 输入文件路径
        output_dir: 输出目录路径
        preset: 预设名称
        custom_rules: 自定义规则（JSON 字符串或文件路径）
        add_title_separator: 是否添加标题分隔符
    """
    output = OutputFormatter(verbose=False)
    
    output.show_banner()
    
    output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    output.console.print(f"[bold white]  {Icons.BOOK} PPC8 章节分割[/bold white]")
    output.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n")

    config_manager = ConfigManager()

    if preset != "balanced":
        config = get_preset(preset)
    else:
        config = config_manager.get_config()

    if add_title_separator is not None:
        config.split.add_title_separator = add_title_separator

    config_details = {
        "输入文件": str(input_file),
        "输出目录": str(output_dir),
        "预设": config.split.preset,
        "添加标题分隔符": "是" if config.split.add_title_separator else "否",
    }
    
    if custom_rules:
        try:
            custom_rules_list = parse_custom_rules(custom_rules)
            config_details["自定义规则"] = f"{len(custom_rules_list)} 条"
            
            rules_preview = []
            for rule in custom_rules_list[:3]:
                rules_preview.append(f"{rule.name} (priority={rule.priority})")
            if len(custom_rules_list) > 3:
                rules_preview.append(f"... 还有 {len(custom_rules_list) - 3} 条")
            
            config_details["规则预览"] = ", ".join(rules_preview)
        except ValueError as e:
            output.error_panel(
                f"解析自定义规则失败：{e}",
                title="配置错误",
                error_type="ValueError",
                suggestion="请检查 JSON 格式是否正确或文件路径是否存在"
            )
            sys.exit(1)

    if not input_file.exists():
        output.error_panel(
            f"输入文件不存在：{input_file}",
            title="文件错误",
            error_type="FileNotFoundError",
            suggestion="请检查路径是否正确，或使用绝对路径"
        )
        sys.exit(1)

    if output_dir is None:
        output_dir = input_file.with_name(f"{input_file.stem}_chapters")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    for key, value in config_details.items():
        output.console.print(f"[dim]{key}:[/dim] [cyan]{value}[/cyan]")
    
    output.console.print()

    async def run_split():
        async with SplitterExecutor(config, custom_rules=custom_rules_list if 'custom_rules_list' in locals() else []) as executor:
            result = await executor.execute(input_file, output_dir)

            if result.success:
                chapter_count = len(result.data)
                
                output.console.print(f"\n[bold {BrandColors.SUCCESS}]{'═' * 60}[/bold {BrandColors.SUCCESS}]")
                output.console.print(f"[bold white]  {Icons.SUCCESS} 分割完成报告[/bold white]")
                output.console.print(f"[bold {BrandColors.SUCCESS}]{'═' * 60}[/bold {BrandColors.SUCCESS}]\n")
                
                from rich.table import Table
                from rich.box import SIMPLE
                
                summary_table = Table(show_header=False, box=SIMPLE, border_style=BrandColors.SUCCESS)
                summary_table.add_column("指标", style="bold", width=20)
                summary_table.add_column("值", style="cyan", width=25)
                
                summary_table.add_row("输入文件", input_file.name)
                summary_table.add_row("输出目录", output_dir.name)
                summary_table.add_row("章节数量", f"[{BrandColors.SUCCESS}]{chapter_count}[/{BrandColors.SUCCESS}]")
                summary_table.add_row("总用时", f"{result.metrics.duration_seconds:.1f}s")
                
                if chapter_count > 0:
                    avg_size = sum(f.stat().st_size for f in result.data[:10]) / min(10, chapter_count)
                    summary_table.add_row("平均章节大小", f"{avg_size/1024:.1f} KB")
                
                output.console.print(summary_table)
                output.console.print()
                
                if chapter_count > 0:
                    output.console.print(f"[bold {BrandColors.INFO}]📄 前 5 个章节:[/bold {BrandColors.INFO}]\n")
                    
                    chapters_table = Table(show_header=True, box=SIMPLE, border_style=BrandColors.INFO)
                    chapters_table.add_column("序号", style="yellow", width=6)
                    chapters_table.add_column("文件名", style="white", width=40)
                    chapters_table.add_column("大小", style="cyan", width=12)
                    
                    for i, chapter_file in enumerate(result.data[:5], 1):
                        size_kb = chapter_file.stat().st_size / 1024
                        chapters_table.add_row(
                            str(i),
                            chapter_file.name,
                            f"{size_kb:.1f} KB"
                        )
                    
                    output.console.print(chapters_table)
                    
                    if chapter_count > 5:
                        output.console.print(f"\n[dim]... 还有 {chapter_count - 5} 个章节[/dim]")
                
                output.console.print(f"\n[bold {BrandColors.SUCCESS}]{'─' * 60}[/bold {BrandColors.SUCCESS}]")
                from datetime import datetime
                output.console.print(f"[dim]报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")

            else:
                output.error_panel(
                    f"分割失败：{result.error}",
                    title="分割错误",
                    error_type="SplitError",
                    suggestion="请检查输入文件格式或查看日志获取更多信息"
                )

            return result.success

    try:
        import asyncio
        success = asyncio.run(run_split())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        output.warning_panel(
            "用户中断操作",
            title="中断",
            suggestion="如需继续，请重新运行命令"
        )
        sys.exit(130)
    except Exception as e:
        output.error_panel(
            f"执行失败：{e}",
            title="执行错误",
            error_type=type(e).__name__,
            suggestion="使用 --verbose 参数查看详细错误信息"
        )
        sys.exit(1)
