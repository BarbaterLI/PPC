"""batch 指令实现 - 冰璃岩开发组 (BLY Team)

批量归档和管理文件，支持批次规划、预览模式和自动分组功能。
"""

import sys
from pathlib import Path

from ...config import ConfigManager, get_preset
from ..executors import BatcherExecutor
from ..output import OutputFormatter, BrandColors, Icons
from rich.table import Table
from rich.box import SIMPLE, ROUNDED


def handle_batch(
    source_dir: Path,
    batch_size: int,
    dry_run: bool
):
    """处理 batch 命令
    
    参数:
        source_dir: 源目录路径
        batch_size: 批次大小
        dry_run: 是否仅预览
    """
    output = OutputFormatter(verbose=False)
    
    output.show_banner()
    
    output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    output.console.print(f"[bold white]  {Icons.FOLDER} PPC8 批量归档[/bold white]")
    output.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n")

    config_manager = ConfigManager()
    config = config_manager.get_config()

    if batch_size:
        config.batch.max_files_per_batch = batch_size

    config_details = {
        "源目录": str(source_dir),
        "批次大小": f"{config.batch.max_files_per_batch} 文件",
        "模式": "预览" if dry_run else "执行",
    }
    
    for key, value in config_details.items():
        output.console.print(f"[dim]{key}:[/dim] [cyan]{value}[/cyan]")
    
    output.console.print()

    if not source_dir.exists():
        output.error_panel(
            f"源目录不存在：{source_dir}",
            title="目录错误",
            error_type="FileNotFoundError",
            suggestion="请检查路径是否正确，或使用绝对路径"
        )
        sys.exit(1)

    async def run_batch():
        async with BatcherExecutor(config) as executor:
            if dry_run:
                result = await executor.dry_run(source_dir)
            else:
                output_dir = source_dir / "batches"
                result = await executor.execute(source_dir, output_dir)

            if result.success:
                batches = result.data
                batch_count = len(batches)
                
                output.console.print(f"\n[bold {BrandColors.SUCCESS}]{'═' * 60}[/bold {BrandColors.SUCCESS}]")
                output.console.print(f"[bold white]  {Icons.SUCCESS} 批次规划报告[/bold white]")
                output.console.print(f"[bold {BrandColors.SUCCESS}]{'═' * 60}[/bold {BrandColors.SUCCESS}]\n")
                
                from datetime import datetime
                
                summary_table = Table(show_header=False, box=SIMPLE, border_style=BrandColors.SUCCESS)
                summary_table.add_column("指标", style="bold", width=20)
                summary_table.add_column("值", style="cyan", width=25)
                
                total_files = sum(len(batch.files) for batch in batches)
                total_size = sum(batch.total_size for batch in batches)
                
                summary_table.add_row("批次数量", f"[{BrandColors.SUCCESS}]{batch_count}[/{BrandColors.SUCCESS}]")
                summary_table.add_row("文件总数", str(total_files))
                summary_table.add_row("总大小", f"{total_size/1024/1024:.2f} MB")
                summary_table.add_row("平均每批文件", f"{total_files/batch_count:.1f}" if batch_count > 0 else "0")
                summary_table.add_row("总用时", f"{result.metrics.duration_seconds:.1f}s")
                
                output.console.print(summary_table)
                output.console.print()
                
                if batch_count > 0:
                    output.console.print(f"[bold {BrandColors.INFO}]📦 批次详情 (前 10 个):[/bold {BrandColors.INFO}]\n")
                    
                    batches_table = Table(show_header=True, box=SIMPLE, border_style=BrandColors.INFO)
                    batches_table.add_column("序号", style="yellow", width=6)
                    batches_table.add_column("批次名", style="white", width=25)
                    batches_table.add_column("文件数", style="cyan", width=10, justify="right")
                    batches_table.add_column("大小", style="cyan", width=12, justify="right")
                    
                    for i, batch in enumerate(batches[:10], 1):
                        size_mb = batch.total_size / 1024 / 1024
                        batches_table.add_row(
                            str(i),
                            batch.name,
                            str(len(batch.files)),
                            f"{size_mb:.2f} MB"
                        )
                    
                    output.console.print(batches_table)
                    
                    if batch_count > 10:
                        output.console.print(f"\n[dim]... 还有 {batch_count - 10} 个批次[/dim]")
                    
                    if dry_run:
                        output.console.print(f"\n[{BrandColors.WARNING}]⚠ 预览模式：未实际创建批次文件[/ {BrandColors.WARNING}]")
                        output.console.print(f"[dim]提示：移除 --dry-run 参数以执行实际操作[/dim]")
                
                output.console.print(f"\n[bold {BrandColors.SUCCESS}]{'─' * 60}[/bold {BrandColors.SUCCESS}]")
                output.console.print(f"[dim]报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")

            else:
                output.error_panel(
                    f"规划失败：{result.error}",
                    title="规划错误",
                    error_type="BatchError",
                    suggestion="请检查源目录是否有读取权限且包含有效文件"
                )

            return result.success

    try:
        import asyncio
        success = asyncio.run(run_batch())
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
