"""批量命令实现 - 文件归档与管理"""

import sys
from pathlib import Path

from ...config import ConfigManager, get_preset
from ...executors import BatcherExecutor
from ..output import OutputFormatter, BrandColors, Icons
from ..errors import CLIError, ErrorCode as E
from rich.table import Table
from rich.box import SIMPLE


def handle_batch(
    source_dir: Path,
    batch_size: int,
    dry_run: bool,
    group_by_volume: bool = False,
    strict: bool = False,
):
    """处理批量命令"""
    output = OutputFormatter(verbose=False)

    output.show_banner()

    output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'█' * 60}[/bold {BrandColors.PRIMARY}]")
    output.console.print(f"[bold white]  {Icons.FOLDER} PPC10 批量归档[/bold white]")
    output.console.print(f"[bold {BrandColors.PRIMARY}]{'█' * 60}[/bold {BrandColors.PRIMARY}]\n")

    config_manager = ConfigManager()
    config = config_manager.get_config()

    if batch_size:
        config.batch.max_files_per_batch = batch_size

    config_details = {
        "源目录": str(source_dir),
        "批次大小": f"{config.batch.max_files_per_batch} 文件",
        "模式": "预览" if dry_run else "执行",
    }

    if group_by_volume:
        config_details["归档方式"] = "按卷归档"

    for key, value in config_details.items():
        output.console.print(f"[dim]{key}:[/dim] [cyan]{value}[/cyan]")

    output.console.print()

    if not source_dir.exists():
        raise CLIError(
            E.E_INPUT_NOT_FOUND,
            f"源目录不存在: {source_dir}",
            hint="请检查路径是否正确,或使用绝对路径",
        )

    # 空输入友好处理 (Spec 9)
    txt_files = sorted(source_dir.glob("*.txt"))
    if not txt_files:
        msg = f"No .txt files found in {source_dir}. Nothing to do."
        if strict:
            raise CLIError(
                E.E_INPUT_EMPTY,
                msg,
                hint="Remove --strict to allow empty input",
            )
        output.info(msg)
        return

    async def run_batch():
        async with BatcherExecutor(config) as executor:
            if dry_run:
                result = await executor.dry_run(source_dir)
            elif group_by_volume:
                output_dir = source_dir / "archives"
                result = await executor.group_by_volume(source_dir, output_dir)
            else:
                output_dir = source_dir / "batches"
                result = await executor.execute(source_dir, output_dir)

            if result.success:
                if group_by_volume:
                    archives = result.data
                    archive_count = len(archives)

                    output.console.print(f"\n[bold {BrandColors.SUCCESS}]{'█' * 60}[/bold {BrandColors.SUCCESS}]")
                    output.console.print(f"[bold white]  {Icons.SUCCESS} 按卷归档报告[/bold white]")
                    output.console.print(f"[bold {BrandColors.SUCCESS}]{'█' * 60}[/bold {BrandColors.SUCCESS}]\n")

                    from datetime import datetime

                    summary_table = Table(show_header=False, box=SIMPLE, border_style=BrandColors.SUCCESS)
                    summary_table.add_column("指标", style="bold", width=20)
                    summary_table.add_column("值", style="cyan", width=25)

                    total_size = sum(archive.stat().st_size for archive in archives)

                    summary_table.add_row("归档文件数量", f"[{BrandColors.SUCCESS}]{archive_count}[/{BrandColors.SUCCESS}]")
                    summary_table.add_row("总大小", f"{total_size/1024:.1f} KB")
                    summary_table.add_row("总用时", f"{result.metrics.duration:.1f}s")

                    output.console.print(summary_table)
                    output.console.print()

                    if archive_count > 0:
                        output.console.print(f"[bold {BrandColors.INFO}]📦 归档文件:[/bold {BrandColors.INFO}]\n")

                        archive_table = Table(show_header=True, box=SIMPLE, border_style=BrandColors.INFO)
                        archive_table.add_column("序号", style="yellow", width=6)
                        archive_table.add_column("归档名", style="white", width=50)
                        archive_table.add_column("大小", style="cyan", width=12, justify="right")

                        for i, archive in enumerate(archives, 1):
                            size_kb = archive.stat().st_size / 1024
                            archive_table.add_row(
                                str(i),
                                archive.name,
                                f"{size_kb:.1f} KB"
                            )

                        output.console.print(archive_table)

                    if dry_run:
                        output.console.print(f"\n[{BrandColors.WARNING}]⚠️ 预览模式：未实际创建归档文件[/{BrandColors.WARNING}]")
                        output.console.print(f"[dim]提示：移除--dry-run 参数以执行实际操作[/dim]")

                    output.console.print(f"\n[bold {BrandColors.SUCCESS}]{'─' * 60}[/bold {BrandColors.SUCCESS}]")
                    output.console.print(f"[dim]报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")
                else:
                    batches = result.data
                    batch_count = len(batches)

                    output.console.print(f"\n[bold {BrandColors.SUCCESS}]{'█' * 60}[/bold {BrandColors.SUCCESS}]")
                    output.console.print(f"[bold white]  {Icons.SUCCESS} 批次规划报告[/bold white]")
                    output.console.print(f"[bold {BrandColors.SUCCESS}]{'█' * 60}[/bold {BrandColors.SUCCESS}]\n")

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
                    summary_table.add_row("总用时", f"{result.metrics.duration:.1f}s")

                    output.console.print(summary_table)
                    output.console.print()

                    if batch_count > 0:
                        output.console.print(f"[bold {BrandColors.INFO}]📦 批次详情 (前10 项:[/bold {BrandColors.INFO}]\n")

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
                            output.console.print(f"\n[{BrandColors.WARNING}]⚠️ 预览模式：未实际创建批次文件[/{BrandColors.WARNING}]")
                            output.console.print(f"[dim]提示：移除--dry-run 参数以执行实际操作[/dim]")

                    output.console.print(f"\n[bold {BrandColors.SUCCESS}]{'─' * 60}[/bold {BrandColors.SUCCESS}]")
                    output.console.print(f"[dim]报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")

            else:
                raise CLIError(
                    E.E_BUSINESS,
                    f"规划失败: {result.error}",
                    hint="请检查源目录是否有读取权限且包含有效文件",
                )

            return result.success

    try:
        import asyncio
        success = asyncio.run(run_batch())
        if not success:
            raise CLIError(E.E_BUSINESS, "批量归档未成功,请查看上文错误")
    except KeyboardInterrupt:
        raise CLIError(E.E_BUSINESS, "用户中断操作 (Ctrl+C)", exit_code=130)
    except CLIError:
        raise
    except Exception as e:
        raise CLIError(
            E.E_BUSINESS,
            f"执行失败: {e}",
            hint="使用 --verbose 参数查看详细错误信息",
        ) from e
