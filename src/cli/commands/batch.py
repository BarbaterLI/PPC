"""批量命令实现 - 文件归档与管理"""

from pathlib import Path

from src.cli.typer_app import get_output

from ...config import ConfigManager
from ...executors import BatcherExecutor
from ..errors import CLIError
from ..errors import ErrorCode as E


def handle_batch(
    source_dir: Path,
    batch_size: int,
    dry_run: bool,
    group_by_volume: bool = False,
    strict: bool = False,
):
    """处理批量命令"""
    output = get_output()

    output.show_banner()
    output.print_panel("PPC10 批量归档", title="批量归档", style="primary")

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
        output.info(f"{key}: {value}")

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

                    from datetime import datetime

                    total_size = sum(archive.stat().st_size for archive in archives)
                    summary_rows = [
                        ["归档文件数量", archive_count],
                        ["总大小", f"{total_size / 1024:.1f} KB"],
                        ["总用时", f"{result.metrics.duration:.1f}s"],
                    ]

                    output.print_panel("按卷归档完成", title="按卷归档", style="success")
                    output.print_table(["指标", "值"], summary_rows, title="汇总")

                    if archive_count > 0:
                        archive_rows = [
                            [str(i), archive.name, f"{archive.stat().st_size / 1024:.1f} KB"]
                            for i, archive in enumerate(archives, 1)
                        ]
                        output.print_table(
                            ["序号", "归档名", "大小"],
                            archive_rows,
                            title="归档文件",
                        )

                    if dry_run:
                        output.warning_panel(
                            "预览模式：未实际创建归档文件",
                            title="预览",
                            suggestion="移除 --dry-run 参数以执行实际操作",
                        )

                    output.info(f"报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    batches = result.data
                    batch_count = len(batches)

                    total_files = sum(len(batch.files) for batch in batches)
                    total_size = sum(batch.total_size for batch in batches)
                    avg_files = f"{total_files / batch_count:.1f}" if batch_count > 0 else "0"

                    summary_rows = [
                        ["批次数量", batch_count],
                        ["文件总数", total_files],
                        ["总大小", f"{total_size / 1024 / 1024:.2f} MB"],
                        ["平均每批文件", avg_files],
                        ["总用时", f"{result.metrics.duration:.1f}s"],
                    ]

                    output.print_panel("批次规划完成", title="批次规划", style="success")
                    output.print_table(["指标", "值"], summary_rows, title="汇总")

                    if batch_count > 0:
                        batch_rows = [
                            [str(i), batch.name, str(len(batch.files)), f"{batch.total_size / 1024 / 1024:.2f} MB"]
                            for i, batch in enumerate(batches[:10], 1)
                        ]
                        output.print_table(
                            ["序号", "批次名", "文件数", "大小"],
                            batch_rows,
                            title="批次详情 (前10 项)",
                        )

                        if batch_count > 10:
                            output.info(f"... 还有 {batch_count - 10} 个批次")

                    if dry_run:
                        output.warning_panel(
                            "预览模式：未实际创建批次文件",
                            title="预览",
                            suggestion="移除 --dry-run 参数以执行实际操作",
                        )
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
        raise CLIError(E.E_BUSINESS, "用户中断操作 (Ctrl+C)", exit_code=130) from None
    except CLIError:
        raise
    except Exception as e:
        raise CLIError(
            E.E_BUSINESS,
            f"执行失败: {e}",
            hint="使用 --verbose 参数查看详细错误信息",
        ) from e
