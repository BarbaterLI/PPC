"""resume 命令实现 - 从已有的 .cache 分段重建断点续传检查点。

用于之前未启用 --resume 但中途中断的场景：扫描 output 目录下的 .cache，
根据已存在的段文件和对应 input 文件生成 checkpoint，再用 `convert --resume`
继续完成转换。
"""

from pathlib import Path

from ...executors.checkpoint import CheckpointManager
from ..errors import CLIError
from ..errors import ErrorCode as E
from ..typer_app import get_output


def handle_resume(
    input: Path,
    output: Path,
    voice: str | None = None,
    checkpoint_path: Path | None = None,
) -> None:
    """处理 resume 命令。

    Args:
        input: 输入目录（含原始 .txt 文件）
        output: 输出目录（含 .cache 子目录）
        voice: 语音模型，默认从配置文件读取
        checkpoint_path: 检查点文件路径，默认 output/.ppc10_checkpoint.json
    """
    output_formatter = get_output()

    if not input.exists() or not input.is_dir():
        raise CLIError(
            E.E_INPUT_NOT_FOUND,
            f"输入目录不存在: {input}",
            hint="请提供包含原始 .txt 文件的输入目录",
        )
    if not output.exists() or not output.is_dir():
        raise CLIError(
            E.E_INPUT_NOT_FOUND,
            f"输出目录不存在: {output}",
            hint="请提供包含 .cache 子目录的输出目录",
        )

    if voice is None:
        from ...config import ConfigManager

        config = ConfigManager().get_config()
        voice = config.tts.voice

    ckpt_path = checkpoint_path or output / ".ppc10_checkpoint.json"
    manager = CheckpointManager(ckpt_path)
    data = manager.rebuild_from_cache(
        input_dir=input,
        output_dir=output,
        voice=voice,
    )

    if data is None:
        output_formatter.warning_panel(
            "未找到可恢复的任务",
            title="resume",
            details="\n".join(
                f"{k}: {v}"
                for k, v in {
                    "input": str(input),
                    "output": str(output),
                    "cache": str(output / ".cache"),
                }.items()
            ),
        )
        return

    saved = manager.save()
    if not saved:
        raise CLIError(
            E.E_BUSINESS,
            f"保存检查点失败: {ckpt_path}",
            hint="请检查路径权限或磁盘空间",
        )

    output_formatter.success_panel(
        f"已重建断点续传检查点: {ckpt_path}",
        title="resume",
        details={
            "总任务": data.total_tasks,
            "已完成": data.completed_tasks,
            "待处理": data.pending_tasks,
            "语音": data.voice,
        },
    )
    output_formatter.info("接下来可运行：ppc10 convert <input> <output> --resume")
