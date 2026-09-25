"""转换命令实现 - 批量 TTS 转换与进度追踪"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.cli.typer_app import get_output

from ...config import ConfigManager, get_preset
from ...executors import TTSExecutor
from ...reliability import create_tts_retry_policy
from ...core.exceptions import CLIError
from ...core.exceptions import ErrorCodes as E
from ..simple_progress import SimpleProgressHandler
from ._async_runner import run_cli_async

if TYPE_CHECKING:
    from src.config import PPCConfig
    from src.cli.output import OutputFormatter


@dataclass
class ConvertOptions:
    """convert 命令参数封装。"""

    input: Path
    output: Path | None
    voice: str | None
    concurrency: int | None
    preset: str
    resume: bool = False
    checkpoint: Path | None = None
    timeout_multiplier: float | None = None
    rate: str | None = None
    recursive: bool = False
    ramp_up: float | None = None
    strict: bool = False
    one: bool = False
    timeout_mode: str | None = None
    timeout: int | None = None
    tts_backend: str | None = None
    tts_proxy: str | None = None


def _normalize_rate(rate: str) -> str:
    import re as _re

    rate_val = rate.strip()
    if _re.match(r"^\d+%$", rate_val):
        rate_val = f"+{rate_val}"
    elif not _re.match(r"^[+-]\d+%$", rate_val):
        raise CLIError(
            E.E_BUSINESS,
            f"无效的 rate 参数: '{rate}',应为 '+0%'、'+40%'、'-10%' 等格式",
            hint="rate 必须带正负号前缀和百分号,例如 --rate +40% 或 --rate -10%",
        )
    return rate_val


def _build_retry_policy(config: PPCConfig):
    return create_tts_retry_policy(
        max_retries=config.reliability.tts_retry.max_retries,
        base_delay=config.reliability.tts_retry.base_delay,
        max_delay=config.reliability.tts_retry.max_delay,
        exponential_base=config.reliability.tts_retry.exponential_base,
        jitter=config.reliability.tts_retry.jitter,
    )


def _count_segments(input_file: Path, tts_config) -> int:
    # 预分段以获取段数（让 --one 进度显示总段数而非 1）
    from ...text.segmenter import TextSegmenter
    from ...utils.files import detect_encoding as _detect_enc

    try:
        _enc = _detect_enc(input_file)
        _text = input_file.read_text(encoding=_enc or "utf-8")
    except (OSError, UnicodeDecodeError):
        _text = ""
    if _text and _text.strip():
        if len(_text) <= tts_config.max_segment_length:
            _segments = [_text]
        else:
            _segmenter = TextSegmenter.from_config(tts_config)
            _segments = _segmenter.split(_text, tts_config.max_segment_length) or [_text]
    else:
        _segments = []
    return len(_segments)


def _log_convert_details(options: ConvertOptions, output_dir: Path, config: PPCConfig, output_formatter: OutputFormatter) -> None:
    output_formatter.detail(f"输入: {options.input}")
    output_formatter.detail(f"输出目录: {output_dir}")
    output_formatter.detail(f"语音: {config.tts.voice}")
    output_formatter.detail(f"并发数: {config.tts.concurrency}")
    output_formatter.detail(f"音频速度: {config.tts.rate}")
    if options.one:
        output_formatter.detail("模式: --one（单文件无限重试，无超时）")
    if options.timeout_mode is not None:
        output_formatter.detail(f"超时模式: {options.timeout_mode}")
    if options.timeout is not None:
        output_formatter.detail(f"固定超时: {options.timeout}s")
    if options.timeout_multiplier is not None:
        output_formatter.detail(f"超时倍率: {options.timeout_multiplier}x")
    if config.tts.ramp_up_enabled:
        output_formatter.detail(
            f"并发预热: {config.tts.ramp_up_duration:.0f}s 内从 1 逐步增加到{config.tts.concurrency} 并发"
        )
    if options.resume:
        output_formatter.detail("断点续传: 已启用")
    if options.recursive:
        output_formatter.detail("递归模式: 启用")


def _validate_convert(
    options: ConvertOptions,
) -> tuple[Path, PPCConfig, OutputFormatter]:
    output_dir: Path
    if options.one:
        if not options.input.exists():
            raise CLIError(
                E.E_INPUT_NOT_FOUND,
                f"输入文件不存在: {options.input}",
                hint="--one 模式下 input 必须为已存在的单 .txt 文件",
            )
        if options.input.is_dir():
            raise CLIError(
                E.E_INPUT_NOT_FOUND,
                f"--one 模式下 input 必须是文件,不能是目录: {options.input}",
                hint="移除 --one 进入批量模式,或提供单文件路径",
            )
        if options.input.suffix.lower() != ".txt":
            raise CLIError(
                E.E_INPUT_NOT_FOUND,
                f"--one 模式下 input 必须是 .txt 文件: {options.input}",
                hint="提供 .txt 文件路径",
            )
        if options.output is None:
            options.output = options.input.parent
        options.output.mkdir(parents=True, exist_ok=True)
        output_dir = options.output
    else:
        if options.output is None:
            raise CLIError(
                E.E_INPUT_NOT_FOUND,
                "批量模式必须指定输出目录",
                hint="用法: ppc convert <input_dir> <output_dir>  或  ppc convert <file> --one",
            )
        output_dir = options.output

    output_formatter = get_output()

    config_manager = ConfigManager()

    # balanced 预设直接取管理器共享配置：必须深拷贝，避免 CLI 覆盖
    # （voice/concurrency/rate/...）污染全局配置；其余预设每次构建全新实例。
    if options.preset == "balanced":
        config = copy.deepcopy(config_manager.get_config())
    else:
        config = get_preset(options.preset)

    # 未知 preset 警告：get_preset 对未知名称静默回退到 balanced
    if options.preset != "balanced":
        from ...config import get_preset_names
        if options.preset not in get_preset_names():
            output_formatter.warning(f"未知预设 '{options.preset}'，已回退到 balanced 预设")

    if options.voice is not None:
        config.tts.voice = options.voice
    if options.concurrency is not None:
        config.tts.concurrency = options.concurrency
    if options.rate is not None:
        config.tts.rate = _normalize_rate(options.rate)
    if options.timeout_multiplier is not None:
        config.tts.timeout_multiplier = options.timeout_multiplier
    if options.timeout_mode is not None:
        config.tts.timeout_mode = options.timeout_mode
    if options.timeout is not None:
        config.tts.timeout = options.timeout
    if options.ramp_up is not None:
        config.tts.ramp_up_enabled = True
        config.tts.ramp_up_duration = options.ramp_up

    _log_convert_details(options, output_dir, config, output_formatter)

    if not options.one and not options.input.exists():
        raise CLIError(
            E.E_INPUT_NOT_FOUND,
            f"输入目录不存在: {options.input}",
            hint="请检查路径是否正确,或使用绝对路径",
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, config, output_formatter


def _run_convert_one(
    options: ConvertOptions,
    output_dir: Path,
    config: PPCConfig,
    output_formatter: OutputFormatter,
) -> None:
    output_file = output_dir / (options.input.stem + ".mp3")

    total_segments = _count_segments(options.input, config.tts)
    if total_segments == 0:
        raise CLIError(
            E.E_INPUT_EMPTY,
            f"输入文件内容为空，无法转换: {options.input}",
            hint="请提供非空 .txt 文件，或检查文件编码",
        )
    output_formatter.info(f"分段数: {total_segments}")

    # 单文件模式直接走 execute_one
    async def run_convert_one():
        retry_policy = _build_retry_policy(config)

        progress_handler = SimpleProgressHandler(total_tasks=1, formatter=output_formatter)

        async with TTSExecutor(
            config, retry_policy, tts_backend=options.tts_backend, tts_proxy=options.tts_proxy
        ) as executor:
            progress_handler.start()
            try:
                ok = await executor.execute_one(options.input, output_file, progress_handler=progress_handler)
            except KeyboardInterrupt:
                output_formatter.warning_panel("用户中断", title="中断")
                return False
            finally:
                progress_handler.stop()
            return ok

    def _render_one_result(ok: bool):
        if ok:
            output_formatter.success_panel(
                f"已生成: {output_file}",
                title="完成",
                details={"文件": str(output_file)},
            )
        return ok

    run_cli_async(
        run_convert_one(),
        failure_message="--one 模式未成功",
        failure_exit_code=1,
        catch_io_errors=True,
        timeout_hint="可尝试 --timeout-multiplier 增大超时时间",
        on_result=_render_one_result,
    )


def _print_failed_tasks(output_formatter: OutputFormatter, progress_handler: SimpleProgressHandler) -> None:
    error_tasks = [
        (info.name, info.error)
        for info in progress_handler.task_infos.values()
        if info.status == "failed" and info.error
    ][:5]

    if error_tasks and output_formatter.mode == "human":
        error_rows = [
            [task_name, error_msg[:50] + "..." if len(error_msg) > 50 else error_msg]
            for task_name, error_msg in error_tasks
        ]
        output_formatter.print_table(
            ["任务名", "错误信息"],
            error_rows,
            title="失败任务示例",
        )


def _collect_pending_files(
    options: ConvertOptions,
    output_dir: Path,
    output_formatter: OutputFormatter,
) -> list[Path] | None:
    # 预扫描文件用于空检查与日志展示；实际任务注册与处理由
    # add_batch_with_progress 内部独立发现（保持单一数据源）
    # 此处的 files_to_process 计数可能略大于实际注册数（断点续传场景），
    # progress_handler.total_tasks 会在 executor 内同步为实际值
    txt_files = sorted(options.input.rglob("*.txt")) if options.recursive else sorted(options.input.glob("*.txt"))
    if not txt_files:
        msg = f"No .txt files found in {options.input}. Nothing to do."
        if options.strict:
            raise CLIError(
                E.E_INPUT_EMPTY,
                msg,
                hint="Remove --strict to allow empty input",
            )
        output_formatter.info(msg)
        return None

    files_to_process = []
    for txt_file in txt_files:
        if options.recursive:
            rel_path = txt_file.relative_to(options.input)
            output_file = output_dir / rel_path.with_suffix(".mp3")
        else:
            output_file = output_dir / txt_file.with_suffix(".mp3").name
        if not output_file.exists() or output_file.stat().st_size == 0:
            files_to_process.append(txt_file)

    if not files_to_process:
        output_formatter.success_panel("所有文件已处理完成，无需转换", title="完成", details={"已处理": len(txt_files)})
        return None

    return files_to_process


def _run_convert_batch(
    options: ConvertOptions,
    output_dir: Path,
    config: PPCConfig,
    output_formatter: OutputFormatter,
) -> None:
    files_to_process = _collect_pending_files(options, output_dir, output_formatter)
    if files_to_process is None:
        return

    output_formatter.info(f"待处理文件数: {len(files_to_process)}")

    async def run_convert():
        retry_policy = _build_retry_policy(config)
        progress_handler = SimpleProgressHandler(len(files_to_process), formatter=output_formatter)

        async with TTSExecutor(
            config, retry_policy, tts_backend=options.tts_backend, tts_proxy=options.tts_proxy
        ) as executor:
            if options.resume:
                ckpt_path = options.checkpoint or output_dir / ".ppc_checkpoint.json"
                executor.enable_checkpoint(ckpt_path)

            executor.set_progress_callback(progress_handler)
            progress_handler.start()

            try:
                await executor.add_batch_with_progress(
                    options.input, output_dir, progress_handler=progress_handler, recursive=options.recursive
                )
            except KeyboardInterrupt:
                output_formatter.warning_panel("用户中断，检查点已保存", title="中断")
                output_formatter.info("使用 --resume 参数可从中断处继续")
                return False
            finally:
                progress_handler.stop()

            stats = progress_handler.get_detailed_stats()

            output_formatter.completion_report(
                stats,
                executor_stats=None,
                title="转换完成报告",
            )

            failed = stats["failed"]
            if failed > 0 and progress_handler.task_infos:
                _print_failed_tasks(output_formatter, progress_handler)

            return failed == 0

    def _warn_interrupt():
        output_formatter.warning("\n用户中断")

    # 长时间合成期间阻止系统休眠（失败时工具内部自动降级为无操作）
    from ...utils.keep_awake import keep_awake_context

    with keep_awake_context():
        run_cli_async(
            run_convert(),
            failure_message="存在失败任务,请查看上方重试/错误明细",
            failure_hint="使用 --verbose 查看详细错误信息,或调整 --timeout-multiplier / 减少并发",
            interrupt_message="用户中断 (Ctrl+C)",
            on_interrupt=_warn_interrupt,
            catch_io_errors=True,
        )


def handle_convert(
    options: ConvertOptions,
) -> None:
    """处理转换命令

    - one=False: 批量模式，处理 input 目录下所有 .txt → output
    - one=True : 单文件模式，input 为单 .txt，output 可缺省（默认 input.parent）
    """
    output_dir, config, output_formatter = _validate_convert(options)

    if options.one:
        _run_convert_one(options, output_dir, config, output_formatter)
        return

    _run_convert_batch(options, output_dir, config, output_formatter)
