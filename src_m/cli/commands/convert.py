"""转换命令实现 - 批量 TTS 转换与进度追踪"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from collections import deque

from ...config import ConfigManager, get_preset
from ...executors import TTSExecutor
from ...reliability import create_tts_retry_policy
from ..output import OutputFormatter, TaskStatus, RetryInfo, console, BrandColors, Icons, ParallelProgress
from ..simple_progress import SimpleProgressHandler
from ..errors import CLIError, ErrorCode as E


@dataclass
class TaskTrackingInfo:
    """任务追踪信息"""
    name: str
    status: str = "pending"
    retries: int = 0
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_type: Optional[str] = None


class ConvertProgressHandler:
    """转换进度处理器"""

    def __init__(self, output: OutputFormatter, total_tasks: int, max_retries: int = 3):
        self.output = output
        self.total_tasks = total_tasks
        self._max_retries = max_retries
        self.completed = 0
        self.failed = 0
        self.total_retries = 0
        self.task_infos: Dict[str, TaskTrackingInfo] = {}
        self._parallel_progress: Optional[ParallelProgress] = None
        self._start_time: Optional[float] = None

        self._speed_window: deque = deque(maxlen=100)
        self._recent_completions: deque = deque(maxlen=50)
        self._last_completion_time: Optional[float] = None
        self._eta_window: deque = deque(maxlen=20)

        self._quarantined_count = 0
        self._circuit_breaker_trips = 0

        self._error_type_counts: Dict[str, int] = {}

    def start(self):
        """启动进度显示"""
        self._start_time = time.time()
        self._parallel_progress = ParallelProgress(console, max_workers=4)
        self._parallel_progress.start(self.total_tasks, "TTS转换")

    def stop(self):
        """停止进度显示"""
        if self._parallel_progress:
            self._parallel_progress.stop()
            self._parallel_progress = None

    def register_task(self, task_id: str, name: str):
        """注册任务"""
        self.task_infos[task_id] = TaskTrackingInfo(name=name)
        if self._parallel_progress:
            self._parallel_progress.add_task(name)

    def on_task_start(self, task_id: str):
        """任务开始回调"""
        if task_id in self.task_infos:
            self.task_infos[task_id].status = "running"
            self.task_infos[task_id].start_time = time.time()

    def on_task_complete(self, task_id: str, success: bool, error: Optional[str] = None):
        """任务完成回调"""
        if task_id in self.task_infos:
            info = self.task_infos[task_id]
            info.status = "completed" if success else "failed"
            info.error = error
            info.end_time = time.time()

            current_time = time.time()
            self._recent_completions.append(current_time)

            if info.start_time:
                task_duration = current_time - info.start_time
                if task_duration > 0:
                    self._speed_window.append(1.0 / task_duration)

            if self._last_completion_time:
                interval = current_time - self._last_completion_time
                if interval > 0 and len(self._recent_completions) >= 2:
                    self._eta_window.append(interval)

            self._last_completion_time = current_time

            if success:
                self.completed += 1
                if self._parallel_progress:
                    self._parallel_progress.complete_task(info.name)
            else:
                self.failed += 1
                if error:
                    if ":" in error:
                        error_type = error.split(":")[0].strip()
                    else:
                        error_type = error[:20].strip() if len(error) > 20 else error.strip()
                    self._error_type_counts[error_type] = self._error_type_counts.get(error_type, 0) + 1
                    info.error_type = error_type

                if self._parallel_progress:
                    self._parallel_progress.fail_task(info.name, error or "未知错误")

    def on_quarantine(self, task_id: str):
        """任务进入隔离区回调"""
        self._quarantined_count += 1

    def on_circuit_breaker_trip(self):
        """熔断器触发回调"""
        self._circuit_breaker_trips += 1

    def on_retry(self, task_id: str, attempt: int, error: str, delay: float):
        """重试回调"""
        if task_id in self.task_infos:
            self.task_infos[task_id].retries = attempt
            self.total_retries += 1

        retry_info = RetryInfo(
            attempt=attempt + 1,
            max_attempts=self._max_retries + 1,
            delay=delay,
            error=error,
            will_retry=True
        )
        self.output.retry_status(retry_info)

    def get_stats(self) -> Dict:
        """获取实时统计"""
        elapsed = time.time() - self._start_time if self._start_time else 0

        current_speed = 0.0
        if self._speed_window:
            current_speed = sum(self._speed_window) / len(self._speed_window)

        average_speed = self.completed / elapsed if elapsed > 0 else 0

        p95_speed = 0.0
        if self._speed_window:
            sorted_speeds = sorted(self._speed_window)
            p95_index = int(len(sorted_speeds) * 0.95)
            p95_speed = sorted_speeds[min(p95_index, len(sorted_speeds) - 1)]

        eta = 0.0
        remaining_tasks = self.total_tasks - self.completed - self.failed
        if remaining_tasks > 0 and self._eta_window:
            avg_interval = sum(self._eta_window) / len(self._eta_window)
            eta = remaining_tasks * avg_interval
        elif remaining_tasks > 0 and average_speed > 0:
            eta = remaining_tasks / average_speed

        return {
            "total": self.total_tasks,
            "completed": self.completed,
            "failed": self.failed,
            "elapsed": elapsed,
            "success_rate": (self.completed / self.total_tasks * 100) if self.total_tasks > 0 else 0,
            "current_speed": current_speed,
            "average_speed": average_speed,
            "p95_speed": p95_speed,
            "eta": eta,
            "quarantined": self._quarantined_count,
            "circuit_breaker_trips": self._circuit_breaker_trips,
        }

    def get_detailed_stats(self) -> Dict:
        """获取详细统计"""
        elapsed = time.time() - self._start_time if self._start_time else 0

        retry_rate = (self.total_retries / (self.completed + self.failed) * 100) if (self.completed + self.failed) > 0 else 0
        quarantined_rate = (self._quarantined_count / self.total_tasks * 100) if self.total_tasks > 0 else 0

        current_speed = 0.0
        if self._speed_window:
            current_speed = sum(self._speed_window) / len(self._speed_window)

        average_speed = self.completed / elapsed if elapsed > 0 else 0

        p95_speed = 0.0
        if self._speed_window:
            sorted_speeds = sorted(self._speed_window)
            p95_index = int(len(sorted_speeds) * 0.95)
            p95_speed = sorted_speeds[min(p95_index, len(sorted_speeds) - 1)]

        eta = 0.0
        remaining_tasks = self.total_tasks - self.completed - self.failed
        if remaining_tasks > 0 and self._eta_window:
            avg_interval = sum(self._eta_window) / len(self._eta_window)
            eta = remaining_tasks * avg_interval
        elif remaining_tasks > 0 and average_speed > 0:
            eta = remaining_tasks / average_speed

        avg_task_duration = 0.0
        task_durations = []
        for info in self.task_infos.values():
            if info.start_time and info.end_time:
                task_durations.append(info.end_time - info.start_time)
        if task_durations:
            avg_task_duration = sum(task_durations) / len(task_durations)

        return {
            "total": self.total_tasks,
            "completed": self.completed,
            "failed": self.failed,
            "pending": remaining_tasks,
            "elapsed": elapsed,
            "success_rate": (self.completed / self.total_tasks * 100) if self.total_tasks > 0 else 0,
            "retry_rate": retry_rate,
            "quarantined_rate": quarantined_rate,
            "total_retries": self.total_retries,
            "quarantined_count": self._quarantined_count,
            "circuit_breaker_trips": self._circuit_breaker_trips,
            "current_speed": current_speed,
            "average_speed": average_speed,
            "p95_speed": p95_speed,
            "eta": eta,
            "avg_task_duration": avg_task_duration,
            "error_type_counts": dict(self._error_type_counts),
        }

    def get_summary(self) -> Dict:
        """获取汇总统计"""
        return {
            "total": self.total_tasks,
            "completed": self.completed,
            "failed": self.failed,
            "retries": self.total_retries,
            "quarantined": self._quarantined_count,
            "circuit_breaker_trips": self._circuit_breaker_trips,
            "error_types": dict(self._error_type_counts)
        }


def handle_convert(
    input: Path,
    output: Optional[Path],
    voice: Optional[str],
    concurrency: Optional[int],
    preset: str,
    resume: bool = False,
    checkpoint_path: Optional[Path] = None,
    timeout_multiplier: Optional[float] = None,
    rate: Optional[str] = None,
    recursive: bool = False,
    ramp_up: Optional[float] = None,
    strict: bool = False,
    one: bool = False,
):
    """处理转换命令

    - one=False: 批量模式，处理 input 目录下所有 .txt → output
    - one=True : 单文件模式，input 为单 .txt，output 可缺省（默认 input.parent）
    """
    output_dir: Path
    if one:
        if not input.exists():
            raise CLIError(
                E.E_INPUT_NOT_FOUND,
                f"输入文件不存在: {input}",
                hint="--one 模式下 input 必须为已存在的单 .txt 文件",
            )
        if input.is_dir():
            raise CLIError(
                E.E_INPUT_NOT_FOUND,
                f"--one 模式下 input 必须是文件,不能是目录: {input}",
                hint="移除 --one 进入批量模式,或提供单文件路径",
            )
        if input.suffix.lower() != ".txt":
            raise CLIError(
                E.E_INPUT_NOT_FOUND,
                f"--one 模式下 input 必须是 .txt 文件: {input}",
                hint="提供 .txt 文件路径",
            )
        if output is None:
            output = input.parent
        output.mkdir(parents=True, exist_ok=True)
        output_dir = output
    else:
        if output is None:
            raise CLIError(
                E.E_INPUT_NOT_FOUND,
                "批量模式必须指定输出目录",
                hint="用法: ppc10 convert <input_dir> <output_dir>  或  ppc10 convert <file> --one",
            )
        output_dir = output

    formatter = OutputFormatter(verbose=False)

    formatter.show_banner()

    config_manager = ConfigManager()

    if preset != "balanced":
        config = get_preset(preset)
    else:
        config = config_manager.get_config()

    if voice is not None:
        config.tts.voice = voice
    if concurrency is not None:
        config.tts.concurrency = concurrency
    if rate is not None:
        import re as _re
        rate_val = rate.strip()
        if _re.match(r'^\d+%$', rate_val):
            rate_val = f"+{rate_val}"
        elif not _re.match(r'^[+-]\d+%$', rate_val):
            raise CLIError(
                E.E_BUSINESS,
                f"无效的 rate 参数: '{rate}',应为 '+0%'、'+40%'、'-10%' 等格式",
                hint="rate 必须带正负号前缀和百分号,例如 --rate +40% 或 --rate -10%",
            )
        config.tts.rate = rate_val
    if timeout_multiplier is not None:
        config.tts._timeout_multiplier = timeout_multiplier
    if ramp_up is not None:
        config.tts.ramp_up_enabled = True
        config.tts.ramp_up_duration = ramp_up

    formatter.info(f"输入: {input}")
    formatter.info(f"输出目录: {output_dir}")
    formatter.info(f"语音: {config.tts.voice}")
    formatter.info(f"并发数: {config.tts.concurrency}")
    formatter.info(f"音频速度: {config.tts.rate}")
    if one:
        formatter.info("模式: --one（单文件无限重试，无超时）")
    if timeout_multiplier is not None:
        formatter.info(f"超时倍率: {timeout_multiplier}x (基于动态计算的超时)")
    if config.tts.ramp_up_enabled:
        formatter.info(f"并发预热: {config.tts.ramp_up_duration:.0f}s 内从 1 逐步增加到{config.tts.concurrency} 并发")
    if resume:
        formatter.info(f"断点续传: 已启用")
    if recursive:
        formatter.info(f"递归模式: 启用")

    if not one and not input.exists():
        raise CLIError(
            E.E_INPUT_NOT_FOUND,
            f"输入目录不存在: {input}",
            hint="请检查路径是否正确,或使用绝对路径",
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    if one:
        output_file = output_dir / (input.stem + ".mp3")

        # 预分段以获取段数（让 --one 进度显示总段数而非 1）
        from ...utils.files import detect_encoding as _detect_enc
        from ...text.segmenter import TextSegmenter
        try:
            _enc = _detect_enc(input)
            _text = input.read_text(encoding=_enc or "utf-8")
        except (OSError, UnicodeDecodeError):
            _text = ""
        if _text and _text.strip():
            if len(_text) <= config.tts.max_segment_length:
                _segments = [_text]
            else:
                _segmenter = TextSegmenter.from_config(config.tts)
                _segments = _segmenter.split(_text, config.tts.max_segment_length) or [_text]
        else:
            _segments = []
        total_segments = len(_segments)
        formatter.info(f"分段数: {total_segments}")

        # 单文件模式直接走 execute_one
        async def run_convert():
            from ...reliability import create_tts_retry_policy
            retry_policy = create_tts_retry_policy(
                max_retries=config.reliability.tts_retry.max_retries,
                base_delay=config.reliability.tts_retry.base_delay,
                max_delay=config.reliability.tts_retry.max_delay,
                exponential_base=config.reliability.tts_retry.exponential_base,
                jitter=config.reliability.tts_retry.jitter
            )
            from ..simple_progress import SimpleProgressHandler
            progress_handler = SimpleProgressHandler(total_tasks=total_segments)
            from ...executors import TTSExecutor
            async with TTSExecutor(config, retry_policy) as executor:
                progress_handler.start()
                try:
                    ok = await executor.execute_one(
                        input, output_file, progress_handler=progress_handler
                    )
                except KeyboardInterrupt:
                    formatter.warning_panel("用户中断", title="中断")
                    return False
                finally:
                    progress_handler.stop()
                return ok

        try:
            success = asyncio.run(run_convert())
            if success:
                formatter.success_panel(
                    f"已生成: {output_file}",
                    title="完成",
                    details={"文件": str(output_file)},
                )
            else:
                raise CLIError(E.E_BUSINESS, "--one 模式未成功", exit_code=1)
            return
        except KeyboardInterrupt:
            formatter.warning("\n用户中断")
            raise CLIError(E.E_BUSINESS, "用户中断 (Ctrl+C)", exit_code=130)
        except CLIError:
            raise
        except OSError as e:
            raise CLIError(
                E.E_BUSINESS,
                f"文件操作失败: {e}",
                hint="请检查文件路径、权限或磁盘空间",
            ) from e
        except TimeoutError as e:
            raise CLIError(
                E.E_BUSINESS,
                f"操作超时: {e}",
                hint="可尝试 --timeout-multiplier 增大超时时间",
            ) from e
        # 顶层错误包装：捕获所有未处理的异常，统一转为 CLIError 以提供一致的错误输出
        except Exception as e:
            raise CLIError(
                E.E_BUSINESS,
                f"执行失败: {e}",
                hint="使用 --verbose 复跑获取 stack trace",
            ) from e

    if recursive:
        txt_files = sorted(input.rglob("*.txt"))
    else:
        txt_files = sorted(input.glob("*.txt"))
    if not txt_files:
        msg = f"No .txt files found in {input}. Nothing to do."
        if strict:
            raise CLIError(
                E.E_INPUT_EMPTY,
                msg,
                hint="Remove --strict to allow empty input",
            )
        formatter.info(msg)
        return

    files_to_process = []
    for txt_file in txt_files:
        if recursive:
            rel_path = txt_file.relative_to(input)
            output_file = output_dir / rel_path.with_suffix(".mp3")
        else:
            output_file = output_dir / txt_file.with_suffix(".mp3").name
        if not output_file.exists() or output_file.stat().st_size == 0:
            files_to_process.append(txt_file)

    if not files_to_process:
        formatter.success_panel(
            "所有文件已处理完成，无需转换",
            title="完成",
            details={"已处理": len(txt_files)}
        )
        return

    formatter.info(f"待处理文件数: {len(files_to_process)}")

    async def run_convert():
        retry_policy = create_tts_retry_policy(
            max_retries=config.reliability.tts_retry.max_retries,
            base_delay=config.reliability.tts_retry.base_delay,
            max_delay=config.reliability.tts_retry.max_delay,
            exponential_base=config.reliability.tts_retry.exponential_base,
            jitter=config.reliability.tts_retry.jitter
        )
        progress_handler = SimpleProgressHandler(len(files_to_process))

        async with TTSExecutor(config, retry_policy) as executor:
            if resume:
                ckpt_path = checkpoint_path or output_dir / ".ppc10_checkpoint.json"
                executor.enable_checkpoint(ckpt_path)

            executor.set_progress_callback(progress_handler)
            progress_handler.start()

            try:
                result = await executor.add_batch_with_progress(
                    input,
                    output_dir,
                    progress_handler=progress_handler,
                    recursive=recursive
                )
            except KeyboardInterrupt:
                formatter.warning_panel("用户中断，检查点已保存", title="中断")
                formatter.info("使用 --resume 参数可从中断处继续")
                return False
            finally:
                progress_handler.stop()

            stats = progress_handler.get_stats()

            formatter.console.print(f"\n[bold {BrandColors.PRIMARY}]{'█' * 60}[/bold {BrandColors.PRIMARY}]")
            formatter.console.print(f"[bold white]  {Icons.STAR} 转换完成报告[/bold white]")
            formatter.console.print(f"[bold {BrandColors.PRIMARY}]{'█' * 60}[/bold {BrandColors.PRIMARY}]\n")

            completed = stats['completed']
            failed = stats['failed']
            total = stats['total']
            elapsed = stats['elapsed']
            avg_speed = stats.get('current_speed', 0)

            success_rate = (completed / total * 100) if total > 0 else 0

            if success_rate >= 90:
                result_icon = Icons.SUCCESS
                result_color = BrandColors.SUCCESS
                result_text = "优秀"
            elif success_rate >= 70:
                result_icon = Icons.WARNING
                result_color = BrandColors.WARNING
                result_text = "良好"
            else:
                result_icon = Icons.ERROR
                result_color = BrandColors.ERROR
                result_text = "需改进"

            formatter.console.print(f"[bold {result_color}]{result_icon} 总体评价：{result_text}[/bold {result_color}]\n")

            from rich.table import Table
            from rich.box import SIMPLE
            summary_table = Table(show_header=False, box=SIMPLE, border_style=BrandColors.PRIMARY)
            summary_table.add_column("指标", style="bold", width=20)
            summary_table.add_column("值", style="cyan", width=20)

            summary_table.add_row("总任务数", str(total))
            summary_table.add_row(f"[{BrandColors.SUCCESS}]成功[/{BrandColors.SUCCESS}]", f"[{BrandColors.SUCCESS}]{completed}[/{BrandColors.SUCCESS}]")
            summary_table.add_row(f"[{BrandColors.ERROR}]失败[/{BrandColors.ERROR}]", f"[{BrandColors.ERROR}]{failed}[/{BrandColors.ERROR}]")
            summary_table.add_row("成功率", f"[{result_color}]{success_rate:.1f}%[/{result_color}]")
            summary_table.add_row("总用时", f"{elapsed:.1f}s")
            summary_table.add_row("平均速度", f"{avg_speed:.2f} 任务/秒")

            # Add retry stats
            total_retries = executor.total_retries
            if total_retries > 0:
                summary_table.add_row("重试次数", str(total_retries))
                summary_table.add_row("实际API请求", f"{total + total_retries} (原始{total} + 重试{total_retries})")

            formatter.console.print(summary_table)
            formatter.console.print()

            if failed > 0 and progress_handler.task_infos:
                error_tasks = [
                    (info.name, info.error)
                    for info in progress_handler.task_infos.values()
                    if info.status == "failed" and info.error
                ][:5]

                if error_tasks:
                    formatter.console.print(f"[bold {BrandColors.ERROR}]📋 失败任务示例:[/bold {BrandColors.ERROR}]\n")
                    error_table = Table(show_header=True, box=SIMPLE, border_style=BrandColors.ERROR)
                    error_table.add_column("任务名", style="yellow", width=30)
                    error_table.add_column("错误信息", style="red", width=40)

                    for task_name, error_msg in error_tasks:
                        short_error = error_msg[:50] + "..." if len(error_msg) > 50 else error_msg
                        error_table.add_row(task_name, short_error)

                    formatter.console.print(error_table)
                    formatter.console.print()

            formatter.console.print(f"[bold {BrandColors.PRIMARY}]{'─' * 60}[/bold {BrandColors.PRIMARY}]")
            from datetime import datetime
            formatter.console.print(f"[dim]报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")

            return failed == 0

    try:
        success = asyncio.run(run_convert())
        if not success:
            raise CLIError(
                E.E_BUSINESS,
                "存在失败任务,请查看上方重试/错误明细",
                hint="使用 --verbose 复跑,或调整 --timeout-multiplier / 减少并发",
            )
    except KeyboardInterrupt:
        formatter.warning("\n用户中断")
        raise CLIError(E.E_BUSINESS, "用户中断 (Ctrl+C)", exit_code=130)
    except CLIError:
        raise
    except OSError as e:
        raise CLIError(
            E.E_BUSINESS,
            f"文件操作失败: {e}",
            hint="请检查文件路径、权限或磁盘空间",
        ) from e
    except TimeoutError as e:
        raise CLIError(
            E.E_BUSINESS,
            f"操作超时: {e}",
            hint="可尝试 --timeout-multiplier 增大超时时间，或减少并发数",
        ) from e
    # 顶层错误包装：捕获所有未处理的异常，统一转为 CLIError 以提供一致的错误输出
    except Exception as e:
        raise CLIError(
            E.E_BUSINESS,
            f"执行失败: {e}",
            hint="使用 --verbose 复跑获取 stack trace",
        ) from e
