"""并发预热（ramp-up）功能单元测试。

覆盖：
1. ``RampUpController`` 的状态机（start / get_current_concurrency / is_complete）
2. ``TTSExecutor.initialize()`` 在 ``ramp_up_enabled`` 启用/禁用时的信号量初始化
   （信号量现由 ``WorkerPool`` 持有）
3. ``WorkerPool.start`` 启动预热控制器与后台调整任务
4. ``WorkerPool._ramp_up_adjuster`` 逐步释放许可、预热完成时释放至目标并发数、
   被取消时仍释放剩余许可

约束：所有测试 SHALL 使用 ``tmp_path`` fixture，绝不写入 ``ms_tts`` 目录。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from src.config.presets import get_preset
from src.executors.tts_executor import RampUpController, TTSExecutor


def _run(coro):
    return asyncio.run(coro)


def _make_executor(*, ramp_up_enabled: bool, concurrency: int = 5, ramp_up_duration: float = 30.0):
    cfg = get_preset("balanced")
    cfg.tts.concurrency = concurrency
    cfg.tts.ramp_up_enabled = ramp_up_enabled
    cfg.tts.ramp_up_duration = ramp_up_duration
    executor = TTSExecutor(cfg)
    return executor


# ---------------------------------------------------------------------------
# RampUpController 状态机
# ---------------------------------------------------------------------------


def test_ramp_up_controller_initial_state_before_start():
    """``start()`` 之前 ``get_current_concurrency`` 返回 1，``is_complete`` 为 False。"""
    controller = RampUpController(target_concurrency=5, duration=10.0)
    assert controller.get_current_concurrency() == 1
    assert controller.is_complete is False
    assert controller.target == 5
    assert controller.duration == 10.0


def test_ramp_up_controller_start_sets_start_time():
    """``start()`` 后 ``_start_time`` 被设置，``is_complete`` 仍为 False。"""
    controller = RampUpController(target_concurrency=5, duration=10.0)
    controller.start()
    assert controller._start_time is not None
    assert controller.is_complete is False
    # 刚启动时 elapsed≈0，concurrency 应为 1
    assert controller.get_current_concurrency() == 1


def test_ramp_up_controller_progresses_to_target():
    """随时间推移，``get_current_concurrency`` 从 1 增加到 target。"""
    controller = RampUpController(target_concurrency=10, duration=10.0)
    # 直接操纵 _start_time 模拟时间流逝
    controller._start_time = time.time() - 5.0  # 已过 5 秒（50%）
    mid = controller.get_current_concurrency()
    assert 1 < mid <= 10, f"中间点并发数应在 (1, 10] 范围内，实际: {mid}"

    controller._start_time = time.time() - 10.0  # 已过 10 秒（100%）
    assert controller.get_current_concurrency() == 10
    assert controller.is_complete is True


def test_ramp_up_controller_completed_returns_target():
    """``is_complete`` 为 True 时，``get_current_concurrency`` 返回 target。"""
    controller = RampUpController(target_concurrency=8, duration=5.0)
    controller._start_time = time.time() - 6.0  # 超过 duration
    assert controller.is_complete is True
    assert controller.get_current_concurrency() == 8


# ---------------------------------------------------------------------------
# TTSExecutor.initialize 信号量初始化（WorkerPool 持有信号量）
# ---------------------------------------------------------------------------


def test_initialize_with_ramp_up_enabled_sets_semaphore_to_1():
    """``ramp_up_enabled=True`` 时，信号量初始限制为 1。"""
    executor = _make_executor(ramp_up_enabled=True, concurrency=5)
    _run(executor.initialize())

    assert executor._worker_pool is not None
    assert executor._worker_pool.semaphore is not None
    # 信号量内部计数器初始值为 1（只有 1 个可用许可）
    assert executor._worker_pool.semaphore._value == 1
    assert executor._ramp_up_controller is not None
    assert executor._ramp_up_controller.target == 5
    assert executor._worker_pool.ramp_up_released == 0
    # 预热控制器同时传给池
    assert executor._worker_pool.ramp_up_controller is executor._ramp_up_controller


def test_initialize_with_ramp_up_disabled_sets_semaphore_to_concurrency():
    """``ramp_up_enabled=False`` 时，信号量初始限制为目标并发数（原行为）。"""
    executor = _make_executor(ramp_up_enabled=False, concurrency=4)
    _run(executor.initialize())

    assert executor._worker_pool is not None
    assert executor._worker_pool.semaphore is not None
    assert executor._worker_pool.semaphore._value == 4
    # 未启用预热时，不应创建 RampUpController
    assert executor._ramp_up_controller is None
    assert executor._worker_pool.ramp_up_controller is None
    assert executor._worker_pool.ramp_up_released == 0


def test_initialize_resets_ramp_up_released_to_zero():
    """``initialize()`` 重建 WorkerPool，``ramp_up_released`` 归零（即使之前被修改过）。"""
    executor = _make_executor(ramp_up_enabled=True, concurrency=3)
    _run(executor.initialize())
    # 模拟之前释放过一些许可
    executor._worker_pool._ramp_up_released = 99
    # 重新初始化
    _run(executor.initialize())
    assert executor._worker_pool.ramp_up_released == 0


# ---------------------------------------------------------------------------
# WorkerPool.start 启动预热控制器与后台调整任务
# ---------------------------------------------------------------------------


def test_start_processing_starts_ramp_up_controller():
    """``WorkerPool.start`` 调用 ``ramp_up_controller.start()``。"""
    executor = _make_executor(ramp_up_enabled=True, concurrency=3)
    _run(executor.initialize())

    start_called = {"v": False}
    original_start = executor._ramp_up_controller.start

    def tracked_start():
        start_called["v"] = True
        original_start()

    executor._ramp_up_controller.start = tracked_start  # type: ignore[assignment]

    async def drive():
        await executor._worker_pool.start(progress_handler=None)
        # 清理：取消启动的 worker 和 adjuster 任务
        for w in executor._worker_pool.workers:
            if not w.done():
                w.cancel()
        await asyncio.gather(*executor._worker_pool.workers, return_exceptions=True)

    _run(drive())

    assert start_called["v"], "WorkerPool.start 必须调用 ramp_up_controller.start()"


def test_start_processing_creates_ramp_up_adjuster_task():
    """``WorkerPool.start`` 启动后台调整任务（加入 ``workers`` 列表）。"""
    executor = _make_executor(ramp_up_enabled=True, concurrency=3)
    _run(executor.initialize())

    async def drive():
        await executor._worker_pool.start(progress_handler=None)
        # workers 应包含 adjuster 任务 + concurrency 个 worker 任务
        # 即 len(workers) == concurrency + 1
        worker_count = len(executor._worker_pool.workers)
        # 清理
        for w in executor._worker_pool.workers:
            if not w.done():
                w.cancel()
        await asyncio.gather(*executor._worker_pool.workers, return_exceptions=True)
        return worker_count

    worker_count = _run(drive())
    assert worker_count == 4, (
        f"启用预热时 workers 应有 4 个任务（1 adjuster + 3 worker），实际: {worker_count}"
    )


def test_start_processing_without_ramp_up_does_not_create_adjuster():
    """``ramp_up_enabled=False`` 时不创建 adjuster 任务。"""
    executor = _make_executor(ramp_up_enabled=False, concurrency=3)
    _run(executor.initialize())

    async def drive():
        await executor._worker_pool.start(progress_handler=None)
        worker_count = len(executor._worker_pool.workers)
        for w in executor._worker_pool.workers:
            if not w.done():
                w.cancel()
        await asyncio.gather(*executor._worker_pool.workers, return_exceptions=True)
        return worker_count

    worker_count = _run(drive())
    # 未启用预热时只有 concurrency 个 worker，没有 adjuster
    assert worker_count == 3


# ---------------------------------------------------------------------------
# WorkerPool._ramp_up_adjuster 逐步释放许可
# ---------------------------------------------------------------------------


class _ScriptedRampUpController:
    """脚本化 RampUpController：按预设序列返回 ``get_current_concurrency``。

    当 ``constant`` 不为 None 时，序列耗尽后返回 ``constant`` 而非 ``target``，
    用于模拟预热中途（永远不达到目标）的场景。
    """

    def __init__(self, target: int, concurrency_sequence: list[int], *, constant: int | None = None):
        self._target = target
        self._sequence = list(concurrency_sequence)
        self._constant = constant
        self._call_count = 0
        self._started = False
        self._is_complete = False

    def start(self):
        self._started = True

    def get_current_concurrency(self) -> int:
        if self._call_count < len(self._sequence):
            val = self._sequence[self._call_count]
            self._call_count += 1
            return val
        # 序列耗尽后：如果有 constant 则返回 constant，否则返回 target
        return self._constant if self._constant is not None else self._target

    @property
    def is_complete(self) -> bool:
        return self._is_complete

    @property
    def target(self) -> int:
        return self._target

    @property
    def duration(self) -> float:
        return 30.0

    def force_complete(self):
        self._is_complete = True


def test_ramp_up_adjuster_releases_permits_gradually():
    """``_ramp_up_adjuster`` 根据 ``get_current_concurrency`` 逐步释放许可。

    初始信号量=1，controller 返回 3 时释放 2 个许可（总计 3），
    controller 返回 5 时再释放 2 个许可（总计 5）。
    """
    executor = _make_executor(ramp_up_enabled=True, concurrency=5)
    _run(executor.initialize())

    # 替换为脚本化 controller：返回 3, 5, 5, ... 然后完成
    scripted = _ScriptedRampUpController(
        target=5,
        concurrency_sequence=[3, 5],
    )
    executor._worker_pool.ramp_up_controller = scripted  # type: ignore[assignment]

    async def drive():
        # patch asyncio.sleep：替换为立即返回但仍让出事件循环控制权的版本。
        # 关键：必须 await original_sleep(0) 让出控制权，否则 adjuster 会
        # 在紧密循环中独占事件循环，其他任务无法运行。
        original_sleep = asyncio.sleep

        async def fast_sleep(seconds):
            await original_sleep(0)  # 让出事件循环控制权
            return None

        asyncio.sleep = fast_sleep  # type: ignore[assignment]
        try:
            task = asyncio.create_task(executor._worker_pool._ramp_up_adjuster())
            # 用 asyncio.wait 让 adjuster 运行若干次循环
            await asyncio.wait([task], timeout=0.05)
            # 第一轮循环后 controller 返回 3，应释放 2 个许可
            mid_value = executor._worker_pool.semaphore._value
            mid_released = executor._worker_pool.ramp_up_released
            # 标记完成，让 adjuster 退出
            scripted.force_complete()
            await asyncio.wait([task], timeout=0.05)
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            final_value = executor._worker_pool.semaphore._value
            final_released = executor._worker_pool.ramp_up_released
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]
        return mid_value, mid_released, final_value, final_released

    mid_value, mid_released, final_value, final_released = _run(drive())

    # 最终：信号量应有 5 个许可（1 初始 + 4 释放），_ramp_up_released == 4
    assert final_value == 5, f"预热完成后信号量应有 5 个许可，实际: {final_value}"
    assert final_released == 4, f"应释放 4 个许可，实际: {final_released}"


def test_ramp_up_adjuster_completes_to_target():
    """预热完成后，信号量累计释放至目标并发数。"""
    executor = _make_executor(ramp_up_enabled=True, concurrency=8)
    _run(executor.initialize())

    # controller 立即完成
    scripted = _ScriptedRampUpController(target=8, concurrency_sequence=[])
    scripted.force_complete()
    executor._worker_pool.ramp_up_controller = scripted  # type: ignore[assignment]

    async def drive():
        original_sleep = asyncio.sleep

        async def fast_sleep(seconds):
            await original_sleep(0)
            return None

        asyncio.sleep = fast_sleep  # type: ignore[assignment]
        try:
            await executor._worker_pool._ramp_up_adjuster()
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]
        return executor._worker_pool.semaphore._value, executor._worker_pool.ramp_up_released

    final_value, final_released = _run(drive())

    assert final_value == 8, f"预热完成后信号量应为 8，实际: {final_value}"
    assert final_released == 7, f"应释放 7 个许可（8-1），实际: {final_released}"


def test_ramp_up_adjuster_cancelled_still_releases_remaining():
    """``_ramp_up_adjuster`` 被取消时仍释放剩余许可至目标并发数。"""
    executor = _make_executor(ramp_up_enabled=True, concurrency=6)
    _run(executor.initialize())

    # controller 永远不完成，返回固定并发数 2（模拟预热中途被取消）
    # 使用 constant=2 确保序列耗尽后仍返回 2，而非 target（6）
    scripted = _ScriptedRampUpController(
        target=6,
        concurrency_sequence=[2, 2, 2, 2, 2],
        constant=2,
    )
    executor._worker_pool.ramp_up_controller = scripted  # type: ignore[assignment]

    async def drive():
        original_sleep = asyncio.sleep

        async def fast_sleep(seconds):
            await original_sleep(0)
            return None

        asyncio.sleep = fast_sleep  # type: ignore[assignment]
        try:
            task = asyncio.create_task(executor._worker_pool._ramp_up_adjuster())
            # 让 adjuster 运行几个循环
            await asyncio.wait([task], timeout=0.05)
            # 此时 adjuster 已释放至 2 个许可（_ramp_up_released == 1）
            mid_released = executor._worker_pool.ramp_up_released
            mid_value = executor._worker_pool.semaphore._value
            # 取消任务
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            final_value = executor._worker_pool.semaphore._value
            final_released = executor._worker_pool.ramp_up_released
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]
        return mid_released, mid_value, final_released, final_value

    mid_released, mid_value, final_released, final_value = _run(drive())

    # 取消前：信号量有 2 个许可（1 初始 + 1 释放）
    assert mid_value == 2, f"取消前信号量应有 2 个许可，实际: {mid_value}"
    assert mid_released == 1
    # 取消后：信号量释放至目标 6 个许可（1 初始 + 5 释放）
    assert final_value == 6, f"取消后信号量应释放至 6，实际: {final_value}"
    assert final_released == 5, f"取消后应释放 5 个许可（6-1），实际: {final_released}"


# ---------------------------------------------------------------------------
# _merge_segments 调用 merge_async 而非 merge（SubTask 5.2）
# ---------------------------------------------------------------------------


def test_merge_segments_calls_merge_async_not_merge(tmp_path):
    """``_merge_segments`` SHALL 调用 ``merge_async`` 而非同步 ``merge``。

    原因：同步 ``merge()`` 调用 ffmpeg 阻塞事件循环，导致并发段合成的
    WebSocket 清理被延迟，引发 "Connector is closed" 错误；同时 ffmpeg
    子进程可能收到信号被中断（error code 255）。
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from src.engines.tts_engine import TTSEngine

    cfg = get_preset("balanced")
    cfg.tts.api_concurrency = 2
    engine = TTSEngine(cfg)
    engine._api_semaphore = asyncio.Semaphore(2)

    output = tmp_path / "out.mp3"
    segments = ["段 1", "段 2"]

    async def fake_synth(text, out_path, **_kw):
        out_path.write_bytes(b"ID3")
        return MagicMock(success=True, error=None)

    engine.synthesize = fake_synth  # type: ignore[assignment]

    fake_merger = MagicMock()
    fake_merger._load_audio = MagicMock(return_value=MagicMock())
    fake_merger.merge = MagicMock(name="merge_FORBIDDEN")  # 不应被调用
    fake_merger.merge_async = AsyncMock(
        return_value=MagicMock(success=True, error=None, file_count=2)
    )
    engine._audio_merger = fake_merger  # type: ignore[assignment]

    _run(engine._merge_segments(segments, output, 0.0, disable_timeout=True))

    # 核心断言：merge_async 被调用，merge 未被调用
    assert fake_merger.merge_async.called, "必须调用 merge_async"
    assert not fake_merger.merge.called, "不应调用同步 merge（会阻塞事件循环）"
