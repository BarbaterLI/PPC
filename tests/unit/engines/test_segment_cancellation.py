"""段并发合成的结构化并发语义（12.x P2-2 回归）。

修复前：``asyncio.gather(..., return_exceptions=False)`` 在某段抛异常时立即
向上抛，**其余段任务不被取消**，继续在后台写 ``cache_dir``；上层重试时会与
残留任务并发写同一缓存目录。

修复后：任一任务异常 → 取消其余任务并等待其真正退出（无残留写盘），异常
仍按原语义上抛；段返回 ``fail`` 结果（非异常）时依然保持「部分成功」语义，
不会误取消兄弟任务。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from src.engines.tts_engine import SegmentMerger
from src.reliability import ExecutionResult


class _FakeEngine:
    """只实现 SegmentMerger 依赖的两个方法的假引擎。"""

    def __init__(self, *, raise_on: int | None = None, fail_on: int | None = None, hang: bool = True) -> None:
        self.raise_on = raise_on
        self.fail_on = fail_on
        self.hang = hang
        self.started: list[int] = []
        self.finished: list[int] = []
        self.cancelled: list[int] = []

    async def _synthesize_segment(
        self,
        index: int,
        segment: str,
        *,
        cache_dir: Path,
        disable_timeout: bool,
        progress_handler: Any | None,
        journal: Any | None = None,
        voice: str | None = None,
    ) -> tuple[int, ExecutionResult]:
        self.started.append(index)
        if index == self.raise_on:
            raise RuntimeError(f"段 {index} 爆炸")
        if index == self.fail_on:
            return index, ExecutionResult.fail(error="网络错误", error_code=f"SEGMENT_{index}_FAILED")
        try:
            # 模拟"正在合成"的任务：hang=True 时永不自然结束，只能被取消，
            # 从而可确定性地断言"兄弟任务被取消、未留下段文件"。
            if self.hang:
                await asyncio.Event().wait()
            else:
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled.append(index)
            raise
        target = cache_dir / f"{cache_dir.name}_seg_{index:03d}.mp3"
        target.write_bytes(b"ID3audio")
        self.finished.append(index)
        return index, ExecutionResult.ok(target)

    async def _finalize_merged_audio(self, output_path: Path, cache_dir: Path, temp_files, start_time):
        output_path.write_bytes(b"merged")
        return ExecutionResult.ok(output_path)


def _merge(engine: _FakeEngine, segments: list[str], output_path: Path):
    merger = SegmentMerger(engine)  # type: ignore[arg-type]
    return asyncio.run(merger.merge_segments(segments, output_path, 0.0, disable_timeout=True))


def test_exception_cancels_sibling_tasks_and_leaves_no_files(tmp_path: Path) -> None:
    """一段抛异常 → 其余在途段被取消（无残留写盘），异常正确上抛为失败结果。"""
    engine = _FakeEngine(raise_on=2)
    result = _merge(engine, ["段一", "段二", "段三"], tmp_path / "out.mp3")

    assert result.success is False
    assert "段 2 爆炸" in (result.error or "")

    # 兄弟任务被取消并已退出：不会在 merge 返回后继续写盘
    assert sorted(engine.cancelled) == [1, 3]
    assert engine.finished == []
    assert [p for p in tmp_path.rglob("*_seg_*.mp3")] == []


def test_segment_failure_result_keeps_partial_success_semantics(tmp_path: Path) -> None:
    """段返回 fail 结果（非异常）时不取消兄弟任务 —— 保持既有部分成功语义。"""
    engine = _FakeEngine(fail_on=2, hang=False)
    result = _merge(engine, ["段一", "段二", "段三"], tmp_path / "out.mp3")

    assert result.success is False
    assert "分段 2" in (result.error or "")
    assert engine.cancelled == []  # 没有误取消
    assert sorted(engine.finished) == [1, 3]  # 其余段正常完成
