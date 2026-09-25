"""Task 27 回归测试: 消除 executor 运行时 monkey-patch。

覆盖：
* ``TTSExecutor.add_batch`` / ``add_batch_with_progress`` 作为类的正式方法存在
  （直接从 ``tts_executor`` 模块导入，不依赖 ``tts.py`` 的 monkey-patch）
* ``SplitterExecutor.split_directory`` 作为类的正式 ``async`` 方法存在
  （直接从 ``splitter_core`` 模块导入，不依赖 ``splitter.py`` 的 monkey-patch）
* 方法可被 ``await`` 并返回正确结果
* ``add_batch`` 的 ``recursive`` 参数透传给 ``add_batch_with_progress``
  （旧 monkey-patch 实现丢弃了 ``recursive``）
* ``split_directory`` 正确 ``await`` 协程（旧 monkey-patch 返回未 await 的协程）
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

# 关键：直接从定义模块导入，不经过 tts.py / splitter.py 兼容层。
from src.executors.splitter_core import SplitterExecutor
from src.executors.tts_executor import TTSExecutor
from src.reliability import ExecutionResult, create_tts_retry_policy


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _make_tts_executor(tmp_path: Path) -> TTSExecutor:
    """构造一个已初始化、低并发、快速重试的 TTSExecutor。"""
    from src.config.presets import get_preset

    cfg = get_preset("balanced")
    cfg.tts.concurrency = 1
    cfg.reliability.tts_no_audio.delay_seconds = 0.0
    cfg.reliability.tts_no_audio.max_retries = 1
    retry = create_tts_retry_policy(
        max_retries=cfg.reliability.tts_retry.max_retries,
        base_delay=0.1,
        max_delay=1.0,
    )
    executor = TTSExecutor(cfg, retry)
    _run(executor.initialize())
    return executor


# ---------------------------------------------------------------------------
# 方法存在性 + 是否为协程函数
# ---------------------------------------------------------------------------


class TestMethodIsProperAsyncMember:
    """验证方法是直接定义在类上的 ``async def``，而非运行时挂载的同步包装。"""

    def test_add_batch_in_class_dict(self) -> None:
        assert "add_batch" in TTSExecutor.__dict__
        assert asyncio.iscoroutinefunction(TTSExecutor.__dict__["add_batch"])

    def test_add_batch_with_progress_in_class_dict(self) -> None:
        assert "add_batch_with_progress" in TTSExecutor.__dict__
        assert asyncio.iscoroutinefunction(TTSExecutor.__dict__["add_batch_with_progress"])

    def test_split_directory_in_class_dict(self) -> None:
        assert "split_directory" in SplitterExecutor.__dict__
        assert asyncio.iscoroutinefunction(SplitterExecutor.__dict__["split_directory"])

    def test_add_batch_signature_has_recursive(self) -> None:
        sig = inspect.signature(TTSExecutor.add_batch)
        assert "recursive" in sig.parameters


# ---------------------------------------------------------------------------
# TTSExecutor.add_batch_with_progress 功能
# ---------------------------------------------------------------------------


class TestAddBatchWithProgress:
    def test_processes_txt_files_and_returns_batch_result(self, tmp_path: Path) -> None:
        executor = _make_tts_executor(tmp_path)

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        (in_dir / "a.txt").write_text("hello world", encoding="utf-8")
        (in_dir / "b.txt").write_text("another text", encoding="utf-8")

        call_count = {"n": 0}

        async def fake_execute(input_path, output_path, **_kw):
            call_count["n"] += 1
            output_path.write_bytes(b"ID3\x03\x00\x00\x00")
            return ExecutionResult.ok(output_path)

        executor.execute = fake_execute  # type: ignore[assignment]

        result = _run(executor.add_batch_with_progress(in_dir, out_dir))

        assert call_count["n"] == 2
        assert result.total == 2
        assert result.succeeded == 2
        assert result.failed == 0
        assert (out_dir / "a.mp3").exists()
        assert (out_dir / "b.mp3").exists()

    def test_returns_empty_batch_when_no_txt_files(self, tmp_path: Path) -> None:
        executor = _make_tts_executor(tmp_path)

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        result = _run(executor.add_batch_with_progress(in_dir, out_dir))

        assert result.total == 0
        assert result.succeeded == 0
        assert result.failed == 0

    def test_not_initialized_raises(self, tmp_path: Path) -> None:
        from src.config.presets import get_preset

        cfg = get_preset("balanced")
        executor = TTSExecutor(cfg)
        # 未调用 initialize()
        with pytest.raises(RuntimeError, match="未初始化"):
            _run(executor.add_batch_with_progress(tmp_path, tmp_path))


# ---------------------------------------------------------------------------
# TTSExecutor.add_batch 透传 recursive
# ---------------------------------------------------------------------------


class TestAddBatchDelegatesRecursive:
    def test_add_batch_delegates_to_add_batch_with_progress(self, tmp_path: Path) -> None:
        """``add_batch`` 应委托给 ``add_batch_with_progress`` 并透传 ``recursive``。"""
        executor = _make_tts_executor(tmp_path)

        captured: dict = {}

        async def fake_abwp(input_dir, output_dir, progress_handler=None, voice=None, pattern="*.txt", recursive=False):
            captured["input_dir"] = input_dir
            captured["output_dir"] = output_dir
            captured["progress_handler"] = progress_handler
            captured["voice"] = voice
            captured["pattern"] = pattern
            captured["recursive"] = recursive
            from src.reliability.execution import BatchResult

            return BatchResult(total=0)

        executor.add_batch_with_progress = fake_abwp  # type: ignore[assignment]

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        result = _run(executor.add_batch(in_dir, out_dir, voice="zh-CN-XiaoxiaoNeural", recursive=True))

        assert captured["input_dir"] == in_dir
        assert captured["output_dir"] == out_dir
        assert captured["progress_handler"] is None
        assert captured["voice"] == "zh-CN-XiaoxiaoNeural"
        assert captured["recursive"] is True
        assert result.total == 0

    def test_add_batch_default_recursive_false(self, tmp_path: Path) -> None:
        executor = _make_tts_executor(tmp_path)

        captured: dict = {}

        async def fake_abwp(input_dir, output_dir, progress_handler=None, voice=None, pattern="*.txt", recursive=False):
            captured["recursive"] = recursive
            from src.reliability.execution import BatchResult

            return BatchResult(total=0)

        executor.add_batch_with_progress = fake_abwp  # type: ignore[assignment]

        _run(executor.add_batch(tmp_path, tmp_path))
        assert captured["recursive"] is False


# ---------------------------------------------------------------------------
# SplitterExecutor.split_directory 功能
# ---------------------------------------------------------------------------


class TestSplitDirectoryMethod:
    def test_splits_all_txt_files_in_directory(self, tmp_path: Path) -> None:
        executor = SplitterExecutor()
        _run(executor.initialize())

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        # 每章正文需超过 min_chapter_length(默认 100) 才不会被丢弃
        body = "正文内容足够长以通过最小章节长度限制。" * 10 + "\n"
        (in_dir / "book1.txt").write_text(f"第1章 标题一\n{body}第2章 标题二\n{body}", encoding="utf-8")
        (in_dir / "book2.txt").write_text(f"第1章 章节\n{body}", encoding="utf-8")

        result = _run(executor.split_directory(in_dir, out_dir))

        assert result.success is True
        # 每本书一个子目录
        assert (out_dir / "book1").is_dir()
        assert (out_dir / "book2").is_dir()
        # book1 应至少分割出 2 个章节文件
        book1_files = list((out_dir / "book1").glob("*.txt"))
        assert len(book1_files) >= 2

    def test_split_directory_is_awaitable_not_returning_raw_coroutine(self, tmp_path: Path) -> None:
        """``split_directory`` 是 ``async def``，await 后应直接返回 ExecutionResult，
        而非返回一个未 await 的协程对象。"""
        executor = SplitterExecutor()
        _run(executor.initialize())

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        (in_dir / "x.txt").write_text("第1章 测试\n正文\n", encoding="utf-8")

        result = _run(executor.split_directory(in_dir, out_dir))

        # 不应是协程（旧 monkey-patch 的 sync 包装在某些误用场景下可能返回协程）
        assert not asyncio.iscoroutine(result)
        assert hasattr(result, "success")

    def test_not_initialized_raises(self, tmp_path: Path) -> None:
        executor = SplitterExecutor()
        # 未调用 initialize()
        with pytest.raises(RuntimeError, match="未初始化"):
            _run(executor.split_directory(tmp_path, tmp_path))
