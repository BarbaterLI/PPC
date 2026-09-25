"""Task 35 回归测试: add_batch_with_progress 编码检测一致性。

覆盖：
* ``add_batch_with_progress`` 对非 UTF-8 (GBK) 文件使用 ``detect_encoding``
  计算 ``text_len``, 不再硬编码 UTF-8 (与 ``TTSExecutor.execute`` 一致)

重构后 ``add_batch_with_progress`` 是 ``TTSExecutor`` 的方法（原
``tts_segment`` 模块级函数已并入执行器），本测试用真实执行器 + 排空型
``WorkerPool.start`` 替身驱动（不启动真实 worker）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.config import PPCConfig
from src.executors.tts_executor import TTSExecutor
from src.executors.tts_worker_pool import WorkerPool


def _run(coro):
    return asyncio.run(coro)


def _make_executor(voice: str = "zh-CN-XiaoxiaoNeural") -> TTSExecutor:
    cfg = PPCConfig()
    cfg.tts.voice = voice
    executor = TTSExecutor(cfg)
    executor._initialized = True
    executor._worker_pool = WorkerPool(concurrency=1, processor=None)
    return executor


async def _drain_start(self, progress_handler=None):
    """替身 WorkerPool.start: 排空队列避免 join() 挂起 (不启动真实 worker)。"""
    self._progress_handler = progress_handler
    while not self.queue.empty():
        try:
            self.queue.get_nowait()
            self.queue.task_done()
        except asyncio.QueueEmpty:
            break


class TestTtsSegmentEncoding:
    def test_gbk_file_text_len_uses_detect_encoding(self, tmp_path: Path, monkeypatch) -> None:
        """GBK 文件应通过 detect_encoding 正确解码并计算 text_len (Task 35)。"""
        monkeypatch.setattr(WorkerPool, "start", _drain_start)

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        txt_file = input_dir / "chapter.txt"
        content = "你好世界"
        txt_file.write_bytes(content.encode("gbk"))

        executor = _make_executor()

        result = _run(
            executor.add_batch_with_progress(input_dir, output_dir, pattern="*.txt")
        )

        assert result.total == 1
        assert len(executor._tasks) == 1
        task = next(iter(executor._tasks.values()))
        # GBK 解码后 "你好世界" 为 4 个字符
        assert task.text_len == len(content)

    def test_utf8_file_text_len_still_correct(self, tmp_path: Path, monkeypatch) -> None:
        """UTF-8 文件 text_len 计算不受影响 (Task 35 回归)。"""
        monkeypatch.setattr(WorkerPool, "start", _drain_start)

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        txt_file = input_dir / "chapter.txt"
        content = "hello world 你好"
        txt_file.write_text(content, encoding="utf-8")

        executor = _make_executor()

        result = _run(
            executor.add_batch_with_progress(input_dir, output_dir, pattern="*.txt")
        )

        assert result.total == 1
        task = next(iter(executor._tasks.values()))
        assert task.text_len == len(content)

    def test_gbk_file_does_not_raise(self, tmp_path: Path, monkeypatch) -> None:
        """GBK 文件在旧实现下会抛 UnicodeDecodeError, 修复后应正常 (Task 35)。"""
        monkeypatch.setattr(WorkerPool, "start", _drain_start)

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        txt_file = input_dir / "gbk.txt"
        # 纯中文 GBK 内容, UTF-8 解码必失败
        txt_file.write_bytes("中文测试内容".encode("gbk"))

        executor = _make_executor()

        # 不应抛出 UnicodeDecodeError
        result = _run(
            executor.add_batch_with_progress(input_dir, output_dir, pattern="*.txt")
        )
        assert result.total == 1
