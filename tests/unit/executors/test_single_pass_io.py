"""12.0.0 W2-B 单遍 IO 管线守卫。

断言批量转换每个输入文件只发生 **一次全量读取**：

* ``_collect_input_tasks`` 阶段 ``read_text_auto`` 读一次并预缓存文本；
* ``execute()`` 阶段命中预缓存 → 不再触发第二次 ``Path.read_text``；
* 编码探测为一次轻量头部 probe（``Path.open``），``detect_encoding``
  热路径调用数为 0。
"""

from __future__ import annotations

import asyncio
import pathlib
from pathlib import Path

from src.config import PPCConfig
from src.executors import TTSExecutor
from src.reliability import ExecutionResult


def _make_corpus(tmp_path: Path, n: int = 6) -> tuple[Path, Path]:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    for i in range(n):
        (in_dir / f"c{i:02d}.txt").write_text(f"第{i}章内容，" * 60, encoding="utf-8")
    return in_dir, out_dir


class _HoldWorkerPool:
    """替身池：enqueue 只记录，不消费——模拟 worker 尚未启动的收集阶段。"""

    def __init__(self) -> None:
        self.queued: list = []

    async def enqueue(self, task) -> None:
        self.queued.append(task)

    async def start(self, progress_handler=None) -> None:
        pass

    async def wait_all(self) -> None:
        while self.queued:
            task = self.queued.pop(0)
            await self.processor(task, "w1")

    async def shutdown(self) -> None:
        pass

    def clear_workers(self) -> None:  # 与真实 WorkerPool 一致：同步方法
        pass


def test_batch_single_full_read_per_file(tmp_path: Path, monkeypatch) -> None:
    in_dir, out_dir = _make_corpus(tmp_path, n=6)
    reads: list[str] = []
    probes: list[str] = []

    orig_read_text = pathlib.Path.read_text
    orig_open = pathlib.Path.open

    corpus = str(in_dir)

    def read_text_counter(self, *a, **kw):
        if str(self).startswith(corpus):
            reads.append(str(self))
        return orig_read_text(self, *a, **kw)

    def open_counter(self, *a, **kw):
        if str(self).startswith(corpus):
            probes.append(str(self))
        return orig_open(self, *a, **kw)

    monkeypatch.setattr(pathlib.Path, "read_text", read_text_counter)
    monkeypatch.setattr(pathlib.Path, "open", open_counter)

    import src.utils.files as files_mod

    detect_calls: list[str] = []
    orig_detect = files_mod.detect_encoding

    def detect_counter(p, *a, **kw):
        detect_calls.append(str(p))
        return orig_detect(p, *a, **kw)

    monkeypatch.setattr(files_mod, "detect_encoding", detect_counter)

    cfg = PPCConfig()
    cfg.tts.concurrency = 4
    cfg.tts.ramp_up_enabled = False

    async def synth_stub(self, text, output_path, **kw):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00")
        return ExecutionResult.ok(output_path)

    from src.engines.tts_engine import TTSEngine

    monkeypatch.setattr(TTSEngine, "synthesize_segmented", synth_stub)

    async def _run():
        ex = TTSExecutor(cfg)
        await ex.initialize()
        pool = _HoldWorkerPool()
        pool.processor = ex._execute_task_with_retry
        ex._worker_pool = pool  # type: ignore[assignment]
        batch = await ex.add_batch_with_progress(in_dir, out_dir)
        await ex.cleanup()
        return batch

    batch = asyncio.run(_run())
    assert batch.total == 6

    # 每个文件恰好一次全量 Path.read_text（收集阶段预读，执行阶段复用）
    assert sorted(reads) == sorted(str(in_dir / f"c{i:02d}.txt") for i in range(6)), reads
    # 热路径零 detect_encoding 调用（编码判定走 quick_probe 头部探测）
    assert not detect_calls
    # 每文件至多两次轻量 open：quick_probe 探测 + read_text_auto 全读
    assert max(probes.count(p) for p in set(probes)) <= 2


def test_execute_reuses_prefetched_text(tmp_path: Path, monkeypatch) -> None:
    """显式预读缓存命中时 execute() 不再触碰磁盘读取。"""
    src_file = tmp_path / "a.txt"
    text = "复用测试文本。" * 50
    src_file.write_text(text, encoding="utf-8")

    cfg = PPCConfig()

    class _StubEngine:
        async def synthesize_segmented(self, t, output_path, **kw):
            assert t == text
            output_path.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00")
            return ExecutionResult.ok(output_path)

        async def cleanup(self) -> None:
            pass

    def _boom(*a, **kw):
        raise AssertionError("缓存命中后不应再读取磁盘")

    monkeypatch.setattr(pathlib.Path, "read_text", _boom)

    async def _run():
        ex = TTSExecutor(cfg)
        await ex.initialize()
        ex._engine = _StubEngine()

        ex._remember_text(str(src_file), text)
        result = await ex.execute(src_file, tmp_path / "a.mp3")
        return result

    result = asyncio.run(_run())
    assert result.success
