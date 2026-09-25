"""synthesize_to_file 原子写入测试。

覆盖三种路径：
1. 成功：临时文件 ``.part`` 通过 ``os.replace`` 原子重命名为目标路径
2. 失败：临时文件被删除，目标路径不存在
3. 取消（``CancelledError``）：临时文件被删除，``CancelledError`` 重新抛出

约束：所有测试 SHALL 使用 ``tmp_path`` fixture，绝不写入 ``ms_tts`` 目录。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.core.exceptions import TransientError
from src.engines.http_client import EdgeTTSHttpClient


def _run(coro):
    return asyncio.run(coro)


class _ScriptedCommunicate:
    """脚本化 Communicate：按预设脚本产出 chunk 或抛异常。

    通过 ``initial_delay`` 模拟网络延迟，确保取消测试有足够的 await 点。
    """

    def __init__(
        self,
        text: str,
        voice: str,
        rate: str = "+0%",
        volume: str = "+0%",
        *,
        chunks: list[bytes] | None = None,
        fail_after: int | None = None,
        fail_with: BaseException | None = None,
        initial_delay: float = 0.0,
    ) -> None:
        self.text = text
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.chunks = chunks or [b"\x00" * 16, b"\x11" * 16]
        self.fail_after = fail_after
        self.fail_with = fail_with
        self.initial_delay = initial_delay
        self.last_selected_ua = None

    def stream(self):
        chunks = self.chunks
        fail_after = self.fail_after
        fail_with = self.fail_with
        initial_delay = self.initial_delay

        async def _gen():
            if initial_delay > 0:
                await asyncio.sleep(initial_delay)
            for i, c in enumerate(chunks):
                if fail_after is not None and i >= fail_after:
                    if fail_with is not None:
                        raise fail_with
                yield {"type": "audio", "data": c}
            yield {"type": "turn.end"}

        return _gen()

    async def save(self, path: str) -> None:  # pragma: no cover - unused
        pass


def _make_client(
    *,
    chunks: list[bytes] | None = None,
    fail_after: int | None = None,
    fail_with: BaseException | None = None,
    initial_delay: float = 0.0,
) -> EdgeTTSHttpClient:
    def communicate_factory(text, voice, rate, volume):
        return _ScriptedCommunicate(
            text=text,
            voice=voice,
            rate=rate,
            volume=volume,
            chunks=chunks,
            fail_after=fail_after,
            fail_with=fail_with,
            initial_delay=initial_delay,
        )

    return EdgeTTSHttpClient(
        communicate_factory=communicate_factory,
        list_voices_factory=None,
    )


# ---------------------------------------------------------------------------
# Scenario 1: 合成成功 — 临时文件原子重命名
# ---------------------------------------------------------------------------


def test_synthesize_to_file_success_atomic_rename(tmp_path: Path):
    """成功路径：``.part`` 临时文件通过 ``os.replace`` 重命名为目标路径。"""
    client = _make_client(chunks=[b"AAAA", b"BBBB", b"CCCC"])
    output = tmp_path / "out.mp3"
    temp = output.with_suffix(output.suffix + ".part")

    written = _run(client.synthesize_to_file("hello", output, "zh-CN-XiaoxiaoNeural"))

    assert output.exists(), "成功后目标文件必须存在"
    assert not temp.exists(), "临时文件 .part 必须已被重命名/删除"
    assert written == len(b"AAAA") + len(b"BBBB") + len(b"CCCC")
    content = output.read_bytes()
    assert content == b"AAAABBBBCCCC"


def test_synthesize_to_file_creates_parent_dirs(tmp_path: Path):
    """父目录不存在时，``mkdir(parents=True, exist_ok=True)`` 会被调用。"""
    client = _make_client(chunks=[b"DATA"])
    output = tmp_path / "nested" / "deep" / "out.mp3"

    _run(client.synthesize_to_file("hello", output, "zh-CN-XiaoxiaoNeural"))

    assert output.exists()
    assert output.read_bytes() == b"DATA"


# ---------------------------------------------------------------------------
# Scenario 2: 合成中途失败 — 临时文件被删除，目标路径不存在
# ---------------------------------------------------------------------------


def test_synthesize_to_file_failure_deletes_temp(tmp_path: Path):
    """合成中途抛异常时，``.part`` 被删除，目标路径不存在。

    注意：``synthesize_stream`` 会把底层 ``RuntimeError`` 包装为
    ``TransientError``，所以测试期望 ``TransientError``。
    """
    client = _make_client(
        chunks=[b"AAAA", b"BBBB"],
        fail_after=1,
        fail_with=RuntimeError("server error mid-stream"),
    )
    output = tmp_path / "out.mp3"
    temp = output.with_suffix(output.suffix + ".part")

    with pytest.raises(TransientError, match="server error mid-stream"):
        _run(client.synthesize_to_file("hello", output, "zh-CN-XiaoxiaoNeural"))

    assert not temp.exists(), "失败时临时文件 .part 必须被删除"
    assert not output.exists(), "失败时目标路径必须不存在（避免缓存复用部分 MP3）"


def test_synthesize_to_file_failure_does_not_touch_existing_target(tmp_path: Path):
    """若目标路径已存在（例如上一次成功的输出），失败时不应破坏它。"""
    client = _make_client(
        chunks=[b"AAAA"],
        fail_after=0,
        fail_with=RuntimeError("immediate failure"),
    )
    output = tmp_path / "out.mp3"
    output.write_bytes(b"PREVIOUS_GOOD_CONTENT")
    temp = output.with_suffix(output.suffix + ".part")

    with pytest.raises(TransientError):
        _run(client.synthesize_to_file("hello", output, "zh-CN-XiaoxiaoNeural"))

    assert not temp.exists(), "失败时临时文件 .part 必须被删除"
    # 已存在的目标文件不被破坏（这次失败未产生新输出）
    assert output.read_bytes() == b"PREVIOUS_GOOD_CONTENT"


# ---------------------------------------------------------------------------
# Scenario 3: 合成被取消 — CancelledError 重新抛出，临时文件被删除
# ---------------------------------------------------------------------------


def test_synthesize_to_file_cancelled_deletes_temp(tmp_path: Path):
    """``CancelledError`` 时临时文件被删除，``CancelledError`` 重新抛出。

    ``CancelledError`` 是 ``BaseException`` 的子类而非 ``Exception``，
    必须用 ``except BaseException`` 才能捕获并清理。
    """
    # 使用 initial_delay 确保任务在等待第一个 chunk 时被取消
    client = _make_client(
        chunks=[b"AAAA", b"BBBB", b"CCCC"],
        initial_delay=0.3,
    )
    output = tmp_path / "out.mp3"
    temp = output.with_suffix(output.suffix + ".part")

    async def drive():
        task = asyncio.create_task(
            client.synthesize_to_file("hello", output, "zh-CN-XiaoxiaoNeural")
        )
        # 等待任务进入 stream 等待状态
        await asyncio.sleep(0.05)
        task.cancel()
        await task  # 期望抛出 CancelledError

    with pytest.raises(asyncio.CancelledError):
        _run(drive())

    assert not temp.exists(), "CancelledError 时临时文件 .part 必须被删除"
    assert not output.exists(), "CancelledError 时目标路径必须不存在"


def test_synthesize_to_file_cancelled_propagates_cancellederror(tmp_path: Path):
    """``CancelledError`` 必须被原样重新抛出，而非被吞掉或转成其他异常。"""
    client = _make_client(
        chunks=[b"AAAA" * 1000],
        initial_delay=0.3,
    )
    output = tmp_path / "out.mp3"

    async def drive():
        task = asyncio.create_task(
            client.synthesize_to_file("hello", output, "zh-CN-XiaoxiaoNeural")
        )
        await asyncio.sleep(0.05)
        task.cancel()
        await task

    # 不能用 pytest.raises(Exception) — CancelledError 不是 Exception 的子类
    with pytest.raises(asyncio.CancelledError):
        _run(drive())


# ---------------------------------------------------------------------------
# Scenario 4: 断点续传模式 — 复用已有文件内容初始化临时文件
# ---------------------------------------------------------------------------


def test_synthesize_to_file_resume_mode_copies_existing_to_temp(tmp_path: Path):
    """断点续传模式（``last_chunk_offset>0`` 且目标已存在）下，
    临时文件以已有文件内容初始化，新 chunk 追加写入。

    注意：``synthesize_stream`` 会跳过 ``chunk.offset < last_chunk_offset``
    的 chunk。因此 fake stream 必须先产出足够的数据使 offset 达到
    ``last_chunk_offset``，之后的 chunk 才会被写入。
    """
    # chunks: b"AAAA" (offset 0, 被跳过) + b"CCCC" (offset 4, 被写入)
    client = _make_client(chunks=[b"AAAA", b"CCCC"])
    output = tmp_path / "out.mp3"
    output.write_bytes(b"AAAA")  # 已有 4 字节内容
    temp = output.with_suffix(output.suffix + ".part")

    # last_chunk_offset=4 触发续传路径：跳过 offset < 4 的 chunk
    written = _run(
        client.synthesize_to_file(
            "hello",
            output,
            "zh-CN-XiaoxiaoNeural",
            last_chunk_offset=4,
        )
    )

    assert output.exists()
    assert not temp.exists()
    # 续传模式下，临时文件初始化为已有内容 + 新 chunk
    assert output.read_bytes() == b"AAAACCCC"
    # 返回值是本次新写入的字节数（不含已有内容）
    assert written == len(b"CCCC")


def test_synthesize_to_file_resume_failure_deletes_temp(tmp_path: Path):
    """断点续传模式下失败时，临时文件被删除，原目标文件保留。"""
    client = _make_client(
        chunks=[b"AAAA", b"CCCC"],
        fail_after=1,
        fail_with=RuntimeError("resume failure"),
    )
    output = tmp_path / "out.mp3"
    output.write_bytes(b"AAAA")
    temp = output.with_suffix(output.suffix + ".part")

    with pytest.raises(TransientError, match="resume failure"):
        _run(
            client.synthesize_to_file(
                "hello",
                output,
                "zh-CN-XiaoxiaoNeural",
                last_chunk_offset=4,
            )
        )

    assert not temp.exists(), "续传失败时临时文件必须被删除"
    # 原目标文件不被破坏
    assert output.read_bytes() == b"AAAA"
