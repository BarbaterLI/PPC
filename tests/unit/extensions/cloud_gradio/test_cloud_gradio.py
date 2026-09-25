"""cloud_gradio 插件单测。

验证：
- 服务端返回真实音频路径时，成功路径把音频复制到 output_path 并返回 4 元组。
- 服务端状态消息含失败信息时，插件透传该错误而非通用提示。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.extensions.cloud_gradio.extension import CloudGradioBackend


def _run(coro):
    return asyncio.run(coro)


def _make_backend():
    backend = CloudGradioBackend()
    _run(backend.initialize())
    return backend


def test_success_path_copies_audio(tmp_path: Path) -> None:
    """服务端返回真实音频路径时，success 路径复制音频并返回 4 元组。"""
    backend = _make_backend()
    src_audio = tmp_path / "server_audio.mp3"
    src_audio.write_bytes(b"ID3 fake audio")
    out = tmp_path / "out.mp3"

    fake_client = MagicMock()
    fake_client.predict.return_value = (str(src_audio), None, "ok", [])

    with patch("gradio_client.Client", return_value=fake_client):
        result = _run(backend.synthesize("你好", "zh-CN-YunxiNeural", rate="+0%", output_path=out))

    assert isinstance(result, tuple) and len(result) == 4
    audio, sub, status, meta = result
    assert status == "ok"
    assert out.exists(), "音频应被复制到 output_path"
    assert out.read_bytes() == b"ID3 fake audio"

    # 验证 predict 参数映射
    args, kwargs = fake_client.predict.call_args
    assert len(args) == 10, "synthesize 应传 10 个位置参数"
    assert kwargs.get("api_name") == "/synthesize"
    _run(backend.cleanup())


def test_server_error_status_raised(tmp_path: Path) -> None:
    """服务端状态消息含失败信息时，透传该错误（NetworkError）。"""
    backend = _make_backend()
    out = tmp_path / "out.mp3"

    fake_client = MagicMock()
    fake_client.predict.return_value = (
        None,
        None,
        "合成失败: Connection timeout to host wss://...",
        [],
    )

    from src.core.exceptions import NetworkError

    with patch("gradio_client.Client", return_value=fake_client), pytest.raises(NetworkError):
        _run(backend.synthesize("你好", "zh-CN-YunxiNeural", output_path=out))

    assert not out.exists(), "失败时不应产生输出文件"
    _run(backend.cleanup())


def test_no_audio_path_raises(tmp_path: Path) -> None:
    """服务端未返回音频路径且无失败状态时，抛出通用错误。"""
    backend = _make_backend()
    fake_client = MagicMock()
    fake_client.predict.return_value = (None, None, "", [])

    with patch("gradio_client.Client", return_value=fake_client), pytest.raises(RuntimeError, match="未返回音频文件路径"):
        _run(backend.synthesize("你好", "zh-CN-YunxiNeural"))
    _run(backend.cleanup())
