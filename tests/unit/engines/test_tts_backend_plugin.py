"""TTS_BACKEND 插件后端选择测试。

验证 EdgeTTSHttpClient 在配置了 backend_plugin 且注入可用插件时走插件合成，
否则走默认路径。
"""

from __future__ import annotations

import asyncio


class _FakeBackend:
    """假 TTSBackend：is_available 恒 True，synthesize 写假音频并记录调用。"""

    def __init__(self, name: str = "fake_backend"):
        self.name = name
        self.called = False
        self.last_text = None

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        return True

    async def synthesize(self, text, voice, *, rate="+0%", volume="+0%", pitch="+0Hz", output_path=None, **kwargs):
        self.called = True
        self.last_text = text
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"ID3\x03\x00\x00\x00")
            return output_path
        return b"ID3\x03\x00\x00\x00"


class _UnavailableBackend:
    """不可用后端：is_available 恒 False。"""

    def __init__(self, name: str = "unavailable"):
        self.name = name
        self.called = False

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        return False

    async def synthesize(self, text, voice, *, rate="+0%", volume="+0%", pitch="+0Hz", output_path=None, **kwargs):
        self.called = True
        return None


def _make_client(plugins=None, backend_plugin=None):
    from src.engines.http_client import EdgeTTSHttpClient

    return EdgeTTSHttpClient(
        tts_backend_plugins=plugins or [],
        backend_plugin=backend_plugin,
    )


def test_backend_plugin_called_when_selected(tmp_path):
    """指定 backend_plugin 且存在可用插件时，synthesize_to_file 应走插件路径。"""
    backend = _FakeBackend()
    client = _make_client(plugins=[backend], backend_plugin="fake_backend")
    out = tmp_path / "out.mp3"

    async def run():
        return await client.synthesize_to_file("hello", out, "voice")

    n = asyncio.run(run())

    assert backend.called, "指定 backend_plugin 时应调用插件 synthesize"
    assert backend.last_text == "hello"
    assert out.exists()
    assert n > 0


def test_backend_plugin_not_called_when_not_selected(tmp_path):
    """未指定 backend_plugin 时走默认路径，不调用插件。"""
    backend = _FakeBackend()
    client = _make_client(plugins=[backend], backend_plugin=None)
    out = tmp_path / "out.mp3"

    async def run():
        return await client.synthesize_to_file("hello", out, "voice")

    # 默认路径会尝试真实网络/本地 edge_tts；这里只验证插件未被调用。
    # 用 monkeypatch 掉默认路径，或直接断言插件未被调用。
    try:
        asyncio.run(run())
    except Exception:
        # 默认路径连不上网络属预期；关键是插件未参与。
        pass

    assert not backend.called, "未指定 backend_plugin 时不应调用插件"


def test_backend_plugin_unavailable_falls_back(tmp_path):
    """指定后端不可用时，_resolve_tts_backend 返回 None（走默认路径），插件不被调用。"""
    backend = _UnavailableBackend()
    client = _make_client(plugins=[backend], backend_plugin="unavailable")

    resolved = client._resolve_tts_backend("unavailable")

    assert resolved is None
    assert not backend.called


# ---------------------------------------------------------------- 注册表回退（ppc.tts.v1）


def test_registry_fallback_routes_synthesis(tmp_path):
    """注入列表未命中时应回落到全局注册表并按 ppc.tts.v1 协商。"""
    from src.engines.tts_protocol import backend_registry

    backend = _FakeBackend(name="reg-fallback-backend")
    backend_registry.register("reg-fallback-backend", backend, source="test")
    try:
        client = _make_client(plugins=[], backend_plugin="reg-fallback-backend")
        out = tmp_path / "via-registry.mp3"

        async def run():
            return await client.synthesize_to_file("hi", out, "v")

        n = asyncio.run(run())
        assert backend.called and n > 0 and out.exists()
    finally:
        backend_registry.unregister("reg-fallback-backend")


def test_list_voices_routes_to_plugin_with_voices_capability():
    """后端声明 voices 能力时，list_voices 走后端而非 edge 服务。"""
    from src.engines.tts_protocol import backend_registry

    class _VoicesBackend(_FakeBackend):
        def capabilities(self):
            return {"voices": True, "probe": False}

        async def list_voices(self):
            return [
                {"name": "af_bella", "short_name": "af_bella", "gender": "Female", "locale": "en", "friendly_name": "Bella"},
                {"name": "am_adam", "short_name": "am_adam", "gender": "Male", "locale": "en", "friendly_name": "Adam"},
            ]

    backend = _VoicesBackend(name="voices-backend")
    backend_registry.register("voices-backend", backend, source="test")
    try:
        client = _make_client(backend_plugin="voices-backend")
        voices = asyncio.run(client.list_voices(gender="female"))
        assert [v.short_name for v in voices] == ["af_bella"]
        assert voices[0].friendly_name == "Bella"
    finally:
        backend_registry.unregister("voices-backend")


def test_config_backend_plugin_field():
    """TTSConfig.backend_plugin 默认空串且可赋值（旧配置文件无此键可正常加载）。"""
    from src.config.schemas.tts import TTSConfig

    assert TTSConfig().backend_plugin == ""
    assert TTSConfig(backend_plugin="openai_http").backend_plugin == "openai_http"