"""``ppc.tts.v1`` 协议层测试：版本协商、能力声明、惰性注册表。"""

from __future__ import annotations

import asyncio

import pytest

from src.engines.tts_protocol import (
    PROTOCOL,
    BackendCapabilities,
    BackendRegistry,
    capabilities_of,
    negotiate,
    parse_protocol_version,
)


class _Good:
    def get_name(self):
        return "good"

    def is_available(self):
        return True

    async def synthesize(self, text, voice, **kw):  # noqa: ARG002
        return b"x"


class _NoSynth:
    def get_name(self):
        return "no-synth"

    def is_available(self):
        return True


class _SyncSynth:
    def get_name(self):
        return "sync"

    def is_available(self):
        return True

    def synthesize(self, text, voice, **kw):  # noqa: ARG002
        return b"x"


class _FutureVersion(_Good):
    def protocol_version(self):
        return "ppc.tts.v9"


class _BrokenVersion(_Good):
    def protocol_version(self):
        return object()


class _ExplodingVersion(_Good):
    def protocol_version(self):
        raise RuntimeError("boom")


# ---------------------------------------------------------------- 版本解析


def test_parse_protocol_version_forms():
    assert parse_protocol_version("ppc.tts.v1") == ("ppc.tts", 1)
    assert parse_protocol_version("v2") == ("ppc.tts", 2)
    assert parse_protocol_version("3") == ("ppc.tts", 3)
    assert parse_protocol_version(1) == ("ppc.tts", 1)
    assert parse_protocol_version("") is None
    assert parse_protocol_version("mp3-v1") is None
    assert parse_protocol_version(None) is None


# ---------------------------------------------------------------- 协商矩阵


def test_negotiate_accepts_v1_and_legacy():
    assert negotiate(_Good()) == "good"  # 无 protocol_version → v1 基线


def test_negotiate_rejects_missing_members():
    assert negotiate(_NoSynth()) is None
    assert negotiate(object()) is None


def test_negotiate_rejects_sync_synthesize():
    assert negotiate(_SyncSynth()) is None


def test_negotiate_rejects_future_major_and_bad_versions():
    assert negotiate(_FutureVersion()) is None
    assert negotiate(_BrokenVersion()) is None
    assert negotiate(_ExplodingVersion()) is None


# ---------------------------------------------------------------- 能力声明


def test_capabilities_from_dataclass_and_dict_and_probe():
    class DC(_Good):
        def capabilities(self):
            return BackendCapabilities(voices=True)

    class Dict(_Good):
        def capabilities(self):
            return {"voices": True, "probe": False, "extra": ["word_marks"]}

    class Bare(_Good):
        async def list_voices(self):
            return []

    assert capabilities_of(DC()).voices is True
    assert capabilities_of(DC()).probe is False
    cap = capabilities_of(Dict())
    assert cap.voices is True and cap.extra == ("word_marks",)
    assert capabilities_of(Bare()).voices is True  # 方法探测回退
    assert capabilities_of(_Good()).voices is False


def test_abc_capabilities_report_honestly():
    from src.extensions.base.interfaces.audio import TTSBackend

    class Bare(TTSBackend):
        def get_name(self):
            return "bare"

        def is_available(self):
            return True

        async def synthesize(self, text, voice, **kw):  # noqa: ARG002
            return b""

    class WithVoices(Bare):
        async def list_voices(self):
            return [{"name": "v"}]

    caps = Bare().capabilities()
    assert caps == {"voices": False, "probe": False}
    assert Bare().protocol_version() == PROTOCOL
    assert asyncio.run(Bare().list_voices()) == []
    ok, _msg = asyncio.run(Bare().probe())
    assert ok is True
    assert WithVoices().capabilities()["voices"] is True


# ---------------------------------------------------------------- 注册表


def test_registry_lazy_factory_and_dedupe():
    reg = BackendRegistry()
    calls = []

    def factory():
        calls.append(1)
        return _Good()

    reg.register("good", factory, source="test")
    assert calls == []  # 未解析不构造
    a = reg.get("good")
    b = reg.get("good")
    assert a is b and calls == [1]
    assert reg.names(source="test") == ["good"]
    assert reg.names(source="builtin") == []


def test_registry_resolve_gates_negotiation_and_availability():
    reg = BackendRegistry()
    reg.register("good", _Good(), source="test")
    reg.register("future", _FutureVersion(), source="test")

    class Unavail(_Good):
        def is_available(self):
            return False

    reg.register("unavail", Unavail(), source="test")
    assert reg.resolve("good") is not None
    assert reg.resolve("future") is None
    assert reg.resolve("unavail") is None
    assert reg.resolve("missing") is None
    assert reg.resolve(None) is None


def test_registry_factory_error_swallowed():
    reg = BackendRegistry()

    def boom():
        raise RuntimeError("x")

    reg.register("boom", boom, source="test")
    assert reg.get("boom") is None
    assert reg.get("boom") is None  # 幂等（工厂失败不缓存半截实例）


def test_registry_unregister():
    reg = BackendRegistry()
    reg.register("a", _Good(), source="s1")
    reg.register("b", _Good(), source="s2")
    reg.unregister("a")
    reg.unregister_source("s2")
    assert reg.names() == []
    assert reg.get("a") is None


# ---------------------------------------------------------------- 内置注册


def test_builtin_backends_register_openai_http():
    from src.engines.tts_protocol import backend_registry, ensure_builtin_backends

    ensure_builtin_backends()
    ensure_builtin_backends()  # 幂等
    assert "openai_http" in backend_registry.names(source="builtin")
    plugin = backend_registry.resolve("openai_http")
    assert plugin is not None
    assert plugin.get_name() == "openai_http"
    assert negotiate(plugin) == "openai_http"


def test_openai_http_synthesize_payload_and_file(tmp_path, monkeypatch):
    from src.engines.backends.openai_http import OpenAIHTTPBackend

    be = OpenAIHTTPBackend({"base_url": "http://x/v1", "api_key": "k", "model": "m"})
    captured = {}

    async def fake_post(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return b"ID3\x04\x00\x00fakeaudio"

    monkeypatch.setattr(be, "_post_bytes", fake_post)
    out = tmp_path / "seg.mp3"
    result = asyncio.run(be.synthesize("你好", "af_bella", rate="+20%", output_path=out))
    assert result == out and out.read_bytes().startswith(b"ID3")
    assert captured["path"] == "/audio/speech"
    assert captured["payload"] == {
        "model": "m", "input": "你好", "voice": "af_bella",
        "speed": 1.2, "response_format": "mp3",
    }

    async def empty_post(path, payload):  # noqa: ARG001
        return b""

    monkeypatch.setattr(be, "_post_bytes", empty_post)
    from src.core.exceptions import TransientError

    with pytest.raises(TransientError):
        asyncio.run(be.synthesize("x", "v"))


def test_openai_http_rate_to_speed():
    from src.engines.backends.openai_http import rate_to_speed

    assert rate_to_speed("+40%") == pytest.approx(1.4)
    assert rate_to_speed("-50%") == pytest.approx(0.5)
    assert rate_to_speed("+0%") == 1.0
    assert rate_to_speed("weird") == 1.0
    assert rate_to_speed("-200%") == 0.25  # 钳制
    assert rate_to_speed("+900%") == 4.0


def test_openai_http_list_voices_and_probe_stubs(monkeypatch):
    from src.engines.backends.openai_http import OpenAIHTTPBackend

    be = OpenAIHTTPBackend()

    async def voices(path):
        if path == "/voices":
            return ["af_bella", {"id": "am_adam"}]
        raise RuntimeError("nope")

    monkeypatch.setattr(be, "_get_json", voices)
    out = asyncio.run(be.list_voices())
    assert [v["short_name"] for v in out] == ["af_bella", "am_adam"]

    async def models(path):  # noqa: ARG001
        return {"data": [{"id": "tts-1"}, {"id": "tts-1-hd"}]}

    monkeypatch.setattr(be, "_get_json", models)
    ok, msg = asyncio.run(be.probe())
    assert ok and "2 models" in msg

    async def down(path):  # noqa: ARG001
        raise OSError("conn refused")

    monkeypatch.setattr(be, "_get_json", down)
    ok, msg = asyncio.run(be.probe())
    assert ok is False and "conn refused" in msg


def test_openai_http_headers_and_env_defaults(monkeypatch):
    from src.engines.backends import openai_http as mod

    monkeypatch.setenv("PPC_TTS_OPENAI_HTTP_BASE_URL", "https://tts.local/api")
    monkeypatch.setenv("PPC_TTS_OPENAI_HTTP_MODEL", "kokoro")
    be = mod.OpenAIHTTPBackend()
    assert be.base_url == "https://tts.local/api"
    assert be.model == "kokoro"
    assert "ppc.tts.v1" == be.protocol_version()
    h = be._headers()
    assert h["Content-Type"] == "application/json"
    be2 = mod.OpenAIHTTPBackend({"api_key": "sk-1"})
    assert be2._headers()["Authorization"] == "Bearer sk-1"
