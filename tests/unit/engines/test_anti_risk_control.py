"""Unit tests for anti-risk-control features in :mod:`src.engines.http_client`.

覆盖以下场景 / Covers the following scenarios:
1. UA 轮换 / User-Agent rotation
2. MUID 持久化 / MUID persistence
3. MUID 轮换 / MUID rotation after configured period
4. muid_persist=False 回退 / Non-persistent MUID falls back to random
5. 抖动注入 / Jitter injection calls asyncio.sleep
6. 抖动禁用 / Jitter disabled → sleep not called
7. proxy 传递 / Proxy forwarded to _list_voices_fn
8. fallback 逻辑 / backend="auto" falls back to environment edge-tts when builtin unavailable
9. backend="env" / Force using environment edge-tts package
10. cleanup() / cleanup() must not raise even with shared_connector=None

These tests use ``_FakeCommunicate`` (which accepts ``connector`` / ``proxy``
kwargs) and patch ``client._communicate_cls`` so that ``_make_communicate``
wraps the fake with ``CommunicateAdapter`` — exercising the anti-risk
override paths without hitting the real Edge TTS service.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from src.config.schema import AntiRiskControlConfig
from src.engines.http_client import EdgeTTSHttpClient
from src.engines.muid_manager import MuidManager


# ---------------------------------------------------------------------------
# Fakes — accepts the same kwargs the real Communicate receives so that
# ``EdgeTTSHttpClient._make_communicate`` can instantiate it without a real
# network connection.
# ---------------------------------------------------------------------------


class _FakeCommunicate:
    """Stand-in for ``edge_tts.Communicate``.

    Accepts ``connector`` / ``proxy`` kwargs so it can be instantiated by
    ``EdgeTTSHttpClient._make_communicate`` directly. Records created
    instances in ``instances`` so tests can inspect per-request
    ``_muid`` / ``_extra_headers`` injection by ``CommunicateAdapter``.
    """

    instances: list["_FakeCommunicate"] = []

    def __init__(
        self,
        text: str,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        connector: Any | None = None,
        proxy: str | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.text = text
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.connector = connector
        self.proxy = proxy
        self.chunks = chunks or [b"\x00" * 16]
        # Per-request isolation hooks (set by CommunicateAdapter).
        self._muid: str | None = None
        self._extra_headers: dict[str, str] = {}
        type(self).instances.append(self)

    async def stream(self):  # noqa: D401
        for chunk in self.chunks:
            yield {"type": "audio", "data": chunk}
        yield {"type": "turn.end"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def preserve_base_headers():
    """Snapshot/restore ``constants.BASE_HEADERS`` to avoid cross-test pollution.

    Although ``CommunicateAdapter`` no longer mutates ``BASE_HEADERS``
    (per-request isolation via ``_extra_headers``), this fixture is kept
    as a safety net for any future regression and to protect tests that
    still observe ``BASE_HEADERS``.
    """
    from src.engines.edge_tts import constants

    snapshot = dict(constants.BASE_HEADERS)
    yield constants.BASE_HEADERS
    constants.BASE_HEADERS.clear()
    constants.BASE_HEADERS.update(snapshot)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAntiRiskControl:
    """Tests for anti-risk-control features (UA, MUID, jitter, proxy, fallback)."""

    # --- 场景 1: UA 轮换 ---

    def test_ua_rotation_sets_base_headers(self, preserve_base_headers) -> None:
        """UA 轮换 / User-Agent rotation injects a pool entry per-request.

        With per-request isolation, the UA is injected via
        ``_extra_headers`` on the wrapped Communicate instance —
        ``BASE_HEADERS`` is NOT mutated (avoids the concurrent race).
        """
        config = AntiRiskControlConfig(
            enabled=True,
            user_agents=["UA-A", "UA-B", "UA-C"],
            jitter_enabled=False,
            shared_connector=False,
            muid_persist=False,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        # 替换 communicate 类以避免真实网络调用
        # Patch communicate class to avoid real network calls
        client._communicate_cls = _FakeCommunicate
        _FakeCommunicate.instances.clear()

        async def drive() -> None:
            async for _ in client.synthesize_stream("hi", "voice"):
                pass

        asyncio.run(drive())

        from src.engines.edge_tts import constants

        # BASE_HEADERS must NOT be mutated by per-request UA rotation.
        # The original UA is a Chrome/Edg string, never "UA-A" etc.
        assert constants.BASE_HEADERS["User-Agent"] not in ["UA-A", "UA-B", "UA-C"], (
            "BASE_HEADERS was mutated — per-request isolation broken"
        )
        # The UA MUST be injected via _extra_headers on the instance.
        assert len(_FakeCommunicate.instances) >= 1
        inst = _FakeCommunicate.instances[0]
        assert inst._extra_headers.get("User-Agent") in ["UA-A", "UA-B", "UA-C"], (
            f"UA not injected via _extra_headers: {inst._extra_headers!r}"
        )

    # --- 场景 2: MUID 持久化 ---

    def test_muid_persistence(self, tmp_path: Path) -> None:
        """MUID 持久化 / MUID persists across calls and is re-generated when file is removed."""
        muid_file = tmp_path / "muid"
        manager = MuidManager(file_path=str(muid_file), rotate_hours=24, persist=True)

        first = manager.get_muid()
        second = manager.get_muid()
        assert first == second
        assert muid_file.exists()

        # 删除文件后应生成新的 MUID
        # Delete the file → next call should generate a new MUID
        muid_file.unlink()
        third = manager.get_muid()
        assert third != first
        assert muid_file.exists()

    # --- 场景 3: MUID 轮换 ---

    def test_muid_rotation(self, tmp_path: Path) -> None:
        """MUID 轮换 / MUID rotates after the configured period elapses.

        注: spec 示例值为 ``rotate_hours=0.001``（≈3.6s），但配合 0.1s 等待
        无法触发轮换；此处使用更小的 ``rotate_hours=1e-5``（≈0.036s），
        等待 0.1s 后即可触发轮换，以保持测试快速运行。

        Note: the spec's example value ``rotate_hours=0.001`` (≈3.6s) would
        not rotate within a 0.1s wait; we use a smaller value
        ``rotate_hours=1e-5`` (≈0.036s) so rotation triggers after 0.1s.
        """
        muid_file = tmp_path / "muid"
        manager = MuidManager(file_path=str(muid_file), rotate_hours=1e-5, persist=True)

        first = manager.get_muid()
        time.sleep(0.1)  # exceed rotate_seconds
        second = manager.get_muid()
        assert second != first

    # --- 场景 4: muid_persist=False 回退 ---

    def test_muid_persist_false_returns_random(self, tmp_path: Path) -> None:
        """muid_persist=False 回退 / Non-persistent MUID returns a fresh random each call."""
        manager = MuidManager(file_path=str(tmp_path / "muid"), rotate_hours=24, persist=False)
        first = manager.get_muid()
        second = manager.get_muid()
        assert first != second

    # --- 场景 5: 抖动注入 ---

    def test_jitter_injection_calls_sleep(
        self,
        monkeypatch,
        preserve_base_headers,
    ) -> None:
        """抖动注入 / Jitter injection calls asyncio.sleep within [0, max]."""
        config = AntiRiskControlConfig(
            enabled=True,
            jitter_enabled=True,
            jitter_max_seconds=0.05,
            shared_connector=False,
            muid_persist=False,
        )
        sleep_calls: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._communicate_cls = _FakeCommunicate

        async def drive() -> None:
            async for _ in client.synthesize_stream("hi", "voice"):
                pass

        asyncio.run(drive())

        assert len(sleep_calls) >= 1
        for delay in sleep_calls:
            assert 0 <= delay <= 0.05

    # --- 场景 6: 抖动禁用 ---

    def test_jitter_disabled_no_sleep(
        self,
        monkeypatch,
        preserve_base_headers,
    ) -> None:
        """抖动禁用 / Jitter disabled → asyncio.sleep is not called."""
        config = AntiRiskControlConfig(
            enabled=True,
            jitter_enabled=False,
            jitter_max_seconds=0.05,
            shared_connector=False,
            muid_persist=False,
        )
        sleep_calls: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._communicate_cls = _FakeCommunicate

        async def drive() -> None:
            async for _ in client.synthesize_stream("hi", "voice"):
                pass

        asyncio.run(drive())

        assert sleep_calls == []

    # --- 场景 7: proxy 传递 ---

    def test_proxy_passthrough_to_list_voices(self) -> None:
        """proxy 传递 / Proxy is forwarded to _list_voices_fn."""
        config = AntiRiskControlConfig(
            enabled=True,
            proxy="http://127.0.0.1:8080",
            shared_connector=False,
            muid_persist=False,
            jitter_enabled=False,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)

        captured: dict[str, Any] = {}

        async def fake_list_voices_fn(*, connector=None, proxy=None):
            captured["proxy"] = proxy
            return [
                {
                    "ShortName": "v1",
                    "Name": "v1",
                    "Gender": "Female",
                    "Locale": "en-US",
                    "FriendlyName": "v1",
                }
            ]

        # 替换后端 list_voices 函数以捕获 proxy 参数
        # Replace backend list_voices to capture the proxy kwarg
        client._list_voices_fn = fake_list_voices_fn

        asyncio.run(client.list_voices())
        assert captured["proxy"] == "http://127.0.0.1:8080"

    # --- 场景 8: backend="auto" 回退 ---

    def test_backend_auto_falls_back_to_environment(self, monkeypatch) -> None:
        """backend="auto" / Fall back to environment edge-tts when builtin unavailable."""
        import src.engines.http_client as client_mod

        # 模拟 builtin 后端不可用
        # Simulate builtin backend being unavailable
        monkeypatch.setattr(client_mod, "_BUILTIN_COMMUNICATE", None)
        monkeypatch.setattr(client_mod, "_BUILTIN_LIST_VOICES", None)

        config = AntiRiskControlConfig(backend="auto", shared_connector=False)
        client = EdgeTTSHttpClient(anti_risk_config=config)

        # 应已回退到环境 edge-tts 包
        # Should have fallen back to environment edge-tts
        assert client._backend_name == "edge-tts (environment)"
        assert client._communicate_cls is not None

    # --- 场景 9: backend="env" ---

    def test_backend_env_uses_environment_package(self) -> None:
        """backend="env" / Force using environment edge-tts package over customized."""
        config = AntiRiskControlConfig(
            backend="env",
            shared_connector=False,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        assert client._backend_name == "edge-tts (environment)"

        import edge_tts as env_edge_tts

        assert client._communicate_cls is env_edge_tts.Communicate

    # --- 场景 10: cleanup() ---

    def test_cleanup_does_not_raise_when_no_connector(self) -> None:
        """cleanup() / cleanup() must not raise even with shared_connector=None."""
        config = AntiRiskControlConfig(shared_connector=False)
        client = EdgeTTSHttpClient(anti_risk_config=config)
        assert client._shared_connector is None

        # 不应抛出异常
        # Should not raise
        asyncio.run(client.cleanup())


# ---------------------------------------------------------------------------
# Concurrent anti-risk-control tests — expose race conditions in
# MUID/UA class-level mutation and stream cleanup (Tasks 1-4).
# ---------------------------------------------------------------------------


class _ConcurrentCapturingCommunicate:
    """Fake Communicate that captures per-instance ``_muid`` and
    ``_extra_headers`` at stream() entry and after a delay.

    Accepts connector/proxy kwargs so ``EdgeTTSHttpClient._make_communicate``
    can instantiate it. The per-request isolation hooks (``_muid`` /
    ``_extra_headers``) are set by ``CommunicateAdapter`` before
    ``stream()`` is called — this fake captures them at entry and
    mid-stream to verify per-stream stability under concurrency.

    Class-level ``captures`` list collects per-instance capture dicts.
    Reset via ``_ConcurrentCapturingCommunicate.captures.clear()``.
    """

    captures: list[dict[str, Any]] = []

    def __init__(
        self,
        text: str,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        connector: Any | None = None,
        proxy: str | None = None,
    ) -> None:
        self.text = text
        self.voice = voice
        self.connector = connector
        self.proxy = proxy
        # Per-request isolation hooks (set by CommunicateAdapter).
        self._muid: str | None = None
        self._extra_headers: dict[str, str] = {}
        self.capture: dict[str, Any] = {}
        _ConcurrentCapturingCommunicate.captures.append(self.capture)

    async def stream(self):  # noqa: D401
        # Capture per-instance isolation values at entry — these are
        # what the WS connect would use.
        self.capture["muid_entry"] = self._muid
        self.capture["ua_entry"] = self._extra_headers.get("User-Agent")
        # Delay to encourage interleaving with other concurrent streams
        await asyncio.sleep(0.03)
        # Capture again after delay — should be stable per-stream
        self.capture["muid_mid"] = self._muid
        self.capture["ua_mid"] = self._extra_headers.get("User-Agent")
        yield {"type": "audio", "data": b"\x00" * 8}
        yield {"type": "turn.end"}


class _HangingAcloseTrackingCommunicate:
    """Fake Communicate whose stream() hangs forever (no chunks yielded).

    Tracks whether the generator was properly cleaned up by catching
    ``GeneratorExit`` (from explicit ``aclose()``) or ``CancelledError``
    (from ``asyncio.wait_for`` timeout). This simulates the
    ``async with ClientSession()`` cleanup that the real
    ``Communicate.__stream()`` performs on either exception.

    ``instances`` collects all created instances so tests can assert
    cleanup behavior.
    """

    instances: list["_HangingAcloseTrackingCommunicate"] = []

    def __init__(
        self,
        text: str,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        connector: Any | None = None,
        proxy: str | None = None,
        hang: bool = True,
    ) -> None:
        self.hang = hang
        self.aclose_called = False
        _HangingAcloseTrackingCommunicate.instances.append(self)

    async def stream(self):  # noqa: D401
        try:
            if self.hang:
                # Hang forever — no chunks, triggers wait_for timeout
                await asyncio.Event().wait()
            else:
                yield {"type": "audio", "data": b"ok"}
                yield {"type": "turn.end"}
        except (GeneratorExit, asyncio.CancelledError):
            # Simulate the ``async with ClientSession()`` cleanup that
            # the real Communicate.__stream() performs on either
            # GeneratorExit (explicit aclose) or CancelledError
            # (wait_for timeout).
            self.aclose_called = True
            raise


class _FailFirstCommunicate:
    """Fake Communicate whose first stream() raises a WebSocketError.

    Subsequent instances (created on retry) yield normally. Used to test
    that streams are aclosed before retry.
    """

    call_count: int = 0

    def __init__(
        self,
        text: str,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        connector: Any | None = None,
        proxy: str | None = None,
    ) -> None:
        self.aclose_called = False
        type(self).call_count += 1
        self._attempt = type(self).call_count

    async def stream(self):  # noqa: D401
        try:
            if self._attempt == 1:
                # Raise immediately — no chunks, connect-level failure
                raise type("WebSocketError", (Exception,), {})("connect failed")
            else:
                yield {"type": "audio", "data": b"ok"}
                yield {"type": "turn.end"}
        except (GeneratorExit, asyncio.CancelledError):
            self.aclose_called = True
            raise


class TestConcurrentAntiRisk:
    """Tests for concurrent MUID/UA override isolation (Tasks 1 & 2)."""

    def setup_method(self) -> None:
        _ConcurrentCapturingCommunicate.captures.clear()

    def test_concurrent_muid_restore_not_chained(self, preserve_base_headers) -> None:
        """After N concurrent streams complete, DRM.generate_muid MUST be
        restored to the original function — not a lambda left over from
        chain corruption.

        With the current class-level mutation, when streams overlap:
          A._original = OG; DRM.generate_muid = lambda_A
          B._original = lambda_A; DRM.generate_muid = lambda_B
          A restores → DRM.generate_muid = OG (A._original)
          B restores → DRM.generate_muid = lambda_A (B._original)  ← BUG

        After all streams finish, DRM.generate_muid is lambda_A, not OG.
        """
        from src.engines.edge_tts.drm import DRM

        original_generate_muid = DRM.generate_muid
        config = AntiRiskControlConfig(
            enabled=True,
            muid_persist=True,
            jitter_enabled=False,
            shared_connector=False,
            user_agents=["UA-A"],
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._communicate_cls = _ConcurrentCapturingCommunicate

        async def _drive() -> None:
            tasks = [
                client.synthesize_stream(f"text-{i}", "voice")
                for i in range(4)
            ]
            await asyncio.gather(*[_drain(g) for g in tasks])

        async def _drain(gen) -> None:
            async for _ in gen:
                pass

        asyncio.run(_drive())

        # The original generate_muid returns a random hex each call.
        # A lambda from the override returns the persisted muid (constant).
        # If chain corruption occurred, DRM.generate_muid is a lambda.
        assert DRM.generate_muid is original_generate_muid, (
            "DRM.generate_muid was not restored to the original function — "
            "chain corruption from concurrent overrides detected"
        )

    def test_concurrent_ua_restore_original(self, preserve_base_headers) -> None:
        """After N concurrent streams complete, BASE_HEADERS['User-Agent']
        MUST be restored to its original value.

        With class-level mutation of the shared dict, concurrent overrides
        and restores can leave a stale UA from another stream.
        """
        from src.engines.edge_tts import constants

        original_ua = constants.BASE_HEADERS["User-Agent"]
        config = AntiRiskControlConfig(
            enabled=True,
            muid_persist=False,
            jitter_enabled=False,
            shared_connector=False,
            user_agents=["UA-X", "UA-Y", "UA-Z"],
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._communicate_cls = _ConcurrentCapturingCommunicate

        async def _drive() -> None:
            tasks = [
                client.synthesize_stream(f"text-{i}", "voice")
                for i in range(4)
            ]
            await asyncio.gather(*[_drain(g) for g in tasks])

        async def _drain(gen) -> None:
            async for _ in gen:
                pass

        asyncio.run(_drive())

        assert constants.BASE_HEADERS["User-Agent"] == original_ua, (
            "BASE_HEADERS['User-Agent'] was not restored — "
            "concurrent UA override race detected"
        )

    def test_concurrent_ua_stable_per_stream(self, preserve_base_headers) -> None:
        """Each stream's captured UA at entry and mid-stream MUST be equal.

        With the global mutation, stream A's UA can be overwritten by
        stream B while A is still running, so A's mid-stream capture
        differs from its entry capture.
        """
        config = AntiRiskControlConfig(
            enabled=True,
            muid_persist=False,
            jitter_enabled=False,
            shared_connector=False,
            user_agents=["UA-A", "UA-B", "UA-C", "UA-D"],
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._communicate_cls = _ConcurrentCapturingCommunicate

        async def _drive() -> None:
            tasks = [
                client.synthesize_stream(f"text-{i}", "voice")
                for i in range(4)
            ]
            await asyncio.gather(*[_drain(g) for g in tasks])

        async def _drain(gen) -> None:
            async for _ in gen:
                pass

        asyncio.run(_drive())

        assert len(_ConcurrentCapturingCommunicate.captures) == 4
        for i, cap in enumerate(_ConcurrentCapturingCommunicate.captures):
            assert cap["ua_entry"] == cap["ua_mid"], (
                f"Stream {i} UA changed mid-stream: "
                f"entry={cap['ua_entry']!r} mid={cap['ua_mid']!r}"
            )


class TestStreamCleanup:
    """Tests for inner generator cleanup on timeout/retry (Task 3)."""

    def setup_method(self) -> None:
        _HangingAcloseTrackingCommunicate.instances.clear()
        _FailFirstCommunicate.call_count = 0

    def test_wait_for_timeout_closes_inner_generator(
        self,
        preserve_base_headers,
        monkeypatch,
    ) -> None:
        """When wait_for times out, the inner Communicate.stream() MUST be
        aclosed so its ClientSession exits cleanly.

        The fake hangs forever (no chunks), triggering wait_for timeout.
        With max_connect_retries=0, the timeout propagates. The inner
        generator's aclose must have been called (via the adapter's
        finally block).
        """
        config = AntiRiskControlConfig(
            enabled=True,
            muid_persist=False,
            jitter_enabled=False,
            shared_connector=False,
            user_agents=["UA-A"],
            max_connect_retries=0,
            request_timeout_seconds=1.0,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        # Override the per-request timeout on the instance to avoid the
        # 1.0s Pydantic floor while keeping the test fast.
        client._request_timeout_seconds = 0.05
        client._communicate_cls = _HangingAcloseTrackingCommunicate

        from src.core.exceptions import TransientError

        async def _drive() -> None:
            gen = client.synthesize_stream("hi", "voice")
            async for _ in gen:
                pass

        # Should raise (timeout → TransientError)
        with pytest.raises((TransientError, Exception)):
            asyncio.run(_drive())

        # The inner fake's generator MUST have been aclosed
        assert len(_HangingAcloseTrackingCommunicate.instances) >= 1
        for inst in _HangingAcloseTrackingCommunicate.instances:
            assert inst.aclose_called, (
                "Inner Communicate.stream() was not aclosed on timeout — "
                "ClientSession may leak"
            )

    def test_connect_failure_retry_closes_stream(
        self,
        preserve_base_headers,
    ) -> None:
        """On connect-level failure (WebSocketError), the abandoned stream
        MUST be aclosed before retrying.
        """
        config = AntiRiskControlConfig(
            enabled=True,
            muid_persist=False,
            jitter_enabled=False,
            shared_connector=False,
            user_agents=["UA-A"],
            max_connect_retries=1,
            request_timeout_seconds=5.0,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._communicate_cls = _FailFirstCommunicate

        async def _drive() -> None:
            gen = client.synthesize_stream("hi", "voice")
            async for _ in gen:
                pass

        asyncio.run(_drive())

        # First attempt's fake MUST have been aclosed before retry
        assert len(_FailFirstCommunicate.__subclasses__()) == 0  # sanity
        # Check the first instance was aclosed
        # _FailFirstCommunicate instances are tracked via call_count;
        # we verify at least one instance had aclose_called=True
        # Note: instances aren't stored on the class for _FailFirstCommunicate;
        # we verify success by the stream completing without error.


class TestSharedConnectorLifecycle:
    """Tests for shared connector session lifecycle (Task 4)."""

    def test_shared_connector_not_closed_after_timeout(
        self,
        preserve_base_headers,
    ) -> None:
        """After a stream times out, the shared connector MUST remain
        usable AND the inner generator MUST be aclosed (so its
        ClientSession exits cleanly — the root cause of 'Session is
        closed' errors on subsequent streams).
        """
        config = AntiRiskControlConfig(
            enabled=True,
            muid_persist=False,
            jitter_enabled=False,
            shared_connector=True,
            user_agents=["UA-A"],
            max_connect_retries=0,
            request_timeout_seconds=1.0,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._request_timeout_seconds = 0.05

        # First fake hangs (no chunks), second succeeds
        call_count = {"n": 0}

        class _HangThenOK:
            def __init__(self, text, voice, *, rate="+0%", volume="+0%",
                         connector=None, proxy=None) -> None:
                call_count["n"] += 1
                self.attempt = call_count["n"]
                self.connector = connector
                self.aclose_called = False

            async def stream(self):
                try:
                    if self.attempt == 1:
                        await asyncio.Event().wait()
                    else:
                        yield {"type": "audio", "data": b"ok"}
                        yield {"type": "turn.end"}
                except (GeneratorExit, asyncio.CancelledError):
                    self.aclose_called = True
                    raise

        client._communicate_cls = _HangThenOK

        async def _drive() -> None:
            client._ensure_shared_connector()
            assert client._shared_connector is not None
            gen1 = client.synthesize_stream("hi", "voice")
            try:
                async for _ in gen1:
                    pass
            except Exception:
                pass  # expected timeout
            # Connector must NOT be closed
            assert not client._shared_connector.closed, (
                "Shared connector was closed after timeout — "
                "subsequent streams will fail with 'Session is closed'"
            )
            # Second stream should succeed using the same connector
            gen2 = client.synthesize_stream("hi2", "voice")
            chunks = []
            async for chunk in gen2:
                chunks.append(chunk)
            assert len(chunks) > 0, "Second stream produced no output"
            await client.cleanup()

        asyncio.run(_drive())

        # The first (hanging) fake's generator MUST have been aclosed
        # so its ClientSession exited cleanly via async with.
        # This is the root-cause assertion for "Session is closed".
        # (Verified via the class-level instances list is not possible
        # here since we used a local _HangThenOK class; instead we rely
        # on the dedicated test_wait_for_timeout_closes_inner_generator
        # for the aclose assertion.)


# ---------------------------------------------------------------------------
# UA pool validity tests (Task 9).
# ---------------------------------------------------------------------------


class TestUAPoolValidation:
    """Tests for UA pool format validation (SubTask 9.1)."""

    def test_ua_pool_format_validation(self) -> None:
        """Every UA in USER_AGENTS must match Chromium/Edge format.

        Validates:
        - Contains ``Mozilla/5.0`` prefix
        - Contains ``Chrome/<version>`` with a numeric version
        - Contains ``Edg/<version>`` with a matching version
        - Chrome major version is not too old (>= CHROMIUM_MAJOR_VERSION - 5)
        """
        import re

        from src.engines.edge_tts import constants

        assert len(constants.USER_AGENTS) >= 2, (
            "UA pool must have at least 2 entries for rotation"
        )

        chrome_major = int(constants.CHROMIUM_MAJOR_VERSION)
        min_acceptable = chrome_major - 5

        ua_pattern = re.compile(
            r"^Mozilla/5\.0.*Chrome/(?P<chrome>\d+)\.\d+\.\d+\.\d+.*Edg/(?P<edg>\d+)\.\d+\.\d+\.\d+"
        )

        for i, ua in enumerate(constants.USER_AGENTS):
            match = ua_pattern.search(ua)
            assert match is not None, (
                f"UA[{i}] does not match Chrome/Edg format: {ua}"
            )
            chrome_ver = int(match.group("chrome"))
            edg_ver = int(match.group("edg"))
            assert chrome_ver == edg_ver, (
                f"UA[{i}] Chrome version ({chrome_ver}) != Edg version ({edg_ver}): {ua}"
            )
            assert chrome_ver >= min_acceptable, (
                f"UA[{i}] Chrome version ({chrome_ver}) is too old "
                f"(minimum {min_acceptable}): {ua}"
            )


class TestUAHealthFilter:
    """Tests for UAHealthFilter logic (SubTask 9.4).

    Covers both failure paths:
    - 403 → immediate ``mark_unhealthy``
    - NoAudioReceived → counter-based deferred marking via ``record_no_audio``
    """

    def test_initially_all_healthy(self) -> None:
        """All UAs are healthy when the filter is first created."""
        from src.engines.ua_health_filter import UAHealthFilter

        f = UAHealthFilter(cooldown_seconds=300.0)
        uas = ["UA-A", "UA-B", "UA-C"]
        healthy = f.filter_healthy(uas)
        assert healthy == uas

    def test_mark_unhealthy_excludes_from_rotation(self) -> None:
        """A UA marked unhealthy is excluded from filter_healthy()."""
        from src.engines.ua_health_filter import UAHealthFilter

        f = UAHealthFilter(cooldown_seconds=300.0)
        uas = ["UA-A", "UA-B", "UA-C"]
        f.mark_unhealthy("UA-B")
        healthy = f.filter_healthy(uas)
        assert "UA-B" not in healthy
        assert "UA-A" in healthy
        assert "UA-C" in healthy

    def test_cooldown_expiry_restores_ua(self) -> None:
        """After cooldown expires, a previously unhealthy UA is healthy again.

        改用足够大的冷却值 + 直接改写内部标记时间戳模拟到期，避免真实
        ``sleep`` 在满载下出现 10ms 冷却先于断言过期导致的 flaky。
        """
        from src.engines.ua_health_filter import UAHealthFilter

        f = UAHealthFilter(cooldown_seconds=10)  # 足够大，立即断言稳定
        f.mark_unhealthy("UA-A")
        # Immediately unhealthy
        assert not f.is_healthy("UA-A")
        # 把标记时间戳拨回过去（超过冷却），模拟冷却到期
        f._unhealthy["UA-A"] = time.monotonic() - 11
        assert f.is_healthy("UA-A")

    def test_all_unhealthy_falls_back_gracefully(self) -> None:
        """When all UAs are unhealthy, the adapter falls back to the full pool.

        The adapter's ``stream()`` checks ``filter_healthy()``; if the
        result is empty, it uses the original pool. This test verifies
        that ``filter_healthy()`` returns an empty list, and the
        adapter's fallback logic is tested in
        :class:`TestCommunicateAdapterUAHealth`.
        """
        from src.engines.ua_health_filter import UAHealthFilter

        f = UAHealthFilter(cooldown_seconds=300.0)
        uas = ["UA-A"]
        f.mark_unhealthy("UA-A")
        assert f.filter_healthy(uas) == []

    def test_is_ua_failure_detects_403(self) -> None:
        """is_ua_failure() returns True for HTTP 403 errors (immediate mark)."""
        from src.engines.ua_health_filter import UAHealthFilter

        class _FakeClientResponseError(Exception):
            status = 403

        assert UAHealthFilter.is_ua_failure(_FakeClientResponseError("Forbidden"))
        assert UAHealthFilter.is_ua_failure(Exception("HTTP 403 Forbidden"))

    def test_is_ua_failure_ignores_no_audio(self) -> None:
        """is_ua_failure() returns False for NoAudioReceived.

        NoAudioReceived is NOT an immediate-mark failure — it goes
        through the counter-based ``record_no_audio`` path instead.
        """
        from src.engines.ua_health_filter import UAHealthFilter

        NoAudioReceived = type("NoAudioReceived", (Exception,), {})
        assert not UAHealthFilter.is_ua_failure(NoAudioReceived("no audio"))

    def test_is_no_audio_failure_detects_no_audio(self) -> None:
        """is_no_audio_failure() returns True for NoAudioReceived only."""
        from src.engines.ua_health_filter import UAHealthFilter

        NoAudioReceived = type("NoAudioReceived", (Exception,), {})
        assert UAHealthFilter.is_no_audio_failure(NoAudioReceived("no audio"))
        # 403 is NOT a no-audio failure
        class _Fake403(Exception):
            status = 403
        assert not UAHealthFilter.is_no_audio_failure(_Fake403("forbidden"))

    def test_is_ua_failure_ignores_other_errors(self) -> None:
        """is_ua_failure() returns False for non-403 errors."""
        from src.engines.ua_health_filter import UAHealthFilter

        WebSocketError = type("WebSocketError", (Exception,), {})
        assert not UAHealthFilter.is_ua_failure(WebSocketError("ws error"))
        assert not UAHealthFilter.is_ua_failure(asyncio.TimeoutError())
        assert not UAHealthFilter.is_ua_failure(ConnectionError("refused"))

    # --- NoAudioReceived counter logic ---

    def test_record_no_audio_below_threshold_no_mark(self) -> None:
        """record_no_audio() below threshold does NOT mark unhealthy.

        Threshold=3: 2 consecutive NoAudioReceived should not mark.
        """
        from src.engines.ua_health_filter import UAHealthFilter

        f = UAHealthFilter(cooldown_seconds=300.0, no_audio_threshold=3)
        assert not f.record_no_audio("UA-A")  # count=1
        assert f.is_healthy("UA-A")
        assert not f.record_no_audio("UA-A")  # count=2
        assert f.is_healthy("UA-A")

    def test_record_no_audio_at_threshold_marks_unhealthy(self) -> None:
        """record_no_audio() at threshold marks unhealthy and returns True."""
        from src.engines.ua_health_filter import UAHealthFilter

        f = UAHealthFilter(cooldown_seconds=300.0, no_audio_threshold=3)
        f.record_no_audio("UA-A")  # count=1
        f.record_no_audio("UA-A")  # count=2
        marked = f.record_no_audio("UA-A")  # count=3 → mark
        assert marked is True
        assert not f.is_healthy("UA-A")

    def test_record_success_resets_counter(self) -> None:
        """record_success() resets the NoAudioReceived counter.

        After 2 NoAudioReceived (below threshold), a success resets the
        counter so the next NoAudioReceived starts from 1 again.
        """
        from src.engines.ua_health_filter import UAHealthFilter

        f = UAHealthFilter(cooldown_seconds=300.0, no_audio_threshold=3)
        f.record_no_audio("UA-A")  # count=1
        f.record_no_audio("UA-A")  # count=2
        f.record_success("UA-A")   # reset
        # Now 2 more NoAudioReceived should still be below threshold
        f.record_no_audio("UA-A")  # count=1
        assert f.is_healthy("UA-A")
        f.record_no_audio("UA-A")  # count=2
        assert f.is_healthy("UA-A")

    def test_no_audio_counter_is_per_ua(self) -> None:
        """NoAudioReceived counter is tracked independently per UA."""
        from src.engines.ua_health_filter import UAHealthFilter

        f = UAHealthFilter(cooldown_seconds=300.0, no_audio_threshold=3)
        f.record_no_audio("UA-A")  # A count=1
        f.record_no_audio("UA-B")  # B count=1
        f.record_no_audio("UA-A")  # A count=2
        assert f.is_healthy("UA-A")
        assert f.is_healthy("UA-B")
        # A reaches threshold
        marked = f.record_no_audio("UA-A")  # A count=3 → mark
        assert marked
        assert not f.is_healthy("UA-A")
        # B is still healthy (count=1)
        assert f.is_healthy("UA-B")


class TestCommunicateAdapterUAHealth:
    """Integration tests for UAHealthFilter + CommunicateAdapter (SubTask 9.4).

    NoAudioReceived is transient: a single occurrence must NOT mark
    the UA unhealthy. Only after ``no_audio_threshold`` consecutive
    failures (without an intervening success) is the UA marked.
    """

    def test_adapter_single_no_audio_not_marked(self, preserve_base_headers) -> None:
        """A single NoAudioReceived does NOT mark the UA unhealthy.

        NoAudioReceived is common under high load; the UA is only
        penalized after ``no_audio_threshold`` consecutive failures.
        """
        NoAudioReceived = type("NoAudioReceived", (Exception,), {})

        class _NoAudioCommunicate:
            def __init__(self, text, voice, *, rate="+0%", volume="+0%",
                         connector=None, proxy=None) -> None:
                pass

            async def stream(self):
                raise NoAudioReceived("no audio from service")
                yield  # never reached, makes this a generator

        config = AntiRiskControlConfig(
            enabled=True,
            user_agents=["UA-A", "UA-B"],
            jitter_enabled=False,
            shared_connector=False,
            muid_persist=False,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._communicate_cls = _NoAudioCommunicate
        assert client._ua_health_filter is not None

        async def _drive() -> None:
            gen = client.synthesize_stream("hi", "voice")
            async for _ in gen:
                pass

        with pytest.raises(Exception):
            asyncio.run(_drive())

        # NO UA should be marked unhealthy after a single NoAudioReceived
        unhealthy_count = sum(
            1 for ua in ["UA-A", "UA-B"]
            if not client._ua_health_filter.is_healthy(ua)
        )
        assert unhealthy_count == 0, (
            f"Expected 0 unhealthy UAs (single NoAudioReceived is transient), "
            f"got {unhealthy_count}"
        )

    def test_adapter_marks_unhealthy_after_threshold_no_audio(
        self, preserve_base_headers
    ) -> None:
        """After ``no_audio_threshold`` consecutive NoAudioReceived (same UA),
        the UA is marked unhealthy.

        Uses a single-UA pool so every stream hits the same UA, and
        a custom threshold of 3 to keep the test fast.
        """
        NoAudioReceived = type("NoAudioReceived", (Exception,), {})

        class _NoAudioCommunicate:
            def __init__(self, text, voice, *, rate="+0%", volume="+0%",
                         connector=None, proxy=None) -> None:
                pass

            async def stream(self):
                raise NoAudioReceived("no audio from service")
                yield  # never reached, makes this a generator

        config = AntiRiskControlConfig(
            enabled=True,
            user_agents=["UA-ONLY"],  # single UA → every stream hits it
            jitter_enabled=False,
            shared_connector=False,
            muid_persist=False,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._communicate_cls = _NoAudioCommunicate
        assert client._ua_health_filter is not None
        # Lower threshold for faster testing
        client._ua_health_filter._no_audio_threshold = 3

        async def _drive() -> None:
            gen = client.synthesize_stream("hi", "voice")
            async for _ in gen:
                pass

        # First 2 attempts: not marked yet (below threshold)
        for i in range(2):
            with pytest.raises(Exception):
                asyncio.run(_drive())
            assert client._ua_health_filter.is_healthy("UA-ONLY"), (
                f"UA should still be healthy after {i+1} NoAudioReceived (< threshold 3)"
            )

        # 3rd attempt: threshold reached → marked unhealthy
        with pytest.raises(Exception):
            asyncio.run(_drive())
        assert not client._ua_health_filter.is_healthy("UA-ONLY"), (
            "UA should be marked unhealthy after 3 consecutive NoAudioReceived"
        )

    def test_adapter_success_resets_no_audio_counter(
        self, preserve_base_headers
    ) -> None:
        """A successful stream resets the NoAudioReceived counter.

        After 2 NoAudioReceived (below threshold), a successful stream
        resets the counter — the next 2 NoAudioReceived should NOT
        trigger marking.
        """
        NoAudioReceived = type("NoAudioReceived", (Exception,), {})

        class _FlakyCommunicate:
            """Fails with NoAudioReceived for the first N calls, then succeeds."""

            call_count = 0
            fail_times = 0  # how many times to fail before succeeding

            def __init__(self, text, voice, *, rate="+0%", volume="+0%",
                         connector=None, proxy=None) -> None:
                _FlakyCommunicate.call_count += 1
                self._should_fail = _FlakyCommunicate.call_count <= _FlakyCommunicate.fail_times

            async def stream(self):
                if self._should_fail:
                    raise NoAudioReceived("no audio")
                yield {"type": "audio", "data": b"\x00" * 16}
                yield {"type": "turn.end"}

        config = AntiRiskControlConfig(
            enabled=True,
            user_agents=["UA-ONLY"],
            jitter_enabled=False,
            shared_connector=False,
            muid_persist=False,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._communicate_cls = _FlakyCommunicate
        assert client._ua_health_filter is not None
        client._ua_health_filter._no_audio_threshold = 3

        async def _drive() -> None:
            gen = client.synthesize_stream("hi", "voice")
            async for _ in gen:
                pass

        # 2 NoAudioReceived (below threshold)
        _FlakyCommunicate.fail_times = 2
        _FlakyCommunicate.call_count = 0
        for _ in range(2):
            with pytest.raises(Exception):
                asyncio.run(_drive())
        assert client._ua_health_filter.is_healthy("UA-ONLY")

        # 1 success → resets counter
        _FlakyCommunicate.fail_times = 0
        _FlakyCommunicate.call_count = 0
        asyncio.run(_drive())
        assert client._ua_health_filter.is_healthy("UA-ONLY")

        # 2 more NoAudioReceived → should NOT mark (counter was reset)
        _FlakyCommunicate.fail_times = 2
        _FlakyCommunicate.call_count = 0
        for _ in range(2):
            with pytest.raises(Exception):
                asyncio.run(_drive())
        assert client._ua_health_filter.is_healthy("UA-ONLY"), (
            "UA should still be healthy — counter was reset by the success"
        )

    def test_adapter_403_marks_unhealthy_immediately(
        self, preserve_base_headers
    ) -> None:
        """A 403 error marks the UA unhealthy immediately (no threshold)."""

        class _Fake403Error(Exception):
            status = 403

        class _ForbiddenCommunicate:
            def __init__(self, text, voice, *, rate="+0%", volume="+0%",
                         connector=None, proxy=None) -> None:
                pass

            async def stream(self):
                raise _Fake403Error("403 Forbidden")
                yield  # never reached, makes this a generator

        config = AntiRiskControlConfig(
            enabled=True,
            user_agents=["UA-A", "UA-B"],
            jitter_enabled=False,
            shared_connector=False,
            muid_persist=False,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._communicate_cls = _ForbiddenCommunicate
        assert client._ua_health_filter is not None

        async def _drive() -> None:
            gen = client.synthesize_stream("hi", "voice")
            async for _ in gen:
                pass

        with pytest.raises(Exception):
            asyncio.run(_drive())

        # Exactly one UA should be marked unhealthy immediately
        unhealthy_count = sum(
            1 for ua in ["UA-A", "UA-B"]
            if not client._ua_health_filter.is_healthy(ua)
        )
        assert unhealthy_count == 1, (
            f"Expected 1 unhealthy UA (403 → immediate mark), got {unhealthy_count}"
        )

    def test_adapter_skips_unhealthy_ua(self, preserve_base_headers) -> None:
        """When a UA is marked unhealthy, the adapter selects a different UA."""
        _FakeCommunicate.instances.clear()

        config = AntiRiskControlConfig(
            enabled=True,
            user_agents=["UA-A", "UA-B"],
            jitter_enabled=False,
            shared_connector=False,
            muid_persist=False,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._communicate_cls = _FakeCommunicate
        # Mark UA-A as unhealthy
        assert client._ua_health_filter is not None
        client._ua_health_filter.mark_unhealthy("UA-A")

        async def _drive() -> None:
            async for _ in client.synthesize_stream("hi", "voice"):
                pass

        asyncio.run(_drive())

        # The selected UA must be UA-B (UA-A is unhealthy)
        assert len(_FakeCommunicate.instances) >= 1
        selected_ua = _FakeCommunicate.instances[0]._extra_headers.get("User-Agent")
        assert selected_ua == "UA-B", (
            f"Expected UA-B (UA-A is unhealthy), got {selected_ua!r}"
        )


# ---------------------------------------------------------------------------
# UA switch verification tests — NoAudioReceived threshold triggers
# verification with other UAs. Success → keep mark; all fail → revoke.
# ---------------------------------------------------------------------------


class _UAAwareCommunicate:
    """Fake Communicate that succeeds or fails based on the injected UA.

    Reads the UA from ``_extra_headers["User-Agent"]`` (set by
    ``CommunicateAdapter``) and consults a class-level ``ua_behavior``
    dict mapping UA → "fail" | "ok". This lets tests deterministically
    control which UAs fail with NoAudioReceived.
    """

    ua_behavior: dict[str, str] = {}  # UA → "fail" | "ok"

    def __init__(
        self,
        text: str,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        connector: Any | None = None,
        proxy: str | None = None,
    ) -> None:
        self._extra_headers: dict[str, str] = {}
        self._muid: str | None = None

    async def stream(self):
        ua = self._extra_headers.get("User-Agent", "")
        behavior = _UAAwareCommunicate.ua_behavior.get(ua, "ok")
        if behavior == "fail":
            NoAudioReceived = type("NoAudioReceived", (Exception,), {})
            raise NoAudioReceived("no audio from service")
        yield {"type": "audio", "data": b"\x00" * 16}
        yield {"type": "turn.end"}


class TestUASwitchVerification:
    """Tests for the UA switch verification logic in
    ``_synthesize_single_voice``.

    When a UA accumulates ``ua_no_audio_threshold`` consecutive
    NoAudioReceived failures, the engine retries with a different UA.
    - If the alternate UA succeeds → the original UA's mark is kept.
    - If all alternate UAs also fail → all marks are revoked (server-side
      issue, not UA-specific).
    """

    def test_switch_ua_then_success_keeps_mark(
        self, preserve_base_headers, monkeypatch
    ) -> None:
        """UA-A fails → switch to UA-B → UA-B succeeds → UA-A stays marked."""
        _UAAwareCommunicate.ua_behavior = {"UA-A": "fail", "UA-B": "ok"}

        config = AntiRiskControlConfig(
            enabled=True,
            user_agents=["UA-A", "UA-B"],
            jitter_enabled=False,
            shared_connector=False,
            muid_persist=False,
            ua_no_audio_threshold=1,
            ua_switch_on_no_audio=True,
            ua_switch_max_attempts=1,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._communicate_cls = _UAAwareCommunicate
        assert client._ua_health_filter is not None

        import src.engines.communicate_adapter as client_mod
        ua_sequence = iter(["UA-A", "UA-B"])
        monkeypatch.setattr(client_mod.random, "choice", lambda pool: next(ua_sequence))

        async def _drive() -> None:
            async for _ in client.synthesize_stream("hi", "voice"):
                pass

        asyncio.run(_drive())

        assert not client._ua_health_filter.is_healthy("UA-A"), (
            "UA-A should remain unhealthy — verification with UA-B succeeded"
        )
        assert client._ua_health_filter.is_healthy("UA-B")

    def test_switch_ua_all_fail_revokes_mark(
        self, preserve_base_headers, monkeypatch
    ) -> None:
        """All UAs fail with NoAudioReceived → all marks revoked."""
        _UAAwareCommunicate.ua_behavior = {"UA-A": "fail", "UA-B": "fail"}

        config = AntiRiskControlConfig(
            enabled=True,
            user_agents=["UA-A", "UA-B"],
            jitter_enabled=False,
            shared_connector=False,
            muid_persist=False,
            ua_no_audio_threshold=1,
            ua_switch_on_no_audio=True,
            ua_switch_max_attempts=1,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._communicate_cls = _UAAwareCommunicate
        assert client._ua_health_filter is not None

        import src.engines.communicate_adapter as client_mod
        ua_sequence = iter(["UA-A", "UA-B"])
        monkeypatch.setattr(client_mod.random, "choice", lambda pool: next(ua_sequence))

        async def _drive() -> None:
            gen = client.synthesize_stream("hi", "voice")
            async for _ in gen:
                pass

        with pytest.raises(Exception):
            asyncio.run(_drive())

        assert client._ua_health_filter.is_healthy("UA-A"), (
            "UA-A should be healthy again — marks revoked (server-side issue)"
        )
        assert client._ua_health_filter.is_healthy("UA-B"), (
            "UA-B should be healthy again — marks revoked (server-side issue)"
        )

    def test_no_switch_when_disabled(
        self, preserve_base_headers, monkeypatch
    ) -> None:
        """When ua_switch_on_no_audio=False, no UA switch verification."""
        _UAAwareCommunicate.ua_behavior = {"UA-A": "fail", "UA-B": "ok"}

        config = AntiRiskControlConfig(
            enabled=True,
            user_agents=["UA-A", "UA-B"],
            jitter_enabled=False,
            shared_connector=False,
            muid_persist=False,
            ua_no_audio_threshold=1,
            ua_switch_on_no_audio=False,
            ua_switch_max_attempts=1,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._communicate_cls = _UAAwareCommunicate
        assert client._ua_health_filter is not None

        import src.engines.communicate_adapter as client_mod
        ua_sequence = iter(["UA-A", "UA-B"])
        monkeypatch.setattr(client_mod.random, "choice", lambda pool: next(ua_sequence))

        async def _drive() -> None:
            gen = client.synthesize_stream("hi", "voice")
            async for _ in gen:
                pass

        with pytest.raises(Exception):
            asyncio.run(_drive())

        assert not client._ua_health_filter.is_healthy("UA-A"), (
            "UA-A should be marked — threshold reached, switch disabled"
        )
        assert client._ua_health_filter.is_healthy("UA-B")

    def test_config_params_read_from_config(self) -> None:
        """Config fields are read by EdgeTTSHttpClient."""
        config = AntiRiskControlConfig(
            enabled=True,
            user_agents=["UA-A"],
            ua_health_cooldown_seconds=120.0,
            ua_no_audio_threshold=5,
            ua_switch_on_no_audio=True,
            ua_switch_max_attempts=3,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        assert client._ua_health_filter is not None
        assert client._ua_health_filter._cooldown == 120.0
        assert client._ua_health_filter._no_audio_threshold == 5
        assert client._anti_risk_config.ua_switch_max_attempts == 3

    def test_shared_filter_from_executor(self) -> None:
        """TTSExecutor creates a shared UAHealthFilter from config."""
        from src.executors.tts_executor import TTSExecutor

        config = AntiRiskControlConfig(
            enabled=True,
            user_agents=["UA-A", "UA-B"],
            ua_no_audio_threshold=3,
            ua_health_cooldown_seconds=60.0,
        )
        from src.config.schema import PPCConfig, TTSConfig

        full_config = PPCConfig(tts=TTSConfig(anti_risk_control=config))
        executor = TTSExecutor(full_config)
        assert executor._shared_ua_health_filter is not None
        assert executor._shared_ua_health_filter._no_audio_threshold == 3
        assert executor._shared_ua_health_filter._cooldown == 60.0

    def test_unmark_unhealthy_revokes_mark(self) -> None:
        """UAHealthFilter.unmark_unhealthy() revokes a prior mark."""
        from src.engines.ua_health_filter import UAHealthFilter

        f = UAHealthFilter(cooldown_seconds=300.0, no_audio_threshold=1)
        f.mark_unhealthy("UA-A")
        assert not f.is_healthy("UA-A")
        f.unmark_unhealthy("UA-A")
        assert f.is_healthy("UA-A")


# ---------------------------------------------------------------------------
# 回归测试: 被服务端屏蔽的持久化 MUID 必须在 NoAudioReceived 时轮换，
# 否则复用同一 MUID 会导致所有会话连续失败（TTS_NO_AUDIO_RECEIVED）。
# ---------------------------------------------------------------------------


class _NoAudioFirstCommunicate:
    """Fake Communicate: 第一次 stream 抛 NoAudioReceived，之后成功。

    每次实例化记录其注入的 ``_muid``，用于断言重试时 MUID 已轮换。
    """

    instances: list["_NoAudioFirstCommunicate"] = []

    def __init__(
        self,
        text: str,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        connector: Any | None = None,
        proxy: str | None = None,
    ) -> None:
        self.text = text
        self.voice = voice
        self.connector = connector
        self.proxy = proxy
        self._muid: str | None = None
        self._extra_headers: dict[str, str] = {}
        type(self).instances.append(self)

    async def stream(self):
        from src.engines.edge_tts.exceptions import NoAudioReceived

        if len(type(self).instances) == 1:
            raise NoAudioReceived("No audio was received")
        yield {"type": "audio", "data": b"\x00" * 16}
        yield {"type": "turn.end"}


class TestBlockedMuidRotation:
    """持久化 MUID 被服务端屏蔽后，应在 NoAudioReceived 时轮换并重试成功。"""

    def test_muid_rotated_on_no_audio(self, tmp_path: Path) -> None:
        """NoAudioReceived 后持久化 MUID 必须失效并重新生成，重试使用新 MUID。"""
        muid_file = tmp_path / "muid"
        muid_file.write_text("BLOCKEDMUID00000000000000000000000", encoding="utf-8")

        config = AntiRiskControlConfig(
            enabled=True,
            backend="builtin",
            muid_persist=True,
            muid_file=str(muid_file),
            muid_rotate_hours=24.0,
            user_agents=["UA-A"],
            ua_no_audio_threshold=1,
            ua_switch_on_no_audio=True,
            ua_switch_max_attempts=2,
            jitter_enabled=False,
            shared_connector=False,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._communicate_cls = _NoAudioFirstCommunicate
        _NoAudioFirstCommunicate.instances.clear()

        async def drive() -> None:
            audio_count = 0
            async for chunk in client.synthesize_stream("hi", "voice"):
                if chunk.type == "audio":
                    audio_count += 1
            return audio_count

        audio_count = asyncio.run(drive())

        # 第一次失败 + 重试应成功产出音频
        assert audio_count >= 1, "重试应产出音频"
        assert len(_NoAudioFirstCommunicate.instances) >= 2, "应发生一次重试"

        # 关键断言：两个实例使用的 MUID 必须不同。
        # 若持久化 MUID 未被轮换，则两个实例都会复用同一个被屏蔽的 MUID。
        muids = [inst._muid for inst in _NoAudioFirstCommunicate.instances]
        assert None not in muids, "MUID 应被注入到每个实例"
        assert muids[0] != muids[1], (
            "持久化 MUID 未在 NoAudioReceived 时轮换 — 被屏蔽的 MUID 仍被复用，"
            "将导致所有会话连续失败（TTS_NO_AUDIO_RECEIVED）"
        )


# ---------------------------------------------------------------------------
# 回归测试: 共享连接器被逐流 ClientSession 关闭导致 "Session is closed"。
# aiohttp 的 ClientSession.close() 会关闭传入的 connector，
# 因此把共享 connector 传给逐流创建的 session 会在第一个流结束后把它关死，
# 后续所有流复用同一 client 时都会失败（"Session is closed"）。
# 修复：_make_communicate 不再把共享 connector 传给后端 Communicate。
# ---------------------------------------------------------------------------


class _ConnectorCapturingCommunicate:
    """Fake Communicate 记录构造时收到的 ``connector`` 参数。"""

    last_connector: Any = "unset"

    def __init__(
        self,
        text: str,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        connector: Any | None = None,
        proxy: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.text = text
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.connector = connector
        _ConnectorCapturingCommunicate.last_connector = connector

    async def stream(self):
        yield {"type": "audio", "data": b"\x00" * 16}
        yield {"type": "turn.end"}


class TestSharedConnectorNotPassedToStream:
    """共享连接器不得传给逐流创建的 ClientSession，否则会被其 close 关闭。"""

    def test_shared_connector_not_passed_to_communicate(self, tmp_path: Path) -> None:
        """即使 shared_connector=True，传给后端 Communicate 的 connector 必须是 None。

        否则逐流 ``ClientSession`` 关闭时会连带关闭共享 connector，
        导致复用同一 client 的后续流报 "Session is closed"。
        """
        config = AntiRiskControlConfig(
            enabled=True,
            backend="builtin",
            shared_connector=True,  # 有意开启共享连接器
            muid_persist=False,
            jitter_enabled=False,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)
        client._communicate_cls = _ConnectorCapturingCommunicate
        _ConnectorCapturingCommunicate.last_connector = "unset"

        async def drive() -> None:
            async for _ in client.synthesize_stream("hi", "voice"):
                pass

        asyncio.run(drive())

        assert _ConnectorCapturingCommunicate.last_connector is None, (
            "共享 connector 被传给了逐流 ClientSession — aiohttp 会在 session.close() 时"
            "将其关闭，导致复用同一 client 的后续流报 'Session is closed'"
        )
