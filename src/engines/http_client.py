"""Edge TTS default HTTP client with anti-risk-control support.

Split out of the former monolithic ``edge_tts_client`` module.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import (
    Any,
)

import aiohttp

from src.config.schema import AntiRiskControlConfig
from src.core.exceptions import (
    PermanentError,
    QuotaError,
)

from .client_base import EdgeTTSClient
from .communicate_adapter import CommunicateAdapter
from .muid_manager import MuidManager
from .tts_types import (
    DEFAULT_CHUNK_BYTES,
    DEFAULT_CONNECT_TIMEOUT,
    TTSChunk,
    VoiceInfo,
)
from .ua_health_filter import UAHealthFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend selection: prefer the customized src.engines.edge_tts,
# fall back to the environment edge-tts package on import failure.
# ---------------------------------------------------------------------------
_BUILTIN_COMMUNICATE = None
_BUILTIN_LIST_VOICES = None
_BUILTIN_EXCEPTIONS = None
_BUILTIN_BACKEND_NAME = None

try:
    from src.engines.edge_tts import Communicate as _BUILTIN_COMMUNICATE
    from src.engines.edge_tts import list_voices as _BUILTIN_LIST_VOICES
    from src.engines.edge_tts import exceptions as _BUILTIN_EXCEPTIONS

    _BUILTIN_BACKEND_NAME = "src.engines.edge_tts (customized)"
except ImportError as _exc:
    logger.warning(
        "Customized edge_tts backend unavailable, falling back to environment edge-tts: %s",
        _exc,
    )

# Environment edge-tts as fallback (always imported to support backend=auto/env)
try:
    import edge_tts as _ENV_EDGE_TTS

    _ENV_COMMUNICATE = _ENV_EDGE_TTS.Communicate
    _ENV_LIST_VOICES = _ENV_EDGE_TTS.list_voices
    _ENV_EXCEPTIONS = _ENV_EDGE_TTS.exceptions
    _ENV_BACKEND_NAME = "edge-tts (environment)"
except ImportError:  # pragma: no cover - exercised only in stripped env
    _ENV_EDGE_TTS = None
    _ENV_COMMUNICATE = None
    _ENV_LIST_VOICES = None
    _ENV_EXCEPTIONS = None
    _ENV_BACKEND_NAME = None


@dataclass
class _UAVerificationState:
    """UA switch verification state for a single synthesis run.

    Tracks UAs marked during this synthesis so marks can be revoked if
    all UAs fail (server-side issue, not UA-specific).

    ``tried`` records the distinct UAs already attempted in this run so
    that "tried N different UAs without success" is a reachable state
    and the next retry is steered to an untried UA.
    """

    marked: list[str] = field(default_factory=list)
    tried: set[str] = field(default_factory=set)
    attempts: int = 0
    max_attempts: int = 0


class EdgeTTSHttpClient(EdgeTTSClient):
    """Default Edge TTS client with anti-risk-control support."""

    def __init__(
        self,
        *,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        communicate_factory: Any | None = None,
        list_voices_factory: Any | None = None,
        anti_risk_config: AntiRiskControlConfig | None = None,
        ua_health_filter: UAHealthFilter | None = None,
        tts_backend_plugins: list | None = None,
        backend_plugin: str | None = None,
    ) -> None:
        super().__init__(
            connect_timeout=connect_timeout,
            chunk_bytes=chunk_bytes,
        )
        # ``communicate_factory`` allows tests to inject a fake factory
        # without monkey-patching ``edge_tts.Communicate`` globally.
        self._communicate_factory = communicate_factory
        # ``list_voices_factory`` allows tests to inject a fake voice
        # manager without monkey-patching ``edge_tts.list_voices``.
        self._list_voices_factory = list_voices_factory
        self._anti_risk_config = anti_risk_config or AntiRiskControlConfig()

        # TTS 后端插件：注入已注册的 ``TTSBackend`` 插件实例（默认空列表）。
        # 当 ``backend_plugin`` 配置了名称且能解析到可用插件时，合成走插件
        # 路径；否则走默认的本地 edge_tts 路径（完全向后兼容）。
        self._tts_backend_plugins: list = list(tts_backend_plugins or [])
        self._backend_plugin_name: str | None = backend_plugin

        # 实际产出音频的语音名（可能因 QuotaError 回退而与请求语音不同）。
        # 由 ``_synthesize`` 在流成功结束时写入，供引擎构造**缓存写入键**使用：
        # 回退产物绝不能存到「请求语音」的 key 下，否则后续该语音的请求会命中
        # 缓存并静默返回另一语音的音频（且经 L2 磁盘缓存跨进程固化）。
        self.last_effective_voice: str | None = None

        # Backend selection — driven by the three-state ``backend`` field
        # (auto | builtin | env). The legacy boolean ``prefer_builtin`` is
        # migrated to ``backend`` by ``AntiRiskControlConfig`` and is no
        # longer read here.
        backend = self._anti_risk_config.backend
        if backend == "auto":
            # Prefer the customized backend; fall back to the environment
            # edge-tts package with a WARNING when it is unavailable.
            self._communicate_cls = _BUILTIN_COMMUNICATE or _ENV_COMMUNICATE
            self._list_voices_fn = _BUILTIN_LIST_VOICES or _ENV_LIST_VOICES
            self._backend_name = _BUILTIN_BACKEND_NAME if _BUILTIN_COMMUNICATE else _ENV_BACKEND_NAME
            if not _BUILTIN_COMMUNICATE and _ENV_COMMUNICATE:
                logger.warning(
                    "定制版 edge_tts 不可用，回退到环境 edge-tts 包 / "
                    "Customized edge_tts unavailable, falling back to environment edge-tts package"
                )
        elif backend == "builtin":
            if _BUILTIN_COMMUNICATE is None:
                raise PermanentError(
                    "Customized edge_tts backend required but unavailable"
                )
            self._communicate_cls = _BUILTIN_COMMUNICATE
            self._list_voices_fn = _BUILTIN_LIST_VOICES
            self._backend_name = _BUILTIN_BACKEND_NAME
        else:  # backend == "env"
            if _ENV_COMMUNICATE is None:
                raise PermanentError(
                    "Environment edge-tts package required but not installed"
                )
            self._communicate_cls = _ENV_COMMUNICATE
            self._list_voices_fn = _ENV_LIST_VOICES
            self._backend_name = _ENV_BACKEND_NAME

        # Connection-level retry budget and per-request timeout. The timeout
        # is applied either via the backend constructor (when supported) or
        # through ``asyncio.wait_for`` while streaming in ``_synthesize``.
        self._max_connect_retries = self._anti_risk_config.max_connect_retries
        self._request_timeout_seconds = self._anti_risk_config.request_timeout_seconds

        # MuidManager
        self._muid_manager: MuidManager | None = None
        if self._anti_risk_config.muid_persist and self._anti_risk_config.enabled:
            self._muid_manager = MuidManager(
                file_path=self._anti_risk_config.muid_file,
                rotate_hours=self._anti_risk_config.muid_rotate_hours,
                persist=True,
            )

        # UA health filter — shared across all streams so a UA that
        # fails on one stream is excluded from rotation on concurrent
        # streams. When *ua_health_filter* is injected (from the
        # executor layer), it is reused so that NoAudioReceived counts
        # accumulate across executor-level silent retries. Otherwise a
        # new instance is created from config.
        self._ua_health_filter: UAHealthFilter | None = ua_health_filter
        if self._ua_health_filter is None and self._anti_risk_config.enabled and self._anti_risk_config.user_agents:
            self._ua_health_filter = UAHealthFilter(
                cooldown_seconds=self._anti_risk_config.ua_health_cooldown_seconds,
                no_audio_threshold=self._anti_risk_config.ua_no_audio_threshold,
            )

        # 共享连接器（``anti_risk_control.shared_connector``）已废弃：
        # aiohttp 的 ``ClientSession.close()`` 会关闭传入的 connector，而后端
        # ``Communicate._stream`` 是逐流用 ``async with ClientSession(...)``
        # 创建会话的——共享 connector 会被第一个结束的流关死（详见
        # ``_make_communicate`` 注释）。因此本客户端不再创建/持有任何
        # 跨流连接器，``shared_connector`` 配置项仅为兼容旧配置文件保留。

        logger.info("EdgeTTSHttpClient 后端 / backend: %s", self._backend_name)

    def _resolve_tts_backend(self, backend_name: str | None) -> Any | None:
        """按名称解析 ``ppc.tts.v1`` 后端。

        优先命中注入的插件列表（测试/宿主显式传入），其次回落到全局后端
        注册表（内置后端与扩展加载器注册的后端）。两路都要求通过协议协商
        且 ``is_available()``；否则返回 None，调用方回退到本地 edge_tts 路径。
        """
        if not backend_name:
            return None
        for plugin in self._tts_backend_plugins:
            try:
                if plugin.get_name() == backend_name and plugin.is_available():
                    return plugin
            except Exception:  # noqa: BLE001
                logger.debug("TTS 后端插件解析失败 / plugin resolve failed: %r", plugin)
        from .tts_protocol import backend_registry, ensure_builtin_backends

        ensure_builtin_backends()
        return backend_registry.resolve(backend_name)

    def _make_communicate(
        self,
        text: str,
        voice: str,
        rate: str,
        volume: str,
        *,
        exclude_uas: set[str] | None = None,
    ) -> Any:
        # Test hook has highest priority
        if self._communicate_factory is not None:
            return self._communicate_factory(text, voice, rate, volume)

        if self._communicate_cls is None:
            raise PermanentError("No Edge TTS backend available (neither customized nor environment)")

        # 注意：绝不把共享 connector 传给后端 Communicate。
        # aiohttp 的 ``ClientSession.close()`` 会关闭传入的 connector
        # （即使它由外部创建），而后端 ``Communicate._stream`` 是逐流用
        # ``async with ClientSession(connector=...)`` 创建会话的——若把共享
        # connector 传进去，第一个流结束时 session.close() 就会把它关死，
        # 导致复用同一 client 的后续流全部报 "Session is closed"。
        # 因此这里传 ``connector=None``，让每个流拥有并自行关闭自己的 connector。
        # 这也是 ``anti_risk_control.shared_connector`` 被废弃的原因：在该语义下
        # 共享连接器必然被首个流的 session.close() 关死，无法安全复用。
        kwargs: dict[str, Any] = {
            "rate": rate,
            "volume": volume,
            "connector": None,
            "proxy": (self._anti_risk_config.proxy or None) if self._anti_risk_config.enabled else None,
        }
        # Forward ``request_timeout_seconds`` when the backend ``Communicate``
        # constructor exposes a ``timeout`` keyword.  Backends that do not
        # support it (e.g. the customized ``edge_tts.Communicate`` which
        # only has ``connect_timeout``/``receive_timeout``) are instead
        # guarded by ``asyncio.wait_for`` while streaming in ``_synthesize``.
        try:
            sig = inspect.signature(self._communicate_cls.__init__)
            params = sig.parameters
            if "timeout" in params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            ):
                kwargs["timeout"] = self._request_timeout_seconds
        except (ValueError, TypeError):  # pragma: no cover - signature inspection failure
            pass

        communicate = self._communicate_cls(text, voice, **kwargs)

        # Wrap with CommunicateAdapter if anti-risk-control is enabled
        if self._anti_risk_config.enabled:
            communicate = CommunicateAdapter(
                communicate,
                anti_risk_config=self._anti_risk_config,
                muid_manager=self._muid_manager,
                ua_health_filter=self._ua_health_filter,
                exclude_uas=exclude_uas,
            )
        return communicate

    async def _synthesize(
        self,
        text: str,
        voice: str,
        *,
        rate: str,
        volume: str,
    ) -> AsyncIterator[TTSChunk]:
        """Entry point for synthesis with optional voice fallback.

        When ``voice_fallbacks`` is configured, iterates over
        ``[voice] + voice_fallbacks`` and falls back to the next voice
        only on :class:`QuotaError`. Other exceptions
        (``TransientError`` / ``PermanentError`` / ``NetworkError``)
        propagate immediately per the original logic. When all voices
        are exhausted, the original :class:`QuotaError` from the last
        attempt is re-raised without wrapping.

        When ``voice_fallbacks`` is empty, delegates directly to
        :meth:`_synthesize_single_voice` to preserve the original
        single-voice behavior with no extra overhead.

        **已产出音频后禁止回退**：一旦本次尝试已经向下游 yield 过任何块
        （``emitted_any``），再换音色重来会让同一输出文件的前段是音色 A、
        后段是音色 B（缓存又按 effective voice 归档，事后无法察觉）。
        此时直接把 :class:`QuotaError` 上抛，由上层整段重试。

        Side effect: :attr:`last_effective_voice` is updated to the voice that
        actually produced the audio. The engine uses it as the **cache key
        voice** so that a fallback result is never stored under the requested
        (failed) voice — otherwise later requests for that voice would hit the
        cache and silently return a different voice's audio.
        """
        fallbacks = self._anti_risk_config.voice_fallbacks
        if not fallbacks:
            async for chunk in self._synthesize_single_voice(
                text, voice, rate=rate, volume=volume
            ):
                yield chunk
            self.last_effective_voice = voice
            return

        voices_to_try = [voice, *fallbacks]
        last_quota_exc: QuotaError | None = None
        for idx, current_voice in enumerate(voices_to_try):
            # 触发条件：服务端在已经下发若干音频块后才返回 429 / QuotaError。
            # 后果（旧实现）：循环继续换下一个音色并继续 yield，同一个输出
            # 文件里前段是音色 A、后段是音色 B，且无法事后察觉。
            # 因此只要本次尝试已产出过块，就不得回退，直接上抛 QuotaError。
            emitted_any = False
            try:
                async for chunk in self._synthesize_single_voice(
                    text, current_voice, rate=rate, volume=volume
                ):
                    emitted_any = True
                    yield chunk
                self.last_effective_voice = current_voice
                return  # Success — exit the fallback loop.
            except Exception as exc:
                classified = self._classify_exception(exc)
                if not isinstance(classified, QuotaError):
                    # Non-QuotaError: re-raise the original exception;
                    # ``synthesize_stream`` will classify it for callers.
                    raise
                if emitted_any:
                    logger.warning(
                        "已产出音频后遇配额错误，放弃音色回退 / aborting voice "
                        "fallback after partial audio: voice=%s, error=%s",
                        current_voice,
                        classified,
                    )
                    raise classified
                last_quota_exc = classified
                is_last = idx == len(voices_to_try) - 1
                if is_last:
                    logger.warning(
                        "Edge TTS voice 回退耗尽 / voice fallbacks exhausted: "
                        "voice=%s, tried=%s, last_error=%s",
                        voice,
                        voices_to_try,
                        classified,
                    )
                    raise classified
                next_voice = voices_to_try[idx + 1]
                logger.info(
                    "Edge TTS voice 回退 / voice fallback: from %s to %s (reason: %s)",
                    current_voice,
                    next_voice,
                    classified,
                )
        # Defensive — unreachable: every loop iteration either returns
        # (success) or raises (QuotaError / other exception).
        if last_quota_exc is not None:
            raise last_quota_exc

    async def _pump_stream(
        self,
        stream: Any,
    ) -> AsyncIterator[tuple[str, bytes, dict[str, Any] | None]]:
        """Yield normalized ``(kind, data, metadata)`` messages from *stream*.

        ``kind`` is ``"audio"`` for binary frames or the boundary event
        name for metadata frames. Control frames (``turn.end`` etc.) are
        ignored. ``StopAsyncIteration`` terminates the iteration.

        ``asyncio.wait_for`` enforces ``request_timeout_seconds``
        per streamed message — this bounds both the WebSocket
        handshake and the inter-chunk latency.  Backends whose
        ``Communicate`` constructor does not accept a ``timeout``
        kwarg are guarded here.
        """
        while True:
            try:
                message = await asyncio.wait_for(
                    stream.__anext__(),
                    timeout=self._request_timeout_seconds,
                )
            except StopAsyncIteration:
                return
            kind = message.get("type") if isinstance(message, dict) else None
            if kind == "audio":
                data = message.get("data", b"")
                if isinstance(data, str):
                    data = data.encode("utf-8")
                yield kind, data, None
            elif kind in {"WordBoundary", "SentenceBoundary"}:
                yield kind, b"", message
            # "turn.end" or other control frames are ignored.

    @staticmethod
    async def _aclose_stream(stream: Any) -> None:
        """Close an abandoned stream so the WebSocket / session is released."""
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception as e:  # noqa: BLE001
                logger.debug("流关闭失败: %s", e)

    async def _retry_on_no_audio(
        self,
        exc: BaseException,
        communicate: Any,
        stream: Any,
        verification: _UAVerificationState,
    ) -> bool:
        """Handle a NoAudioReceived failure; return True to retry with another UA.

        Records the failure against the selected UA. If the UA hits the
        threshold, it is marked unhealthy and the caller retries with a
        different UA (verification). If all verification attempts also
        fail with NoAudioReceived, every mark set during this synthesis
        is revoked (server-side issue, not UA-specific) and the caller
        re-raises.
        """
        # 被物理屏蔽的持久化 MUID 会表现为 NoAudioReceived，且复用它会
        # 导致所有会话连续失败。这里立即废止持久化 MUID，使下一次
        # ``get_muid()`` 生成全新 MUID，从而在重试时绕开被屏蔽的 MUID。
        if self._muid_manager is not None:
            self._muid_manager.invalidate()
        current_ua = getattr(communicate, "last_selected_ua", None)
        if current_ua is None:
            return False

        marked = self._ua_health_filter.record_no_audio(current_ua)
        if marked and current_ua not in verification.marked:
            # Track for potential revocation if all UAs fail.
            verification.marked.append(current_ua)

        # 是否继续换 UA 验证：
        # - 首次（``attempts == 0``）只在当前 UA **真正达到阈值**时启动，
        #   避免一次瞬时 NoAudioReceived 就无谓轮换；
        # - 已处于验证流程（``attempts > 0``）时，后续任何 NoAudioReceived
        #   都继续换 UA。这是原缺陷的根因：新选 UA 的计数恒从 1 开始，永远
        #   达不到 ``ua_no_audio_threshold``，若仍以 ``marked`` 为门槛，
        #   attempts 永远停在 1，撤销分支（unmark_unhealthy）不可达 →
        #   服务端级故障时可用 UA 池被永久缩小。
        if verification.attempts > 0 or marked:
            verification.tried.add(current_ua)
            if verification.attempts < verification.max_attempts:
                remaining = [
                    ua
                    for ua in self._anti_risk_config.user_agents
                    if ua not in verification.tried
                ]
                # 首次（阈值触发）即使池中已无其他 UA 也重试一次：上面已
                # 废止 MUID，重试会携带新 MUID，可绕开被屏蔽的 MUID；
                # 后续则仅在确实还有「没试过的 UA」时才继续，避免在同一
                # UA 上反复打转却计为多次尝试。
                if remaining or verification.attempts == 0:
                    verification.attempts += 1
                    logger.info(
                        "NoAudioReceived 达阈值，切换 UA 验证 / "
                        "NoAudioReceived threshold reached, switching UA "
                        "(verification %d/%d, excluded UA: %s)",
                        verification.attempts,
                        verification.max_attempts,
                        current_ua[:50] + "..." if len(current_ua) > 50 else current_ua,
                    )
                    # Close the abandoned stream and retry — ``tried`` is
                    # passed to the adapter so the next selection skips
                    # already-attempted UAs.
                    await self._aclose_stream(stream)
                    return True

        # Verification attempts exhausted (or no alternate UA available).
        # Only revoke when we actually tried **at least two distinct** UAs:
        # with a single-UA pool the failure is not evidence of a server-side
        # fault, and revoking would discard a valid threshold mark (also
        # breaks MUID-rotation retries on single-UA pools).
        if (
            verification.attempts > 0
            and verification.attempts >= verification.max_attempts
            and len(verification.tried) >= 2
            and verification.marked
        ):
            logger.info(
                "所有 UA 均 NoAudioReceived，撤销标记（服务端问题）/ "
                "All UAs got NoAudioReceived, revoking marks (server-side issue)"
            )
            for ua in verification.marked:
                self._ua_health_filter.unmark_unhealthy(ua)
            verification.marked.clear()
        return False

    async def _retry_on_connect_failure(
        self,
        exc: BaseException,
        stream: Any,
        produced: bool,
        attempt: int,
        total_max: int,
    ) -> bool:
        """Handle a non-NoAudio failure; return True to retry the connection.

        Only retry on connection-level failures that occurred
        *before* any chunk was yielded — once partial audio has
        been produced we cannot safely restart the stream.
        """
        if produced or not self._is_connect_failure(exc):
            # Close the abandoned stream before raising so the
            # underlying WebSocket / aiohttp session is released
            # synchronously, preventing "Session is closed" errors
            # under high concurrency.
            await self._aclose_stream(stream)
            return False
        # Close the abandoned stream before retrying so the
        # underlying WebSocket / aiohttp session is released.
        await self._aclose_stream(stream)
        if attempt < total_max:
            logger.info(
                "Edge TTS 连接失败，重试 / connect failed, retrying (%d/%d): %s",
                attempt,
                self._max_connect_retries,
                exc,
            )
            return True
        return False

    async def _synthesize_single_voice(
        self,
        text: str,
        voice: str,
        *,
        rate: str,
        volume: str,
    ) -> AsyncIterator[TTSChunk]:
        """Synthesize audio for a single voice with connect-level retries.

        Implements the original ``_synthesize`` logic: a
        ``max_connect_retries`` retry loop around the WebSocket connect
        and initial stream handshake. Only connection-level failures
        (timeouts, connection errors, ``WebSocketError``) are retried;
        any exception raised after partial audio has been yielded is
        re-raised immediately.

        **NoAudioReceived UA switch verification**: when
        ``ua_switch_on_no_audio`` is enabled and a UA accumulates
        ``ua_no_audio_threshold`` consecutive ``NoAudioReceived``
        failures (tracked in the shared ``UAHealthFilter``), the UA is
        marked unhealthy and the synthesis is retried with a different
        UA. If the alternate UA succeeds, the original UA's mark is
        confirmed. If **all** alternate UAs also fail with
        ``NoAudioReceived``, the original UA's mark is **revoked** —
        the failure is server-side, not UA-specific.
        """
        offset = 0
        max_attempts = self._max_connect_retries + 1

        verification = _UAVerificationState(
            max_attempts=(
                self._anti_risk_config.ua_switch_max_attempts
                if self._anti_risk_config.ua_switch_on_no_audio
                else 0
            )
        )
        # Ensure the loop has enough iterations for both connect retries
        # and UA switch verification.
        total_max = max(max_attempts, verification.max_attempts + 1)

        for attempt in range(1, total_max + 1):
            communicate = self._make_communicate(
                text,
                voice,
                rate,
                volume,
                # 已验证失败的 UA 从本次选择的候选池中排除，保证每次重试
                # 真正落到一个新的 UA 上（否则 attempts 可能在同一 UA 上累加，
                # "试过 N 个不同 UA" 的条件仍不可达）。
                exclude_uas=verification.tried,
            )
            stream = communicate.stream()
            produced = False
            try:
                async for kind, data, metadata in self._pump_stream(stream):
                    if kind == "audio":
                        yield TTSChunk(offset=offset, data=data, type="audio")
                        offset += len(data)
                    else:
                        yield TTSChunk(
                            offset=offset,
                            data=b"",
                            type="metadata",
                            metadata=metadata,
                        )
                    produced = True
                return
            except asyncio.CancelledError:
                # 触发条件：外层任务被取消（段级超时 / 用户中断）时，
                # ``wait_for`` 抛出的 CancelledError 继承自 BaseException，
                # 原来的 ``except Exception`` 捕获不到，流不会被 aclose，
                # 底层 ClientSession/WebSocket 只能等 asyncgen GC 才关闭，
                # 高并发下会长期占用连接甚至报 "Session is closed"。
                # 这里显式关闭后原样抛出，保证取消语义不变。
                await self._aclose_stream(stream)
                raise
            except Exception as exc:
                if (
                    not produced
                    and self._ua_health_filter is not None
                    and UAHealthFilter.is_no_audio_failure(exc)
                ):
                    if await self._retry_on_no_audio(exc, communicate, stream, verification):
                        continue  # retry with a different UA
                    raise
                if not await self._retry_on_connect_failure(exc, stream, produced, attempt, total_max):
                    raise

    @staticmethod
    def _is_connect_failure(exc: BaseException) -> bool:
        """Return True if *exc* is a connection-level failure eligible for retry.

        Covers WebSocket / aiohttp connect errors, generic connection errors
        and request timeouts.  ``WebSocketError`` is matched by name because
        the module-level alias may be a plain ``Exception`` fallback when the
        customized backend is unavailable.
        """
        if isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError)):
            return True
        if isinstance(exc, aiohttp.ClientConnectorError):
            return True
        return type(exc).__name__ == "WebSocketError"

    async def _list_voices(
        self,
        *,
        locale: str | None,
        gender: str | None,
    ) -> list[VoiceInfo]:
        # ppc.tts.v1 后端若声明 voices 能力，语音枚举同样路由给后端
        # （ppc voices / WebUI 下拉在插件后端下不再是空列表或 edge 列表）。
        plugin = self._resolve_tts_backend(self._backend_plugin_name)
        if plugin is not None:
            from .tts_protocol import capabilities_of

            if capabilities_of(plugin).voices:
                raw = await plugin.list_voices()
                voices: list[VoiceInfo] = []
                for entry in raw or []:
                    if not isinstance(entry, dict):
                        continue
                    short = str(entry.get("short_name") or entry.get("name") or "")
                    if not short:
                        continue
                    g = str(entry.get("gender") or "")
                    if locale and not short.startswith(locale):
                        continue
                    if gender and g.lower() != gender.lower():
                        continue
                    voices.append(
                        VoiceInfo(
                            name=str(entry.get("name") or short),
                            short_name=short,
                            gender=g,
                            locale=str(entry.get("locale") or ""),
                            friendly_name=str(entry.get("friendly_name") or short),
                        )
                    )
                return voices
        if self._list_voices_factory is not None:
            manager = await self._list_voices_factory()
        else:
            if self._list_voices_fn is None:
                raise PermanentError("No Edge TTS backend available for list_voices")
            proxy = (self._anti_risk_config.proxy or None) if self._anti_risk_config.enabled else None
            manager = await self._list_voices_fn(proxy=proxy)
        # ``edge_tts.list_voices`` returns a list-like manager. In
        # some library versions it is a plain ``List[Voice]`` while in
        # others it exposes a ``.voices`` attribute; we handle both.
        entries = getattr(manager, "voices", None)
        if entries is None:
            entries = list(manager) if manager else []
        voices: list[VoiceInfo] = []
        for entry in entries or []:
            short = entry.get("ShortName", "") if isinstance(entry, dict) else getattr(entry, "ShortName", "")
            if locale and not short.startswith(locale):
                continue
            g = entry.get("Gender", "") if isinstance(entry, dict) else getattr(entry, "Gender", "")
            if gender and g.lower() != gender.lower():
                continue
            name = str(entry.get("Name", short) if isinstance(entry, dict) else getattr(entry, "Name", short))
            locale_val = str(
                entry.get("Locale", short.split("-")[0] if short else "")
                if isinstance(entry, dict)
                else getattr(entry, "Locale", short.split("-")[0] if short else "")
            )
            friendly = str(
                entry.get("FriendlyName", short) if isinstance(entry, dict) else getattr(entry, "FriendlyName", short)
            )
            voices.append(
                VoiceInfo(
                    name=name,
                    short_name=short,
                    gender=g,
                    locale=locale_val,
                    friendly_name=friendly,
                )
            )
        return voices

    async def cleanup(self) -> None:
        """Release shared resources.

        共享连接器已废弃（见 ``_make_communicate`` 注释：aiohttp 的
        ``ClientSession.close()`` 会关闭传入的 connector），本客户端不再
        持有任何跨流共享资源。保留空实现以兼容 ``TTSEngine.cleanup()``
        的调用契约，重复调用无副作用。
        """


__all__ = [
    "EdgeTTSHttpClient",
]
