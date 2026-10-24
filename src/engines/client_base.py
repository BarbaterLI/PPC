"""Edge TTS client base: abstract interface + exception classification.

Split out of the former monolithic ``edge_tts_client`` module.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import (
    Any,
)

import aiohttp

from src.core.exceptions import (
    NetworkError,
    PermanentError,
    QuotaError,
    TransientError,
)
from src.utils.files import atomic_copy_file, atomic_write_bytes

from .tts_types import (
    DEFAULT_CHUNK_BYTES,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_RESUME_OFFSET,
    TTSChunk,
    VoiceInfo,
)

logger = logging.getLogger(__name__)

# Environment edge-tts is the only backend whose exception hierarchy is
# consulted here; the builtin/customized backend's exceptions are matched
# by name in ``_classify_exception``.
try:
    import edge_tts as _ENV_EDGE_TTS
except ImportError:  # pragma: no cover - exercised only in stripped env
    _ENV_EDGE_TTS = None

# Exception class references for ``_classify_exception``.  These are looked
# up by ``type(exc).__name__`` string comparison, so the actual class objects
# are only needed for the optional ``isinstance`` path in
# ``_is_edge_tts_exception``.
_EDGE_TTS_AVAILABLE = _ENV_EDGE_TTS is not None
edge_tts = _ENV_EDGE_TTS
EdgeTTSException = Exception  # type: ignore[misc,assignment]
NoAudioReceived = Exception  # type: ignore[misc,assignment]
SkewAdjustmentError = Exception  # type: ignore[misc,assignment]
UnexpectedResponse = Exception  # type: ignore[misc,assignment]
UnknownResponse = Exception  # type: ignore[misc,assignment]
WebSocketError = Exception  # type: ignore[misc,assignment]


def _is_edge_tts_exception(exc: BaseException) -> bool:
    """Return True if *exc* is a subclass of ``edge_tts``'s base exception.

    Falls back to a name-based check when ``edge_tts`` is unavailable.
    """
    if not _EDGE_TTS_AVAILABLE:
        return False
    try:
        return isinstance(exc, edge_tts.exceptions.EdgeTTSException)
    except AttributeError:
        return type(exc).__name__ == "EdgeTTSException"


class EdgeTTSClient(ABC):
    """Abstract base class for Edge TTS clients.

    Subclasses must implement :meth:`_synthesize` and :meth:`_list_voices`.
    The concrete :class:`EdgeTTSHttpClient` is the default.
    """

    def __init__(
        self,
        *,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        self.connect_timeout = connect_timeout
        self.chunk_bytes = chunk_bytes

    @abstractmethod
    def _synthesize(
        self,
        text: str,
        voice: str,
        *,
        rate: str,
        volume: str,
    ) -> AsyncIterator[TTSChunk]:
        """Backend-specific audio generator."""
        ...

    @abstractmethod
    async def _list_voices(
        self,
        *,
        locale: str | None,
        gender: str | None,
    ) -> list[VoiceInfo]:
        """Backend-specific voice listing."""
        ...

    async def synthesize_stream(
        self,
        text: str,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        last_chunk_offset: int = DEFAULT_RESUME_OFFSET,
    ) -> AsyncIterator[TTSChunk]:
        """Yield audio chunks with optional resume support.

        Any underlying exception is re-classified through
        :meth:`_classify_exception` so callers see the refined
        :mod:`src.core.exceptions` taxonomy.
        """
        if last_chunk_offset and last_chunk_offset < 0:
            raise PermanentError(f"last_chunk_offset must be non-negative, got {last_chunk_offset}")

        offset = int(last_chunk_offset or 0)
        try:
            async for chunk in self._synthesize(text, voice, rate=rate, volume=volume):
                if offset and chunk.offset < offset:
                    continue
                yield chunk
        except Exception as exc:  # noqa: BLE001
            raise self._classify_exception(exc) from exc

    async def list_voices(
        self,
        *,
        locale: str | None = None,
        gender: str | None = None,
    ) -> list[VoiceInfo]:
        try:
            return await self._list_voices(locale=locale, gender=gender)
        except Exception as exc:  # noqa: BLE001
            raise self._classify_exception(exc) from exc

    async def synthesize_to_file(
        self,
        text: str,
        output_path: Any,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        last_chunk_offset: int = DEFAULT_RESUME_OFFSET,
        boundary_sink: Any | None = None,
    ) -> int:
        """Convenience helper that streams audio to a file on disk.

        Writes to a ``.part`` temporary file and atomically renames to
        the target path on success. On failure the temp file is deleted
        so that no partial audio data remains in the cache directory.

        ``boundary_sink`` 为可选的元数据接收器（``callable(TTSChunk)``）：
        传入后，``WordBoundary``/``SentenceBoundary`` 事件不再被丢弃，而是
        逐个交给它——多角色合并优化靠这些 offset/duration 做时间轴对齐与
        音频切分（见 :mod:`src.engines.segment_timeline`）。接收器自身抛
        异常不影响合成主流程（字幕/切分是增强能力，不能反过来打断产出）。

        Returns the total number of bytes written (post-resume).
        """
        import os
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # TTS 后端插件路由：当配置了 backend_plugin 且能解析到可用插件时，
        # 走插件合成路径；否则回退到默认的本地 edge_tts 路径。
        # 通过 getattr 兼容基类 EdgeTTSClient 实例（无插件注入时走默认路径）。
        resolver = getattr(self, "_resolve_tts_backend", None)
        if resolver is not None:
            plugin = resolver(getattr(self, "_backend_plugin_name", None))
            if plugin is not None:
                try:
                    result = await plugin.synthesize(
                        text,
                        voice,
                        rate=rate,
                        volume=volume,
                        pitch="+0Hz",
                        output_path=output_path,
                    )
                    # 插件可能返回已写入文件的路径或音频字节数据；统一确保
                    # output_path 存在且非空，再返回字节数。
                    # 触发条件：插件直接产出字节或另存了临时文件。
                    # 后果（旧实现 ``write_bytes``/``shutil.copy2``）：进程在
                    # 复制途中中断会在 output_path 留下半截文件，且绕开了
                    # ``synthesize_atomic_write`` 的原子落盘保证。
                    # 统一改走原子工具（同目录 .part + fsync + os.replace）。
                    if isinstance(result, (bytes, bytearray)) and result:
                        atomic_write_bytes(path, bytes(result))
                    elif result is not None and str(result) not in ("", str(path)):
                        atomic_copy_file(str(result), str(path))
                    if not path.exists() or path.stat().st_size <= 0:
                        raise TransientError("TTS backend plugin returned no audio")
                    return path.stat().st_size
                except BaseException as exc:
                    # 用现有 _classify_exception 归类，保证可被上层重试。
                    raise self._classify_exception(exc) from exc

        # If resuming and a non-empty target file already exists, open
        # the temp file in append mode; otherwise truncate.
        mode = "ab" if last_chunk_offset and path.exists() and path.stat().st_size > 0 else "wb"
        # 当处于断点续传模式时，用已有文件内容初始化临时文件，
        # 使后续的 append 模式能从正确的偏移继续写入。
        temp_path = path.with_suffix(path.suffix + ".part")
        if mode == "ab" and path.exists() and path.stat().st_size > 0:
            import shutil
            shutil.copy2(str(path), str(temp_path))

        written = 0
        try:
            with temp_path.open(mode) as fh:
                async for chunk in self.synthesize_stream(
                    text,
                    voice,
                    rate=rate,
                    volume=volume,
                    last_chunk_offset=last_chunk_offset,
                ):
                    if chunk.type != "audio" or not chunk.data:
                        # 元数据（WordBoundary/SentenceBoundary）交接收器；
                        # 无接收器时行为与旧实现完全一致（直接丢弃）。
                        if boundary_sink is not None and getattr(chunk, "type", "") == "metadata":
                            try:
                                boundary_sink(chunk)
                            except Exception as sink_exc:  # noqa: BLE001
                                logger.debug("boundary_sink 处理失败（已忽略）: %s", sink_exc)
                        continue
                    fh.write(chunk.data)
                    written += len(chunk.data)
            # 成功时原子性地把临时文件重命名为目标路径
            os.replace(str(temp_path), str(path))
        except BaseException:
            # 任何异常（包括 CancelledError）都要删除临时文件，
            # 避免缓存目录中残留不完整的音频数据。
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise
        return written

    @staticmethod
    def _classify_exception(exc: BaseException) -> Exception:
        """Map any underlying exception to the refined taxonomy.

        Edge TTS exception classes are looked up by *name* rather than
        direct ``isinstance`` checks. This keeps the classification
        robust against monkey-patching during unit tests and against
        edge-tts micro-version changes that rename internal classes.
        """
        if isinstance(exc, (TransientError, PermanentError, QuotaError, NetworkError)):
            return exc

        cls_name = type(exc).__name__
        message = str(exc).lower()

        # Edge TTS library-specific exceptions (looked up by name).
        if cls_name == "NoAudioReceived":
            no_audio_exc = TransientError("Edge TTS returned no audio (transient)")
            setattr(no_audio_exc, "error_code", "TTS_NO_AUDIO_RECEIVED")
            return no_audio_exc
        if cls_name == "SkewAdjustmentError":
            return TransientError("Edge TTS skew adjustment failure (transient)")
        if cls_name == "WebSocketError":
            return NetworkError(f"Edge TTS websocket error: {exc}")
        if cls_name == "UnexpectedResponse":
            return PermanentError(f"Edge TTS unexpected response: {exc}")
        if cls_name == "UnknownResponse":
            return PermanentError(f"Edge TTS unknown response: {exc}")
        if cls_name == "EdgeTTSException" or _is_edge_tts_exception(exc):
            if "quota" in message or "rate" in message or "429" in message:
                return QuotaError(f"Edge TTS quota exceeded: {exc}")
            return TransientError(f"Edge TTS transient failure: {exc}")

        # Generic / network-level exceptions.
        if isinstance(exc, asyncio.TimeoutError):
            return TransientError(f"Edge TTS timeout: {exc}")
        if isinstance(exc, (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError)):
            return NetworkError(f"Edge TTS network failure: {exc}")
        if isinstance(exc, ConnectionError):
            return NetworkError(f"Edge TTS connection failure: {exc}")
        if isinstance(exc, OSError):
            return NetworkError(f"Edge TTS OS-level failure: {exc}")
        if isinstance(exc, ValueError):
            return PermanentError(f"Edge TTS invalid argument: {exc}")
        return TransientError(f"Edge TTS unclassified failure: {exc}")


__all__ = [
    "EdgeTTSClient",
]
