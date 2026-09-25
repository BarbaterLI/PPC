"""Core exception definitions.

Provides a unified exception hierarchy for the PPC_NEXT project.
All exceptions inherit from PPCError and include optional
error codes and structured details.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCodes(StrEnum):
    """Standardized error codes for PPC_NEXT."""

    EMPTY_CONTENT = "EMPTY_CONTENT"
    TTS_SYNTHESIS_FAILED = "TTS_SYNTHESIS_FAILED"
    TTS_NO_AUDIO_RECEIVED = "TTS_NO_AUDIO_RECEIVED"
    TTS_SEGMENTATION_FAILED = "TTS_SEGMENTATION_FAILED"
    TTS_TRANSIENT_FAILED = "TTS_TRANSIENT_FAILED"
    TTS_PERMANENT_FAILED = "TTS_PERMANENT_FAILED"
    TTS_QUOTA_EXCEEDED = "TTS_QUOTA_EXCEEDED"
    TTS_NETWORK_FAILED = "TTS_NETWORK_FAILED"
    TTS_CIRCUIT_OPEN = "TTS_CIRCUIT_OPEN"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_PERMISSION_DENIED = "FILE_PERMISSION_DENIED"
    NO_CHAPTERS = "NO_CHAPTERS"
    CHAPTER_SPLIT_FAILED = "CHAPTER_SPLIT_FAILED"
    CHAPTER_RULE_INVALID = "CHAPTER_RULE_INVALID"
    BATCH_PROCESSING_FAILED = "BATCH_PROCESSING_FAILED"
    INVALID_EPUB = "INVALID_EPUB"
    EPUB_EXTRACTION_FAILED = "EPUB_EXTRACTION_FAILED"
    AUDIO_PROCESSING_FAILED = "AUDIO_PROCESSING_FAILED"
    AUDIO_VALIDATION_FAILED = "AUDIO_VALIDATION_FAILED"
    TEXT_NORMALIZATION_FAILED = "TEXT_NORMALIZATION_FAILED"
    E_INPUT_NOT_FOUND = "E_INPUT_NOT_FOUND"
    E_INPUT_EMPTY = "E_INPUT_EMPTY"
    E_CONFIG_MISSING = "E_CONFIG_MISSING"
    E_CONFIG_INVALID = "E_CONFIG_INVALID"
    E_NETWORK = "E_NETWORK"
    E_BUSINESS = "E_BUSINESS"
    DIR_NOT_FOUND = "DIR_NOT_FOUND"
    NO_FILES = "NO_FILES"
    BATCH_FAILED = "BATCH_FAILED"
    ARCHIVE_FAILED = "ARCHIVE_FAILED"
    NOT_VOLUME_STRUCTURE = "NOT_VOLUME_STRUCTURE"
    GROUP_BY_VOLUME_FAILED = "GROUP_BY_VOLUME_FAILED"


class PPCError(Exception):
    """Base exception for the PPC_NEXT project."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.error_code:
            parts.append(f"[{self.error_code}]")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " ".join(parts)


_DEFAULT_EXIT_CODES: dict[ErrorCodes, int] = {
    ErrorCodes.E_INPUT_NOT_FOUND: 2,
    ErrorCodes.E_INPUT_EMPTY: 2,
    ErrorCodes.E_CONFIG_MISSING: 5,
    ErrorCodes.E_CONFIG_INVALID: 5,
    ErrorCodes.E_NETWORK: 3,
    ErrorCodes.E_BUSINESS: 1,
}


def default_exit_code(code: ErrorCodes) -> int:
    """根据 :class:`ErrorCodes` 返回默认退出码。

    退出码规约(详见 ``docs/exit-codes.md``):

    =====  =================================
    Code  含义
    =====  =================================
    0     成功 / 正常工作(即使"无操作")
    1     业务错误(TTS 调用失败、文件 IO)
    2     参数错误(输入目录不存在等)
    3     网络 / 外部依赖错误(edge-tts)
    4     权限错误(无法写输出目录等)
    5     配置错误(配置缺失/无法解析)
    =====  =================================
    """
    return _DEFAULT_EXIT_CODES.get(code, 1)


class CLIError(PPCError):
    """统一的 CLI 业务异常。

    Parameters
    ----------
    code : ErrorCodes
        错误代码。
    message : str
        给用户看的错误描述(单行,无换行)。
    hint : Optional[str]
        给用户的下一步操作建议(可省略)。
    exit_code : Optional[int]
        自定义退出码;省略时按 ``code`` 取默认值。
    cause : Optional[BaseException]
        原始异常(可选,仅在 ``--verbose`` 模式下渲染 traceback)。
    """

    def __init__(
        self,
        code: ErrorCodes,
        message: str,
        hint: str | None = None,
        exit_code: int | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.code: ErrorCodes = code
        self.hint: str | None = hint
        self.exit_code: int = int(exit_code) if exit_code is not None else default_exit_code(code)
        if cause is not None:
            self.__cause__ = cause
        super().__init__(message, error_code=code.value)

    def to_dict(self) -> dict:
        """序列化为 dict(用于 JSON 输出)。"""
        return {
            "code": self.code.value,
            "message": self.message,
            "hint": self.hint,
            "exit_code": self.exit_code,
        }

    def __str__(self) -> str:  # noqa: D401
        return f"[{self.code.value}] {self.message}"


class EngineError(PPCError):
    """Engine-related errors."""


class ExecutorError(PPCError):
    """Executor-related errors."""


class InitializationError(PPCError):
    """Initialization failures."""


class CleanupError(PPCError):
    """Resource cleanup failures."""


class ValidationError(PPCError):
    """Data or configuration validation failures."""


class ConfigurationError(PPCError):
    """Configuration-related errors."""


class PPCTimeoutError(PPCError):
    """Operation timeout errors."""


class FileIOError(PPCError):
    """File I/O operation errors."""


class TTSError(PPCError):
    """TTS (Text-to-Speech) related errors."""


class ChapterError(PPCError):
    """Chapter processing related errors."""


# ---------------------------------------------------------------------------
# Phase 1 — TTS / Edge TTS refined exception taxonomy
# ---------------------------------------------------------------------------


class TTSClientError(TTSError):
    """Base error for TTS client-side failures.

    All errors raised by :class:`src.engines.edge_tts_client.EdgeTTSClient`
    inherit from this class so callers can catch a single type and dispatch
    on the refined subclasses below.
    """


class TransientError(TTSClientError):
    """Recoverable failures (5xx, timeouts, connection resets).

    The engine SHOULD retry these via the existing retry policy.
    """


class PermanentError(TTSClientError):
    """Non-recoverable failures (invalid input, malformed response).

    Retrying will not help; the engine SHOULD surface the error to the
    caller after a single attempt.
    """


class QuotaError(TTSClientError):
    """Quota / rate-limit errors (HTTP 403/429 from Edge TTS).

    The engine SHOULD trip the circuit breaker and back off.
    """


class NetworkError(TTSClientError):
    """Network-level failures (DNS, TCP, TLS, unreachable host).

    Distinguished from :class:`TransientError` for finer-grained metrics.
    """


# ---------------------------------------------------------------------------
# Phase 1 — Audio / Text processing errors
# ---------------------------------------------------------------------------


class AudioProcessingError(PPCError):
    """Audio processing failures (load, merge, normalize, fingerprint)."""


class AudioValidationError(AudioProcessingError):
    """Audio file is present but invalid (empty, zero-duration, corrupt)."""


class TextProcessingError(PPCError):
    """Text processing failures (segmentation, normalization)."""


class SegmentationError(TextProcessingError):
    """Segmentation strategy could not produce valid output."""


class NormalizationError(TextProcessingError):
    """Normalization rule could not be applied to the input."""


class ChapterRuleError(ChapterError):
    """Chapter rule file (YAML) could not be loaded or parsed."""
