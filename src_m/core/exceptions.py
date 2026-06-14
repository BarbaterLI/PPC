"""Core exception definitions.

Provides a unified exception hierarchy for the PPC10 project.
All exceptions inherit from PPC10Error and include optional
error codes and structured details.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCodes(StrEnum):
    """Standardized error codes for PPC10."""

    EMPTY_CONTENT = "EMPTY_CONTENT"
    TTS_SYNTHESIS_FAILED = "TTS_SYNTHESIS_FAILED"
    TTS_NO_AUDIO_RECEIVED = "TTS_NO_AUDIO_RECEIVED"
    TTS_SEGMENTATION_FAILED = "TTS_SEGMENTATION_FAILED"
    TTS_TRANSIENT_FAILED = "TTS_TRANSIENT_FAILED"
    TTS_PERMANENT_FAILED = "TTS_PERMANENT_FAILED"
    TTS_QUOTA_EXCEEDED = "TTS_QUOTA_EXCEEDED"
    TTS_NETWORK_FAILED = "TTS_NETWORK_FAILED"
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


class PPC10Error(Exception):
    """Base exception for the PPC10 project."""

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


class EngineError(PPC10Error):
    """Engine-related errors."""


class ExecutorError(PPC10Error):
    """Executor-related errors."""


class InitializationError(PPC10Error):
    """Initialization failures."""


class CleanupError(PPC10Error):
    """Resource cleanup failures."""


class ValidationError(PPC10Error):
    """Data or configuration validation failures."""


class ConfigurationError(PPC10Error):
    """Configuration-related errors."""


class PPCTimeoutError(PPC10Error):
    """Operation timeout errors."""


class NetworkError(PPC10Error):
    """Network communication errors."""


class FileIOError(PPC10Error):
    """File I/O operation errors."""


class TTSError(PPC10Error):
    """TTS (Text-to-Speech) related errors."""


class ChapterError(PPC10Error):
    """Chapter processing related errors."""


# ---------------------------------------------------------------------------
# Phase 1 — TTS / Edge TTS refined exception taxonomy
# ---------------------------------------------------------------------------


class TTSClientError(TTSError):
    """Base error for TTS client-side failures.

    All errors raised by :class:`src_m.engines.edge_tts_client.EdgeTTSClient`
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


class AudioProcessingError(PPC10Error):
    """Audio processing failures (load, merge, normalize, fingerprint)."""


class AudioValidationError(AudioProcessingError):
    """Audio file is present but invalid (empty, zero-duration, corrupt)."""


class TextProcessingError(PPC10Error):
    """Text processing failures (segmentation, normalization)."""


class SegmentationError(TextProcessingError):
    """Segmentation strategy could not produce valid output."""


class NormalizationError(TextProcessingError):
    """Normalization rule could not be applied to the input."""


class ChapterRuleError(ChapterError):
    """Chapter rule file (YAML) could not be loaded or parsed."""


