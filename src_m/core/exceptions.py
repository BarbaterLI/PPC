"""Core exception definitions.

Provides a unified exception hierarchy for the PPC9 project.
All exceptions inherit from PPC9Error and include optional
error codes and structured details.
"""

from __future__ import annotations

from typing import Any


class PPC9Error(Exception):
    """Base exception for the PPC9 project."""

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


class EngineError(PPC9Error):
    """Engine-related errors."""


class ExecutorError(PPC9Error):
    """Executor-related errors."""


class InitializationError(PPC9Error):
    """Initialization failures."""


class CleanupError(PPC9Error):
    """Resource cleanup failures."""


class ValidationError(PPC9Error):
    """Data or configuration validation failures."""


class ConfigurationError(PPC9Error):
    """Configuration-related errors."""


class PPCTimeoutError(PPC9Error):
    """Operation timeout errors."""


class NetworkError(PPC9Error):
    """Network communication errors."""


class FileIOError(PPC9Error):
    """File I/O operation errors."""


class TTSError(PPC9Error):
    """TTS (Text-to-Speech) related errors."""


class ChapterError(PPC9Error):
    """Chapter processing related errors."""


