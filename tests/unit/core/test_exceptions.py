"""Unit tests for :mod:`src.core.exceptions`.

Covers the unified exception hierarchy, the ``ErrorCodes`` enum, and the
error-code → default exit code mapping.
"""

from __future__ import annotations

import pytest

from src.core.exceptions import (
    ErrorCodes,
    PPCError,
    CLIError,
    EngineError,
    ExecutorError,
    InitializationError,
    CleanupError,
    ValidationError,
    ConfigurationError,
    PPCTimeoutError,
    FileIOError,
    TTSError,
    ChapterError,
    TTSClientError,
    TransientError,
    PermanentError,
    QuotaError,
    NetworkError,
    AudioProcessingError,
    AudioValidationError,
    TextProcessingError,
    SegmentationError,
    NormalizationError,
    ChapterRuleError,
    default_exit_code,
)


# ---------------------------------------------------------------------------
# ErrorCodes enum
# ---------------------------------------------------------------------------


class TestErrorCodes:
    def test_is_strenum(self) -> None:
        assert isinstance(ErrorCodes.EMPTY_CONTENT, str)

    def test_value_matches_name(self) -> None:
        for member in ErrorCodes:
            assert member.value == member.name

    def test_contains_core_codes(self) -> None:
        assert ErrorCodes.EMPTY_CONTENT.value == "EMPTY_CONTENT"
        assert ErrorCodes.TTS_SYNTHESIS_FAILED.value == "TTS_SYNTHESIS_FAILED"
        assert ErrorCodes.FILE_NOT_FOUND.value == "FILE_NOT_FOUND"
        assert ErrorCodes.E_INPUT_NOT_FOUND.value == "E_INPUT_NOT_FOUND"
        assert ErrorCodes.CHAPTER_SPLIT_FAILED.value == "CHAPTER_SPLIT_FAILED"

    def test_construct_from_value(self) -> None:
        assert ErrorCodes("E_CONFIG_INVALID") is ErrorCodes.E_CONFIG_INVALID


class TestDefaultExitCode:
    def test_argument_errors_map_to_2(self) -> None:
        assert default_exit_code(ErrorCodes.E_INPUT_NOT_FOUND) == 2
        assert default_exit_code(ErrorCodes.E_INPUT_EMPTY) == 2

    def test_config_errors_map_to_5(self) -> None:
        assert default_exit_code(ErrorCodes.E_CONFIG_MISSING) == 5
        assert default_exit_code(ErrorCodes.E_CONFIG_INVALID) == 5

    def test_network_errors_map_to_3(self) -> None:
        assert default_exit_code(ErrorCodes.E_NETWORK) == 3

    def test_business_errors_map_to_1(self) -> None:
        assert default_exit_code(ErrorCodes.E_BUSINESS) == 1

    def test_unmapped_codes_default_to_1(self) -> None:
        assert default_exit_code(ErrorCodes.TTS_SYNTHESIS_FAILED) == 1
        assert default_exit_code(ErrorCodes.FILE_NOT_FOUND) == 1


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    @pytest.mark.parametrize(
        "exception_type",
        [
            CLIError,
            EngineError,
            ExecutorError,
            InitializationError,
            CleanupError,
            ValidationError,
            ConfigurationError,
            PPCTimeoutError,
            FileIOError,
            TTSError,
            ChapterError,
            AudioProcessingError,
            TextProcessingError,
            TTSClientError,
            TransientError,
            PermanentError,
            QuotaError,
            NetworkError,
        ],
    )
    def test_all_derive_from_ppc_error(self, exception_type) -> None:
        assert issubclass(exception_type, PPCError)

    def test_refined_tts_taxonomy(self) -> None:
        assert issubclass(TransientError, TTSClientError)
        assert issubclass(PermanentError, TTSClientError)
        assert issubclass(QuotaError, TTSClientError)
        assert issubclass(NetworkError, TTSClientError)
        assert issubclass(TTSClientError, TTSError)

    def test_audio_validation_derives_from_audio(self) -> None:
        assert issubclass(AudioValidationError, AudioProcessingError)

    def test_text_processing_derivations(self) -> None:
        assert issubclass(SegmentationError, TextProcessingError)
        assert issubclass(NormalizationError, TextProcessingError)

    def test_chapter_rule_error_derives_from_chapter(self) -> None:
        assert issubclass(ChapterRuleError, ChapterError)


# ---------------------------------------------------------------------------
# PPCError base behavior
# ---------------------------------------------------------------------------


class TestPPCError:
    def test_message_and_error_code(self) -> None:
        err = PPCError("boom")
        assert err.message == "boom"
        assert err.error_code is None
        assert err.details == {}

    def test_error_code_and_details(self) -> None:
        err = PPCError("boom", error_code="X", details={"k": 1})
        assert err.error_code == "X"
        assert err.details == {"k": 1}
        assert "boom" in str(err)
        assert "[X]" in str(err)

    def test_empty_details_normalized_to_dict(self) -> None:
        assert PPCError("boom").details == {}
        assert PPCError("boom", details=None).details == {}

    def test_is_an_exception(self) -> None:
        with pytest.raises(PPCError):
            raise PPCError("boom")


# ---------------------------------------------------------------------------
# CLIError behavior
# ---------------------------------------------------------------------------


class TestCLIError:
    def test_inferred_exit_code_from_code(self) -> None:
        err = CLIError(ErrorCodes.E_INPUT_NOT_FOUND, "no dir")
        assert err.exit_code == 2
        assert err.code is ErrorCodes.E_INPUT_NOT_FOUND

    def test_custom_exit_code_override(self) -> None:
        err = CLIError(ErrorCodes.E_BUSINESS, "boom", exit_code=42)
        assert err.exit_code == 42

    def test_hint_and_cause(self) -> None:
        cause = ValueError("inner")
        err = CLIError(ErrorCodes.E_INPUT_EMPTY, "empty", hint="add files", cause=cause)
        assert err.hint == "add files"
        assert err.__cause__ is cause

    def test_to_dict_serialization(self) -> None:
        err = CLIError(ErrorCodes.E_CONFIG_INVALID, "bad config", hint="fix it", exit_code=5)
        data = err.to_dict()
        assert data["code"] == "E_CONFIG_INVALID"
        assert data["message"] == "bad config"
        assert data["hint"] == "fix it"
        assert data["exit_code"] == 5

    def test_str_format(self) -> None:
        err = CLIError(ErrorCodes.E_BUSINESS, "boom")
        assert str(err) == "[E_BUSINESS] boom"

    def test_inherits_ppc_error_and_sets_error_code(self) -> None:
        err = CLIError(ErrorCodes.E_NETWORK, "no net")
        assert isinstance(err, PPCError)
        assert err.error_code == "E_NETWORK"