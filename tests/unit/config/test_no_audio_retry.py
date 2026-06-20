"""NoAudioRetryConfig 默认值与字段校验。"""

import pytest
from pydantic import ValidationError

from src.config.schema import NoAudioRetryConfig, ReliabilityConfig


def test_defaults():
    cfg = NoAudioRetryConfig()
    assert cfg.enabled is True
    assert cfg.max_retries == 5
    assert cfg.delay_seconds == 5.0
    assert cfg.count_in_total_retries is False


def test_reliability_default_has_tts_no_audio():
    rel = ReliabilityConfig()
    assert hasattr(rel, "tts_no_audio")
    assert rel.tts_no_audio.enabled is True
    assert rel.tts_no_audio.max_retries == 5


@pytest.mark.parametrize("bad_value", [-1, 51, 999])
def test_max_retries_bounds(bad_value):
    with pytest.raises(ValidationError):
        NoAudioRetryConfig(max_retries=bad_value)


def test_delay_seconds_non_negative():
    with pytest.raises(ValidationError):
        NoAudioRetryConfig(delay_seconds=-0.1)
