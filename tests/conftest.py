"""Shared pytest fixtures and configuration for the PPC10 test-suite.

This conftest is loaded automatically by ``pytest``. It does two things:

* it ensures the local ``venv_local`` directory (used during development)
  is on ``sys.path`` so that ``pytest`` resolves ``src_m`` correctly
  when run from the repo root without an editable install;
* it exposes a few domain-level fixtures shared by the Phase 1 unit
  tests (mock Edge TTS factories, in-memory caches, fake audio paths).
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VENV_LOCAL = _REPO_ROOT / "venv_local"
if _VENV_LOCAL.exists() and str(_VENV_LOCAL) not in sys.path:
    sys.path.insert(0, str(_VENV_LOCAL))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Edge TTS mock factory
# ---------------------------------------------------------------------------


class _FakeCommunicate:
    """Minimal stand-in for ``edge_tts.Communicate``.

    Yields ``{"type": "audio", "data": <bytes>}`` for each frame in
    ``chunks`` plus a closing ``turn.end`` event.
    """

    def __init__(
        self,
        text: str,
        voice: str,
        rate: str = "+0%",
        volume: str = "+0%",
        chunks: List[bytes] | None = None,
        fail_with: Exception | None = None,
    ) -> None:
        self.text = text
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.chunks = chunks or [b"\x00" * 64]
        self.fail_with = fail_with
        self.saved: List[bytes] = []

    async def stream(self):  # noqa: D401
        if self.fail_with is not None:
            raise self.fail_with
        for chunk in self.chunks:
            yield {"type": "audio", "data": chunk}
        yield {"type": "WordBoundary", "offset": 0, "duration": 0}
        yield {"type": "turn.end"}

    async def save(self, path: str) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        with open(path, "wb") as fh:
            for chunk in self.chunks:
                fh.write(chunk)
                self.saved.append(chunk)


@pytest.fixture
def fake_communicate_factory() -> Any:
    """Return a factory that builds ``_FakeCommunicate`` instances.

    Tests can inject a custom list of ``chunks`` or a ``fail_with``
    exception by populating ``factory.kwargs`` before each call.
    """
    state: Dict[str, Any] = {
        "chunks": [b"FAKE-MP3-CHUNK-1", b"FAKE-MP3-CHUNK-2"],
        "fail_with": None,
    }

    def _factory(text: str, voice: str, rate: str, volume: str) -> _FakeCommunicate:
        return _FakeCommunicate(
            text=text,
            voice=voice,
            rate=rate,
            volume=volume,
            chunks=list(state["chunks"]),
            fail_with=state["fail_with"],
        )

    _factory.state = state  # type: ignore[attr-defined]
    return _factory


@pytest.fixture
def sample_voice_list() -> List[Dict[str, str]]:
    return [
        {
            "Name": "Microsoft Server Speech Text to Speech Voice (zh-CN, XiaoxiaoNeural)",
            "ShortName": "zh-CN-XiaoxiaoNeural",
            "Gender": "Female",
            "Locale": "zh-CN",
            "FriendlyName": "Microsoft Xiaoxiao - Chinese (Mainland)",
        },
        {
            "Name": "Microsoft Server Speech Text to Speech Voice (en-US, JennyNeural)",
            "ShortName": "en-US-JennyNeural",
            "Gender": "Female",
            "Locale": "en-US",
            "FriendlyName": "Microsoft Jenny - English (United States)",
        },
    ]


@pytest.fixture
def tmp_audio_dir(tmp_path: Path) -> Path:
    """Provide a fresh temporary directory for audio output."""
    out = tmp_path / "audio"
    out.mkdir(parents=True, exist_ok=True)
    return out
