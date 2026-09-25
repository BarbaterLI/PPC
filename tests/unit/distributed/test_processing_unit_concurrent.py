"""Task 20 - concurrent convert requests must not overwrite each other's config.

Verifies that ``WorkerUnit.handle_convert_request`` creates a per-request
executor instead of mutating the shared executor's config reference.

Previously, concurrent requests would do ``self.executor.config = cfg`` on
the SAME shared executor, causing race conditions where one request would
use another request's voice/concurrency/rate settings. The fix creates a
separate ``TTSExecutor`` instance per request so there is no shared
mutation.
"""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.distributed.processing_unit import ConvertRequest, WorkerUnit

# ---------------------------------------------------------------------------
# Stub: records config voice AFTER yielding, to detect config races
# ---------------------------------------------------------------------------


class _ConcurrentStubExecutor:
    """Stub ``TTSExecutor`` that records the config voice after a yield.

    The voice is read from ``self.config.tts.voice`` **after** an
    ``await asyncio.sleep`` so that another concurrent request has the
    opportunity to overwrite the shared executor's config.  With the
    per-request executor fix, each instance has its own config and the
    recorded voices are correct.
    """

    instances: list[_ConcurrentStubExecutor] = []
    voices_seen: list[str | None] = []

    def __init__(
        self,
        config: Any,
        retry_policy: Any = None,
        *,
        tts_backend: str | None = None,
        tts_proxy: str | None = None,
    ) -> None:
        self.config = config
        self.retry_policy = retry_policy
        self.tts_backend = tts_backend
        self.tts_proxy = tts_proxy
        self._initialized = False
        _ConcurrentStubExecutor.instances.append(self)

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False

    async def add_batch_with_progress(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_handler: Any | None = None,
        voice: str | None = None,
        pattern: str = "*.txt",
        recursive: bool = False,
    ) -> Any:
        # Yield FIRST so a concurrent request can overwrite the shared
        # executor's config (if the bug is present). Then read the voice
        # from the executor's config — not the ``voice`` argument — to
        # detect the race.
        await asyncio.sleep(0.05)
        voice_from_config = self.config.tts.voice
        _ConcurrentStubExecutor.voices_seen.append(voice_from_config)

        from src.reliability import BatchResult

        return BatchResult(total=1, succeeded=1, failed=0, duration=0.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_stub():
    _ConcurrentStubExecutor.instances.clear()
    _ConcurrentStubExecutor.voices_seen.clear()
    yield
    _ConcurrentStubExecutor.instances.clear()
    _ConcurrentStubExecutor.voices_seen.clear()


def _make_config(voice: str = "base-voice") -> SimpleNamespace:
    """Build a minimal config that supports ``model_copy(deep=True)``.

    Uses ``SimpleNamespace`` instead of ``MagicMock`` because
    ``MagicMock().model_copy(deep=True)`` returns the same ``return_value``
    on every call, which would make concurrent requests share the same
    config copy and defeat the test.
    """
    cfg = SimpleNamespace(
        tts=SimpleNamespace(
            voice=voice,
            concurrency=1,
            rate="+0%",
            ramp_up_enabled=False,
            ramp_up_duration=0.0,
        ),
        reliability=SimpleNamespace(
            tts_retry=SimpleNamespace(
                max_retries=1,
                base_delay=0.01,
                max_delay=0.01,
                exponential_base=2.0,
                jitter=0.0,
            ),
        ),
    )
    cfg.model_copy = lambda deep=True: copy.deepcopy(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWorkerUnitConcurrentConfig:
    """Concurrent convert requests must not overwrite each other's config."""

    def test_concurrent_requests_use_isolated_config(self, monkeypatch, tmp_path):
        """Two concurrent requests with different voices must each see their own voice."""
        import src.executors as executors_module

        monkeypatch.setattr(executors_module, "TTSExecutor", _ConcurrentStubExecutor)

        cfg = _make_config(voice="base-voice")
        unit = WorkerUnit(host="127.0.0.1", port=0, config=cfg)
        asyncio.run(unit.start())

        # Capture the shared executor's original config reference.
        shared_original_config = unit.executor.config
        assert shared_original_config.tts.voice == "base-voice"

        in_dir = tmp_path / "input"
        in_dir.mkdir()
        (in_dir / "a.txt").write_text("hello", encoding="utf-8")

        out_a = tmp_path / "out_a"
        out_b = tmp_path / "out_b"
        out_a.mkdir()
        out_b.mkdir()

        req_a = ConvertRequest(input_dir=in_dir, output_dir=out_a, voice="voice-A")
        req_b = ConvertRequest(input_dir=in_dir, output_dir=out_b, voice="voice-B")

        async def run_concurrent():
            return await asyncio.gather(
                unit.handle_convert_request(req_a),
                unit.handle_convert_request(req_b),
            )

        results = asyncio.run(run_concurrent())

        # Both should succeed.
        assert all(r.success for r in results), [r.error for r in results]

        # Two per-request executors should have been created (one per request).
        assert len(_ConcurrentStubExecutor.instances) >= 2

        # Each request's executor should have seen its own voice — NOT the
        # other request's voice. With the old (buggy) code, both would see
        # "voice-B" because the second request overwrites the shared
        # executor's config before the first request's add_batch reads it.
        voices = sorted(_ConcurrentStubExecutor.voices_seen)
        assert voices == ["voice-A", "voice-B"], (
            f"Expected ['voice-A', 'voice-B'], got {voices} — "
            "concurrent requests overwrote each other's config"
        )

        # The shared executor's config must NOT have been mutated.
        assert unit.executor.config is shared_original_config, (
            "Shared executor's config reference was replaced"
        )
        assert unit.executor.config.tts.voice == "base-voice", (
            "Shared executor's config voice was mutated"
        )

        asyncio.run(unit.stop())

    def test_shared_executor_config_never_mutated_single_request(self, monkeypatch, tmp_path):
        """Even a single request must not mutate the shared executor's config."""
        import src.executors as executors_module

        monkeypatch.setattr(executors_module, "TTSExecutor", _ConcurrentStubExecutor)

        cfg = _make_config(voice="base-voice")
        unit = WorkerUnit(host="127.0.0.1", port=0, config=cfg)
        asyncio.run(unit.start())

        shared_config = unit.executor.config
        original_voice = shared_config.tts.voice

        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()
        out_dir.mkdir()
        (in_dir / "a.txt").write_text("hello", encoding="utf-8")

        req = ConvertRequest(input_dir=in_dir, output_dir=out_dir, voice="changed-voice")
        asyncio.run(unit.handle_convert_request(req))

        # Shared executor config must be untouched.
        assert unit.executor.config is shared_config
        assert unit.executor.config.tts.voice == original_voice

        asyncio.run(unit.stop())

    def test_per_request_executor_is_cleaned_up(self, monkeypatch, tmp_path):
        """Per-request executor must be cleaned up after the request completes."""
        import src.executors as executors_module

        monkeypatch.setattr(executors_module, "TTSExecutor", _ConcurrentStubExecutor)

        cfg = _make_config()
        unit = WorkerUnit(host="127.0.0.1", port=0, config=cfg)
        asyncio.run(unit.start())

        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()
        out_dir.mkdir()
        (in_dir / "a.txt").write_text("hello", encoding="utf-8")

        req = ConvertRequest(input_dir=in_dir, output_dir=out_dir)
        asyncio.run(unit.handle_convert_request(req))

        # The per-request executor should have been cleaned up.
        per_request_executors = _ConcurrentStubExecutor.instances[1:]  # skip the one from start()
        assert len(per_request_executors) == 1
        assert per_request_executors[0]._initialized is False, "per-request executor was not cleaned up"

        asyncio.run(unit.stop())
