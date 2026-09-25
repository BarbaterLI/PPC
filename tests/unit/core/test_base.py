"""Unit tests for :mod:`src.core.base`.

Covers ``BaseEngine`` config assembly, lifecycle (initialize/cleanup), and
``EngineStats`` reporting via a minimal concrete subclass.
"""

from __future__ import annotations

import asyncio

import pytest

from src.core.base import BaseEngine, EngineStats

InputType = str
OutputType = str


class _DummyEngine(BaseEngine[InputType, OutputType]):
    """Minimal concrete BaseEngine used to exercise the abstract base."""

    def __init__(self, config=None):
        super().__init__(config)
        self.processed: list[str] = []

    async def initialize(self) -> None:
        await super().initialize()

    async def cleanup(self) -> None:
        await super().cleanup()

    async def process(self, input_data, /, **kwargs) -> str:
        self._check_initialized()
        self.processed.append(input_data)
        return f"ok:{input_data}"

    def ensure_ready(self) -> None:
        self._check_initialized()


class TestEngineStats:
    def test_defaults(self) -> None:
        s = EngineStats()
        assert s.started_at is None
        assert s.items_processed == 0
        assert s.duration_seconds == 0.0
        assert s.custom_metrics == {}

    def test_to_dict(self) -> None:
        s = EngineStats(items_processed=3, custom_metrics={"k": "v"})
        data = s.to_dict()
        assert data["started_at"] is None
        assert data["items_processed"] == 3
        assert data["custom_metrics"] == {"k": "v"}


class TestBaseEngineConfig:
    def test_config_defaults_to_empty_dict(self) -> None:
        assert _DummyEngine().config == {}

    def test_config_assembled_from_argument(self) -> None:
        engine = _DummyEngine({"rate": 1.0, "voice": "zh"})
        assert engine.config["rate"] == 1.0
        assert engine.config["voice"] == "zh"

    def test_get_config(self) -> None:
        engine = _DummyEngine({"rate": 1.0})
        assert engine.get_config("rate") == 1.0
        assert engine.get_config("missing") is None
        assert engine.get_config("missing", 42) == 42

    def test_set_config(self) -> None:
        engine = _DummyEngine()
        engine.set_config("key", "value")
        assert engine.config["key"] == "value"
        assert engine.get_config("key") == "value"

    def test_is_initialized_false_initially(self) -> None:
        assert _DummyEngine().is_initialized() is False


class TestBaseEngineLifecycle:
    def test_initialize_sets_stats_and_state(self) -> None:
        engine = _DummyEngine()
        asyncio.run(engine.initialize())
        assert engine.is_initialized() is True
        stats = engine.get_stats()
        assert stats["started_at"] is not None
        assert engine._initialized is True

    def test_cleanup_completes_lifecycle(self) -> None:
        engine = _DummyEngine()
        asyncio.run(engine.initialize())
        asyncio.run(engine.cleanup())
        assert engine.is_initialized() is False
        stats = engine.get_stats()
        assert stats["completed_at"] is not None
        assert stats["duration_seconds"] >= 0.0

    def test_initialize_is_idempotent(self) -> None:
        engine = _DummyEngine()
        asyncio.run(engine.initialize())
        asyncio.run(engine.initialize())
        assert engine.is_initialized() is True
        assert engine._start_time is not None

    def test_process_returns_result(self) -> None:
        engine = _DummyEngine()
        asyncio.run(engine.initialize())
        result = asyncio.run(engine.process("hello"))
        assert result == "ok:hello"
        assert engine.processed == ["hello"]

    def test_process_before_initialize_raises(self) -> None:
        engine = _DummyEngine()
        with pytest.raises(RuntimeError):
            engine.ensure_ready()

    def test_async_context_manager(self) -> None:
        engine = _DummyEngine()

        async def use_engine():
            async with engine:
                assert engine.is_initialized() is True
                return await engine.process("ctx")

        result = asyncio.run(use_engine())
        assert result == "ok:ctx"
        assert engine.is_initialized() is False


class TestBaseEngineResetStats:
    def test_reset_stats_preserves_initialization_state(self) -> None:
        engine = _DummyEngine()
        engine._stats.items_processed = 5
        asyncio.run(engine.initialize())
        engine.reset_stats()
        assert engine.is_initialized() is True
        assert engine.get_stats()["items_processed"] == 0
        # Re-initialized so started_at is repopulated
        assert engine.get_stats()["started_at"] is not None