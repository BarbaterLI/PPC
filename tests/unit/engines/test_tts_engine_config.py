"""Unit tests for :class:`src.engines.tts_engine.TTSEngine` config wiring.

Covers spec ``optimize-config-and-edge-selection`` Task 2:

* ``anti_risk_config`` is forwarded to the default ``EdgeTTSHttpClient``
* ``backend_override`` / ``proxy_override`` deep-copy and patch the config
* invalid ``backend_override`` raises ``ValueError``
* an explicit ``edge_client`` takes priority and logs a WARNING when
  overrides are also supplied

These tests construct a real ``EdgeTTSHttpClient`` (no network calls are
made — only construction-time wiring is exercised) so they live in a
``_config`` suffixed file alongside ``test_tts_engine.py``.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from src.config.presets import get_preset
from src.config.schema import AntiRiskControlConfig, PPCConfig
from src.engines.client_base import EdgeTTSClient
from src.engines.http_client import EdgeTTSHttpClient
from src.engines.tts_engine import TTSEngine


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> PPCConfig:
    """Balanced preset config with a populated ``anti_risk_control`` section."""
    return get_preset("balanced")


class _StubEdgeClient(EdgeTTSClient):
    """Minimal stand-in for ``EdgeTTSClient`` used to assert injection priority.

    Does not perform any real synthesis — only exists so it can be passed as
    ``edge_client`` and later identified via identity comparison.
    """

    def __init__(self) -> None:  # noqa: D401
        # Deliberately skip ``EdgeTTSClient.__init__`` to keep the stub
        # lightweight; the abstract base only sets ``connect_timeout`` /
        # ``chunk_bytes`` which are unused here.
        self._anti_risk_config: Any = None  # marker: must remain None

    async def _synthesize(  # type: ignore[override]
        self, text: str, voice: str, *, rate: str, volume: str
    ) -> Any:
        yield  # pragma: no cover  # noqa: F841

    async def _list_voices(  # type: ignore[override]
        self, *, locale: str | None = None, gender: str | None = None
    ) -> list[Any]:
        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAntiRiskConfigWiring:
    """SubTask 2.1 + 2.2: ``anti_risk_config`` consumption & overrides."""

    def test_anti_risk_config_passed_to_client(self, config: PPCConfig) -> None:
        """Default construction forwards ``config.tts.anti_risk_control``."""
        engine = TTSEngine(config, cache=None)

        assert isinstance(engine._edge_client, EdgeTTSHttpClient)
        client_cfg: AntiRiskControlConfig = engine._edge_client._anti_risk_config
        source_cfg: AntiRiskControlConfig = config.tts.anti_risk_control

        # Field-level equality (backend / proxy / user_agents / ...)
        assert client_cfg == source_cfg
        assert client_cfg.backend == source_cfg.backend
        assert client_cfg.proxy == source_cfg.proxy
        assert client_cfg.user_agents == source_cfg.user_agents
        assert client_cfg.enabled == source_cfg.enabled
        assert client_cfg.muid_persist == source_cfg.muid_persist

    def test_backend_override_modifies_client(self, config: PPCConfig) -> None:
        """``backend_override="env"`` patches the client's config.backend."""
        engine = TTSEngine(config, cache=None, backend_override="env")

        assert isinstance(engine._edge_client, EdgeTTSHttpClient)
        assert engine._edge_client._anti_risk_config.backend == "env"

    def test_proxy_override_modifies_client(self, config: PPCConfig) -> None:
        """``proxy_override`` patches the client's config.proxy."""
        proxy_url = "http://127.0.0.1:8080"
        engine = TTSEngine(config, cache=None, proxy_override=proxy_url)

        assert isinstance(engine._edge_client, EdgeTTSHttpClient)
        assert engine._edge_client._anti_risk_config.proxy == proxy_url

    def test_invalid_backend_override_raises(self, config: PPCConfig) -> None:
        """Invalid ``backend_override`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="backend_override"):
            TTSEngine(config, cache=None, backend_override="invalid")

    def test_edge_client_param_takes_priority(
        self, config: PPCConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Explicit ``edge_client`` wins; overrides are ignored with a WARNING."""
        stub = _StubEdgeClient()

        with caplog.at_level(logging.WARNING, logger="src.engines.tts_engine"):
            engine = TTSEngine(
                config,
                edge_client=stub,
                cache=None,
                backend_override="env",
                proxy_override="http://127.0.0.1:8080",
            )

        # The stub is used verbatim — no new client constructed.
        assert engine._edge_client is stub
        # The stub's anti_risk_config marker must remain untouched.
        assert stub._anti_risk_config is None

        # A WARNING was emitted mentioning the ignored overrides.
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("忽略" in m or "ignored" in m for m in messages), (
            f"expected a WARNING about ignored overrides, got: {messages!r}"
        )


class TestOverrideDoesNotMutateOriginal:
    """Ensures ``backend_override`` / ``proxy_override`` deep-copy the config."""

    def test_backend_override_does_not_mutate_source(self, config: PPCConfig) -> None:
        original_backend = config.tts.anti_risk_control.backend
        TTSEngine(config, cache=None, backend_override="env")

        # The source config must be untouched.
        assert config.tts.anti_risk_control.backend == original_backend

    def test_proxy_override_does_not_mutate_source(self, config: PPCConfig) -> None:
        original_proxy = config.tts.anti_risk_control.proxy
        TTSEngine(config, cache=None, proxy_override="http://127.0.0.1:8080")

        assert config.tts.anti_risk_control.proxy == original_proxy
