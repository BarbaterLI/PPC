"""Unit tests for ``EdgeTTSHttpClient`` three-state backend selection.

对应 spec ``optimize-config-and-edge-selection`` Task 9 / SubTask 9.2:

覆盖 ``AntiRiskControlConfig.backend`` 字段驱动的三态后端选择逻辑
(``auto`` / ``builtin`` / ``env``)。三种状态在 ``EdgeTTSHttpClient.__init__``
中分别走不同的后端解析分支，``builtin`` 与 ``env`` 在对应后端不可用时应
抛出 ``PermanentError``，``auto`` 则允许回退。

These tests exercise the construction-time wiring only; no real Edge TTS
network call is performed.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.config.schema import AntiRiskControlConfig
from src.core.exceptions import PermanentError
from src.engines import http_client as client_mod
from src.engines.http_client import EdgeTTSHttpClient


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBackendSelection:
    """``backend`` field drives the three-state resolution in ``__init__``."""

    def test_backend_auto_uses_builtin_first(self) -> None:
        """``backend="auto"`` (default) prefers the customized builtin backend."""
        # The customized backend is importable in this environment, so the
        # default ``auto`` resolution should select it.
        config = AntiRiskControlConfig(
            backend="auto",
            shared_connector=False,
            muid_persist=False,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)

        assert client._backend_name == "src.engines.edge_tts (customized)"
        assert client._communicate_cls is client_mod._BUILTIN_COMMUNICATE

    def test_backend_builtin_forces_customized(self) -> None:
        """``backend="builtin"`` forces the customized backend even when env exists."""
        config = AntiRiskControlConfig(
            backend="builtin",
            shared_connector=False,
            muid_persist=False,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)

        assert client._backend_name == "src.engines.edge_tts (customized)"
        assert client._communicate_cls is client_mod._BUILTIN_COMMUNICATE

    def test_backend_builtin_raises_when_unavailable(self, monkeypatch) -> None:
        """``backend="builtin"`` raises ``PermanentError`` when builtin is missing."""
        # Simulate builtin backend being unavailable at runtime.
        monkeypatch.setattr(client_mod, "_BUILTIN_COMMUNICATE", None)
        monkeypatch.setattr(client_mod, "_BUILTIN_LIST_VOICES", None)
        monkeypatch.setattr(client_mod, "_BUILTIN_BACKEND_NAME", None)

        config = AntiRiskControlConfig(
            backend="builtin",
            shared_connector=False,
            muid_persist=False,
        )
        with pytest.raises(PermanentError, match="Customized edge_tts backend required but unavailable"):
            EdgeTTSHttpClient(anti_risk_config=config)

    def test_backend_env_forces_environment(self) -> None:
        """``backend="env"`` forces the environment edge-tts package."""
        config = AntiRiskControlConfig(
            backend="env",
            shared_connector=False,
            muid_persist=False,
        )
        client = EdgeTTSHttpClient(anti_risk_config=config)

        assert client._backend_name == "edge-tts (environment)"
        assert client._communicate_cls is client_mod._ENV_COMMUNICATE

    def test_backend_env_raises_when_unavailable(self, monkeypatch) -> None:
        """``backend="env"`` raises ``PermanentError`` when env package missing."""
        # Simulate the environment edge-tts package being unavailable.
        monkeypatch.setattr(client_mod, "_ENV_COMMUNICATE", None)
        monkeypatch.setattr(client_mod, "_ENV_LIST_VOICES", None)
        monkeypatch.setattr(client_mod, "_ENV_BACKEND_NAME", None)

        config = AntiRiskControlConfig(
            backend="env",
            shared_connector=False,
            muid_persist=False,
        )
        with pytest.raises(
            PermanentError,
            match="Environment edge-tts package required but not installed",
        ):
            EdgeTTSHttpClient(anti_risk_config=config)

    def test_backend_invalid_value_raises(self) -> None:
        """``AntiRiskControlConfig(backend="invalid")`` must raise ``ValueError``."""
        with pytest.raises(ValueError, match="backend"):
            AntiRiskControlConfig(backend="invalid")
