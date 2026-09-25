"""Unit tests for ``AntiRiskControlConfig`` legacy ``prefer_builtin`` migration.

对应 spec ``optimize-config-and-edge-selection`` Task 9 / SubTask 9.4:

覆盖 ``AntiRiskControlConfig._migrate_prefer_builtin`` model_validator 的
行为：

* ``prefer_builtin=True``  →  ``backend="env"``  + DeprecationWarning
* ``prefer_builtin=False`` →  ``backend="auto"`` + DeprecationWarning
* 显式 ``backend`` 同时传入 →  以 ``backend`` 为准但仍发出 DeprecationWarning
* 未传 ``prefer_builtin``  →  无 DeprecationWarning

The validator runs in ``mode="before"`` and accepts a dict, so we feed
the raw dict to ``AntiRiskControlConfig.model_validate`` to exercise the
migration path.
"""

from __future__ import annotations

import asyncio
import warnings

import pytest

from src.config.schema import AntiRiskControlConfig


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPreferBuiltinMigration:
    """``prefer_builtin`` is migrated to ``backend`` with DeprecationWarning."""

    def test_prefer_builtin_true_migrates_to_env(self) -> None:
        """``prefer_builtin=True`` migrates to ``backend="env"``."""
        with pytest.warns(DeprecationWarning, match="prefer_builtin"):
            config = AntiRiskControlConfig.model_validate({"prefer_builtin": True})
        assert config.backend == "env"

    def test_prefer_builtin_false_migrates_to_auto(self) -> None:
        """``prefer_builtin=False`` migrates to ``backend="auto"``."""
        with pytest.warns(DeprecationWarning, match="prefer_builtin"):
            config = AntiRiskControlConfig.model_validate({"prefer_builtin": False})
        assert config.backend == "auto"

    def test_explicit_backend_overrides_prefer_builtin(self) -> None:
        """``backend`` wins over ``prefer_builtin`` but warning still emitted."""
        with pytest.warns(DeprecationWarning, match="prefer_builtin"):
            config = AntiRiskControlConfig.model_validate(
                {"prefer_builtin": True, "backend": "auto"}
            )
        # ``backend`` takes priority — the migration must not overwrite it.
        assert config.backend == "auto"

    def test_no_prefer_builtin_no_warning(self) -> None:
        """Without ``prefer_builtin`` no DeprecationWarning is emitted."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # Should NOT raise — the migration path is skipped entirely.
            config = AntiRiskControlConfig.model_validate({"backend": "auto"})
        assert config.backend == "auto"
