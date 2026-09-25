"""Unit tests for ``/api/config`` default_value correctness.

覆盖 ``get_all_config`` 的 ``default_value`` 应来自 Pydantic schema 字段默认值，
而非当前值（此前两者相同，导致前端"重置为默认"不可用）。
"""

from __future__ import annotations

from src.web.api.config import _build_schema_map, _flatten_config, _schema_default


def test_schema_map_carries_pydantic_default():
    """``_build_schema_map`` 从 Pydantic 模型收集字段默认值。"""
    from src.config.schemas import PPCConfig

    schema_map = _build_schema_map(PPCConfig)
    entry = schema_map.get("core.progress_interval")
    assert entry is not None
    assert entry["default"] == 10


def test_flatten_uses_schema_default_not_current_value():
    """``default_value`` 取 schema 默认，``value`` 取当前值。"""
    schema_map = {"core.progress_interval": {"default": 10, "type": "number"}}
    items: list[dict] = []
    _flatten_config({"core": {"progress_interval": 5}}, "", items, schema_map=schema_map)

    assert len(items) == 1
    item = items[0]
    assert item["key"] == "core.progress_interval"
    assert item["value"] == "5"
    assert item["default_value"] == "10"


def test_schema_default_falls_back_to_current_value():
    """未声明 schema 默认时回退到当前值（保持兼容）。"""
    assert _schema_default({"type": "number"}, 7) == "7"
    assert _schema_default({"type": "number"}, None) == ""
