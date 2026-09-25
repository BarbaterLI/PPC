"""Unit tests for config ``set()`` type normalization.

覆盖：REST 以字符串写入强类型字段（如 ``concurrency``）时，``_config_dict``
应与 ``_config`` 一致（强转后），同时保留 schema 之外的额外顶层 key（如 UI
遗留的 ``ui:`` 节），避免丢配置。
"""

from __future__ import annotations

from pathlib import Path

from src.config.manager_access import ConfigAccessor
from src.config.manager_state import ConfigStore
from src.config.schemas import PPCConfig


def _make_store(tmp_path: Path) -> ConfigStore:
    store = ConfigStore(tmp_path / "dir", tmp_path / "config.yml", tmp_path / "ppc5.yml")
    base = PPCConfig()
    store._config = base
    store._config_dict = base.model_dump(mode="json")
    # 模拟文件里存在而 schema 已移除的遗留顶层 key
    store._config_dict["ui"] = {"theme": "dark"}
    return store


def test_set_normalizes_types_in_config_dict(tmp_path: Path) -> None:
    """字符串 ``"12"`` 写入 concurrency 后 ``_config_dict`` 存 int。"""
    store = _make_store(tmp_path)
    accessor = ConfigAccessor(store)

    ok = accessor.set("tts.concurrency", "12", persist=False)

    assert ok
    assert isinstance(store._config.tts.concurrency, int)
    assert store._config_dict["tts"]["concurrency"] == 12
    assert isinstance(store._config_dict["tts"]["concurrency"], int)


def test_set_preserves_extra_top_level_keys(tmp_path: Path) -> None:
    """schema 之外的遗留 key（``ui``）在 type 归一化后仍被保留。"""
    store = _make_store(tmp_path)
    accessor = ConfigAccessor(store)

    accessor.set("tts.concurrency", "8", persist=False)

    assert store._config_dict["ui"]["theme"] == "dark"


def test_normalize_dict_returns_json_form(tmp_path: Path) -> None:
    """``_normalize_dict`` 输出与强转后 model 的 JSON form 一致并补回额外 key。"""
    store = _make_store(tmp_path)
    raw = dict(store._config_dict)
    raw["tts"]["concurrency"] = "6"

    model = PPCConfig(**raw)
    normalized = ConfigAccessor._normalize_dict(model, raw)

    assert normalized["tts"]["concurrency"] == 6
    assert normalized["ui"]["theme"] == "dark"
