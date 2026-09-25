"""Unit tests for fanqie config update endpoint contract.

对应 spec ``fix-webui-deep-optimization`` Task 3 / SubTask 3.3：

覆盖 ``PUT /api/fanqie/config`` 端点：

* 支持整体配置对象形态（``Partial<FanqieConfig>``）；
* 兼容旧 ``{"key": ..., "value": ...}`` 形态；
* 空请求体返回 400。

``set_config_value`` 被 mock 为始终返回 True，避免触碰真实文件系统。
"""

from __future__ import annotations

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_set_config_value(monkeypatch: pytest.MonkeyPatch) -> Any:
    """替换动态加载的 ``downloader_core.set_config_value`` 为始终返回 True 的桩。

    端点通过扩展加载器取得 ``ppc_ext_extensions_fanqie`` 模块，
    因此这里也从扩展加载器解析同一模块对象后再打桩。
    """
    from importlib import import_module

    from src.web.api.fanqie import _get_fanqie_module

    module = _get_fanqie_module()
    core_mod = import_module(f"{module.__name__}.downloader_core")

    def _fake_set(key: str, value: str) -> bool:
        return True

    monkeypatch.setattr(core_mod, "set_config_value", _fake_set)
    return _fake_set


@pytest.fixture
def client(patched_set_config_value: Any):
    """Flask 测试客户端。"""
    from src.web.app import create_app

    app = create_app("testing")
    return app.test_client()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFanqieConfigUpdate:
    """``PUT /api/fanqie/config`` 契约测试。"""

    def test_update_with_config_object(self, client: Any) -> None:
        """整体配置对象 → 200, updated=True, keys 包含所有顶层键。"""
        resp = client.put(
            "/api/fanqie/config",
            json={"use_mirror": True, "mirror_host": "gh.test.com"},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        data = body["data"]
        assert data["updated"] is True
        assert set(data["keys"]) == {"use_mirror", "mirror_host"}

    def test_update_with_old_key_value_shape(self, client: Any) -> None:
        """旧形态 ``{"key": ..., "value": ...}`` → 200, updated=True。"""
        resp = client.put(
            "/api/fanqie/config",
            json={"key": "use_mirror", "value": "false"},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        data = body["data"]
        assert data["updated"] is True
        assert data["keys"] == ["use_mirror"]

    def test_update_with_empty_body_returns_400(self, client: Any) -> None:
        """空请求体 → 400。"""
        resp = client.put("/api/fanqie/config")
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["success"] is False
        assert body["error"]["code"] == "BAD_REQUEST"
