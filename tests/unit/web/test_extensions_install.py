"""Unit tests for ``/api/extensions/install`` return shape.

对应 spec ``fix-webui-deep-optimization`` Task 8 / SubTask 8.2：

覆盖以下场景：

* POST /api/extensions/install 未上传文件 → 400；
* POST /api/extensions/install 上传非 ``.ppcext.zip`` 文件 → 400；
* POST /api/extensions/install 上传合法后缀文件 → 200，响应 ``data`` 为
  Extension 形态对象（包含 ``name`` / ``version`` / ``enabled`` 等字段）。

测试通过 ``monkeypatch`` 替换 ``src.extensions.package.ExtensionPackageManager``，
参考 ``test_tasks_override.py`` 的 patching 模式（在函数体内做局部
``from src.extensions.package import ExtensionPackageManager``，patch 模块属性即生效）。
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Stub: ExtensionPackageManager replacement
# ---------------------------------------------------------------------------


def _make_stub_pkg_mgr_factory(install_result: dict[str, Any], info: dict[str, Any] | None) -> type:
    """构造一个 ``ExtensionPackageManager`` 替身。

    替身实现 ``install_extension`` 路由实际触碰的接口表面：

    * ``__init__()`` 无参构造（路由中调用 ``ExtensionPackageManager()``）；
    * ``install_package(zip_path, force=False)`` 返回预设的 ``install_result``；
    * ``get_package_info(name)`` 返回预设的 ``info``（用于验证 Extension 形态返回）。
    """

    class _StubPackageManager:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def install_package(self, zip_path: Any, force: bool = False) -> dict[str, Any]:
            return install_result

        def get_package_info(self, name: str) -> dict[str, Any] | None:
            return info

    return _StubPackageManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_pkg_mgr(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch ``ExtensionPackageManager`` 为返回成功安装结果的替身。"""
    install_result = {
        "success": True,
        "name": "testext",
        "version": "1.0.0",
        "error": None,
    }
    info = {
        "name": "testext",
        "version": "1.0.0",
        "description": "test extension",
        "author": "tester",
        "extension_type": "tool_integration",
        "entry": "extension.py",
        "dependencies": [],
        "enabled": True,
        "path": "/tmp/testext",
        "file_list": [],
    }
    stub_cls = _make_stub_pkg_mgr_factory(install_result, info)

    import src.extensions.package as pkg_mod

    monkeypatch.setattr(pkg_mod, "ExtensionPackageManager", stub_cls)

    return {"install_result": install_result, "info": info}


@pytest.fixture
def app_client(stub_pkg_mgr: dict[str, Any]) -> Any:
    """创建 Flask 测试客户端，patch 已生效。"""
    from src.web.app import create_app

    app = create_app("testing")
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtensionsInstall:
    """``/api/extensions/install`` 返回形状与错误路径。"""

    def test_no_file_returns_400(self, app_client: Any) -> None:
        """未上传 file 字段 → 400 BAD_REQUEST。"""
        resp = app_client.post("/api/extensions/install")
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["success"] is False
        assert body["error"]["code"] == "BAD_REQUEST"

    def test_wrong_extension_returns_400(self, app_client: Any) -> None:
        """上传非 ``.ppcext.zip`` 后缀文件 → 400 BAD_REQUEST。"""
        data = {"file": (BytesIO(b"fake"), "bad.zip")}
        resp = app_client.post(
            "/api/extensions/install",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["success"] is False
        assert body["error"]["code"] == "BAD_REQUEST"

    def test_valid_file_returns_extension_shape(
        self,
        app_client: Any,
        stub_pkg_mgr: dict[str, Any],
    ) -> None:
        """合法 ``.ppcext.zip`` 上传 → 200，``data`` 为 Extension 形态对象。"""
        data = {"file": (BytesIO(b"fake-zip-content"), "testext.ppcext.zip")}
        resp = app_client.post(
            "/api/extensions/install",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True

        data_field = body["data"]
        # 不应返回内部 install result dict（其特征是带 success/error 字段）
        assert "success" not in data_field
        # 应返回 Extension 形态对象
        expected_info = stub_pkg_mgr["info"]
        assert data_field["name"] == expected_info["name"]
        assert data_field["version"] == expected_info["version"]
        assert data_field["enabled"] is True
        assert "description" in data_field
        assert "author" in data_field
