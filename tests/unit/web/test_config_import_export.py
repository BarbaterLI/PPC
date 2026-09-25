"""Unit tests for config import/export endpoints.

对应 spec ``fix-webui-deep-optimization`` Task 4 / SubTask 4.4：

覆盖 ``src/web/api/config.py`` 中：

* ``POST /api/config/import`` 支持 JSON body（``ConfigExportData`` 形态 / 原始
  config dict 形态），并保留 multipart 文件上传向后兼容；
* 无文件且无 JSON body → 400；
* ``GET /api/config/export/file`` 返回 YAML 文件 Blob（带 signature 注释行）。

``ConfigManager`` 与 ``_compute_config_signature`` 通过 ``monkeypatch.setattr``
打桩，避免触碰真实文件系统与 HMAC 密钥。
"""

from __future__ import annotations

import io
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_config_manager(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """替换 ``ConfigManager`` 与 ``_compute_config_signature`` 为桩。

    ``import_config`` / ``export_config_file`` 在函数体内部做局部 import：
    ``from src.config.manager_core import ConfigManager`` 与
    ``from src.config.manager_core import _compute_config_signature``，因此
    patch 模块属性即可生效。
    """
    fixed_config: dict[str, Any] = {
        "tts": {"voice": "default-voice"},
        "system": {"theme": "dark"},
    }
    captured: dict[str, Any] = {
        "imported_paths": [],
        "import_calls": 0,
        "fixed_config": fixed_config,
    }

    class _StubConfigManager:
        def get_all(self) -> dict[str, Any]:
            return fixed_config

        def import_config(self, import_path: Any, merge: bool = True) -> bool:
            captured["import_calls"] += 1
            captured["imported_paths"].append(str(import_path))
            return True

    import src.config.manager_core as manager_mod
    import src.config.manager_core as core_mod

    monkeypatch.setattr(manager_mod, "ConfigManager", _StubConfigManager)

    def _fake_signature(content: str) -> str:
        return "fake-signature-abcdef"

    monkeypatch.setattr(core_mod, "_compute_config_signature", _fake_signature)

    return captured


@pytest.fixture
def client(patched_config_manager: dict[str, Any]) -> Any:
    """Flask 测试客户端（依赖 patch 后的 ``ConfigManager``）。"""
    from src.web.app import create_app

    app = create_app("testing")
    return app.test_client()


# ---------------------------------------------------------------------------
# Tests: JSON import path
# ---------------------------------------------------------------------------


class TestImportConfigJsonBody:
    """``POST /api/config/import`` 接受 JSON body。"""

    def test_import_with_config_wrapper(
        self, client: Any, patched_config_manager: dict[str, Any]
    ) -> None:
        """JSON body ``{"config": {...}}`` (ConfigExportData 形态) → 200。"""
        resp = client.post(
            "/api/config/import",
            json={"config": {"tts": {"voice": "test"}}},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        assert body["data"]["imported"] is True
        assert patched_config_manager["import_calls"] == 1

    def test_import_with_raw_config_dict(
        self, client: Any, patched_config_manager: dict[str, Any]
    ) -> None:
        """JSON body 是原始 config dict → 200。"""
        resp = client.post(
            "/api/config/import",
            json={"tts": {"voice": "test2"}},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        assert body["data"]["imported"] is True
        assert patched_config_manager["import_calls"] == 1

    def test_import_with_no_file_no_json_returns_400(self, client: Any) -> None:
        """无文件且无 JSON body → 400 BAD_REQUEST。"""
        resp = client.post("/api/config/import")
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["success"] is False
        assert body["error"]["code"] == "BAD_REQUEST"


# ---------------------------------------------------------------------------
# Tests: multipart import path (backward compat)
# ---------------------------------------------------------------------------


class TestImportConfigMultipart:
    """``POST /api/config/import`` multipart 文件上传向后兼容。"""

    def test_import_multipart_yaml_file(
        self, client: Any, patched_config_manager: dict[str, Any]
    ) -> None:
        """multipart 上传 YAML 文件 → 200, imported=True。"""
        yaml_bytes = b"tts:\n  voice: test-mp\n"
        resp = client.post(
            "/api/config/import",
            data={"file": (io.BytesIO(yaml_bytes), "config.yml")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        assert body["data"]["imported"] is True
        assert patched_config_manager["import_calls"] == 1


# ---------------------------------------------------------------------------
# Tests: file export
# ---------------------------------------------------------------------------


class TestExportConfigFile:
    """``GET /api/config/export/file`` 返回 YAML 文件 Blob。"""

    def test_export_file_returns_yaml_blob(
        self, client: Any, patched_config_manager: dict[str, Any]
    ) -> None:
        """GET /api/config/export/file → 200, Content-Type=yaml, attachment。"""
        resp = client.get("/api/config/export/file")
        assert resp.status_code == 200
        assert "yaml" in resp.content_type.lower()
        content_disposition = resp.headers.get("Content-Disposition", "")
        assert "attachment" in content_disposition.lower()
        assert "config.yml" in content_disposition

    def test_export_file_body_contains_config_keys(
        self, client: Any, patched_config_manager: dict[str, Any]
    ) -> None:
        """返回 body 非空且包含 config 顶层键与 signature 行。"""
        resp = client.get("/api/config/export/file")
        assert resp.status_code == 200
        body_text = resp.data.decode("utf-8")
        assert len(body_text) > 0
        assert "tts" in body_text
        assert "system" in body_text
        # signature marker appended as a comment line
        assert "__ppc_signature__" in body_text
        assert "fake-signature-abcdef" in body_text
