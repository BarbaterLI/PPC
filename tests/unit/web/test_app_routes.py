"""Unit tests for SPA fallback not swallowing API 404s.

对应 spec ``fix-webui-deep-optimization`` Task 12 / SubTask 12.2：

覆盖以下场景：

* GET /api/nonexistent → 404，响应 JSON 为 ``{success: false, error.code: "NOT_FOUND"}``；
* GET /api/foo/bar/baz → 404（嵌套路径同样不被 SPA fallback 吞没）；
* GET / （根） → 200（serve index.html）；
* GET /some/random/path → 200（非 api 路径仍走 SPA fallback）。

``serve_frontend`` 路由仅在 ``static_folder`` 存在时注册，因此测试通过
``monkeypatch.setenv("FLASK_STATIC_FOLDER", ...)`` 指向一个临时目录（含
``index.html``）来确保路由注册。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def app_client_with_dist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """创建带静态目录的 Flask 测试客户端。

    在 ``tmp_path`` 下构造 ``dist/index.html``，并通过环境变量
    ``FLASK_STATIC_FOLDER`` 传给 ``create_app``，确保 ``serve_frontend`` 路由注册。
    """
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "index.html").write_text("<!DOCTYPE html><html></html>", encoding="utf-8")

    monkeypatch.setenv("FLASK_STATIC_FOLDER", str(dist_dir))

    from src.web.app import create_app

    app = create_app("testing")
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAppRoutesApi404:
    """未知 ``/api/*`` 路径返回 404 JSON，不被 SPA fallback 吞没。"""

    def test_unknown_api_returns_404_json(self, app_client_with_dist: Any) -> None:
        """GET /api/nonexistent → 404 NOT_FOUND JSON。"""
        resp = app_client_with_dist.get("/api/nonexistent")
        assert resp.status_code == 404
        body = resp.get_json()
        assert body["success"] is False
        assert body["error"]["code"] == "NOT_FOUND"

    def test_nested_api_path_returns_404(self, app_client_with_dist: Any) -> None:
        """GET /api/foo/bar/baz → 404（嵌套路径同样命中 api 判断）。"""
        resp = app_client_with_dist.get("/api/foo/bar/baz")
        assert resp.status_code == 404
        body = resp.get_json()
        assert body["success"] is False
        assert body["error"]["code"] == "NOT_FOUND"


class TestAppRoutesSpaFallback:
    """非 api 路径仍走 SPA fallback，返回 index.html。"""

    def test_root_serves_index_html(self, app_client_with_dist: Any) -> None:
        """GET / → 200，serve index.html。"""
        resp = app_client_with_dist.get("/")
        assert resp.status_code == 200
        assert b"<html>" in resp.data

    def test_spa_fallback_for_non_api_path(self, app_client_with_dist: Any) -> None:
        """GET /some/random/path → 200（非 api 路径走 SPA fallback）。"""
        resp = app_client_with_dist.get("/some/random/path")
        assert resp.status_code == 200
        assert b"<html>" in resp.data
