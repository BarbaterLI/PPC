"""Unit tests for optional API Token auth middleware.

覆盖 ``create_api_auth_middleware`` 在配置/未配置 token 时的行为，以及
token 可通过 Bearer 头 / X-Auth-Token 头 / access_token 查询参数三种方式提供。
"""

from __future__ import annotations

import pytest
from flask import Flask


@pytest.fixture
def app():
    app = Flask(__name__)

    from src.web.auth import create_api_auth_middleware

    app.before_request(create_api_auth_middleware())

    @app.route("/api/ping", methods=["GET"])
    def ping():
        return {"pong": True}

    @app.route("/public", methods=["GET"])
    def public():
        return {"ok": True}

    app.config["TESTING"] = True
    return app


def _set_token(monkeypatch: pytest.MonkeyPatch, token: str | None) -> None:
    from src.web.auth import API_TOKEN_ENV

    if token is None:
        monkeypatch.delenv(API_TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(API_TOKEN_ENV, token)


def test_unprotected_path_passes(app, monkeypatch) -> None:
    """未配置 token 或非受保护前缀都放行。"""
    _set_token(monkeypatch, "secret")
    resp = app.test_client().get("/public")
    assert resp.status_code == 200


def test_no_token_configured_passes(app, monkeypatch) -> None:
    """未配置 token 时鉴权关闭，请求直接放行。"""
    _set_token(monkeypatch, None)
    resp = app.test_client().get("/api/ping")
    assert resp.status_code == 200


def test_missing_token_returns_401(app, monkeypatch) -> None:
    """配置 token 后无凭证请求返回 401。"""
    _set_token(monkeypatch, "secret")
    resp = app.test_client().get("/api/ping")
    assert resp.status_code == 401


def test_wrong_token_returns_401(app, monkeypatch) -> None:
    _set_token(monkeypatch, "secret")
    resp = app.test_client().get("/api/ping", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


@pytest.mark.parametrize(
    "kwargs",
    [
        {"headers": {"Authorization": "Bearer secret"}},
        {"headers": {"X-Auth-Token": "secret"}},
        {"query_string": {"access_token": "secret"}},
    ],
)
def test_token_any_source_passes(app, monkeypatch, kwargs) -> None:
    """三种 token 提供方式都通过鉴权。"""
    _set_token(monkeypatch, "secret")
    resp = app.test_client().get("/api/ping", **kwargs)
    assert resp.status_code == 200
