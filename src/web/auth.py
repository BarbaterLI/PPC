"""可选的 Bearer Token 鉴权助手。

为 WebUI 的 ``/api/*`` 与分布式节点的 ``/api/v1/*`` 提供统一的可选鉴权。
未配置 token（``PPC_API_TOKEN`` 环境变量）时自动放行，保持向后兼容；
配置后所有相关端点均需携带 ``Authorization: Bearer <token>``。

两个整合阵营（Flask :mod:`src.web.app` 与 aiohttp :mod:`src.distributed.node_server`）
复用此模块，避免鉴权逻辑分叉。
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Callable

# 环境变量名：配置后对 WebUI / 分布式 API 启用 Bearer Token 鉴权。
API_TOKEN_ENV = "PPC_API_TOKEN"

# 每次取 token 时从环境变量读取，支持运行时切换（测试、运维）。
def get_api_token() -> str | None:
    """返回配置的 API token；未配置返回 ``None``（鉴权关闭）。"""
    token = os.environ.get(API_TOKEN_ENV)
    return token.strip() if token else None


def token_configured() -> bool:
    """是否已配置 API token。"""
    return get_api_token() is not None


def is_valid_token(provided: str | None) -> bool:
    """恒定时间比较，避免时序侧信道。

    Args:
        provided: 请求携带的 token；为 ``None`` 或空串时直接返回 ``False``。

    Returns:
        token 已配置且与请求 token 一致时返回 ``True``。
    """
    expected = get_api_token()
    if not expected or not provided:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), provided.encode("utf-8"))


def extract_bearer(authorization: str | None) -> str | None:
    """从 ``Authorization`` 头原始值中解析 Bearer token。

    Args:
        authorization: ``Authorization`` 头内容，如 ``"Bearer abc123"``。

    Returns:
        解析出的 token；非 ``Bearer`` 方案或缺失时返回 ``None``。
    """
    if not authorization:
        return None
    scheme, _, rest = authorization.partition(" ")
    if scheme.strip().lower() != "bearer" or not rest:
        return None
    return rest.strip()


def check_bearer(authorization: str | None) -> bool:
    """校验 ``Authorization`` 头是否携带有效 token。

    未配置 token 时恒返回 ``True``（鉴权关闭）；配置后要求有效 Bearer token。
    """
    if not token_configured():
        return True
    return is_valid_token(extract_bearer(authorization))


def is_loopback_host(host: str | None) -> bool:
    """判断 host 是否为回环地址（localhost / 127.x / ::1）。"""
    if not host:
        return False
    lowered = host.strip().lower()
    if lowered in ("localhost", "::1"):
        return True
    return lowered.startswith("127.")


def get_header_value(headers_get: Callable[[str], str | None], name: str) -> str | None:
    """从头部访问可调用对象（Flask 的 ``request.headers.get`` / aiohttp 的字典）取头值。

    Args:
        headers_get: 接受头部名并返回值的可调用对象。
        name: 头部名，如 ``"Authorization"``。

    Returns:
        头值；不存在返回 ``None``。
    """
    try:
        return headers_get(name)
    except Exception:  # noqa: BLE001 - 统一视为缺失
        return None


def auth_headers() -> dict[str, str]:
    """返回请求头字典；已配置 token 时附带 ``Authorization`` 头供客户端使用。

    用于分布式主控→节点等客户端侧请求自动携带 token。
    """
    token = get_api_token()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


class AuthService:
    """Bearer Token 鉴权服务（无状态，每次调用时读取环境变量）。"""

    def __init__(self, env_var: str = API_TOKEN_ENV) -> None:
        self._env_var = env_var

    def get_token(self) -> str | None:
        """返回配置的 API token；未配置返回 ``None``（鉴权关闭）。"""
        token = os.environ.get(self._env_var)
        return token.strip() if token else None

    def is_configured(self) -> bool:
        """是否已配置 API token。"""
        return self.get_token() is not None

    def is_valid(self, provided: str | None) -> bool:
        """恒定时间比较，避免时序侧信道。"""
        expected = self.get_token()
        if not expected or not provided:
            return False
        return hmac.compare_digest(expected.encode("utf-8"), provided.encode("utf-8"))

    def extract_bearer(self, authorization: str | None) -> str | None:
        """从 ``Authorization`` 头原始值中解析 Bearer token。"""
        if not authorization:
            return None
        scheme, _, rest = authorization.partition(" ")
        if scheme.strip().lower() != "bearer" or not rest:
            return None
        return rest.strip()

    def check_bearer(self, authorization: str | None) -> bool:
        """校验 ``Authorization`` 头是否携带有效 token。

        未配置 token 时恒返回 ``True``（鉴权关闭）；配置后要求有效 Bearer token。
        """
        if not self.is_configured():
            return True
        return self.is_valid(self.extract_bearer(authorization))


def create_api_auth_middleware(
    auth_service: AuthService | None = None,
    protected_prefix: str = "/api/",
):
    """Flask ``before_request`` 中间件工厂：对受保护前缀执行可选 Bearer Token 鉴权。

    未配置 token 时直接放行（向后兼容）；配置后要求提供有效 token，token 可通过
    三种方式提供：
    * ``Authorization: Bearer <token>``（REST 客户端）
    * ``X-Auth-Token: <token>`` 请求头（简单客户端）
    * ``?access_token=<token>`` 查询参数（供浏览器 EventSource 使用，SSE 无法带请求头）
    """
    from flask import request

    from src.web.api.schema import error_response

    service = auth_service or AuthService()

    def _require_api_auth():
        if not request.path.startswith(protected_prefix):
            return None
        if not service.is_configured():
            return None
        token = _extract_token(request)
        if token is not None and service.is_valid(token):
            return None
        return error_response("Authentication required", code="UNAUTHORIZED", status_code=401)

    return _require_api_auth


def _extract_token(request) -> str | None:
    """从请求中提取 token：Bearer 头 > X-Auth-Token 头 > access_token 查询参数。"""
    bearer_token = get_header_value(request.headers.get, "Authorization")
    if bearer_token:
        token = AuthService().extract_bearer(bearer_token)
        if token:
            return token

    x_token = get_header_value(request.headers.get, "X-Auth-Token")
    if x_token and x_token.strip():
        return x_token.strip()

    query_token = request.args.get("access_token")
    if query_token and query_token.strip():
        return query_token.strip()

    return None


def create_aiohttp_auth_middleware(protected_prefix: str = "/api/v1/"):
    """aiohttp 中间件工厂：对受保护前缀执行可选 Bearer Token 鉴权。

    未配置 token 时直接放行（向后兼容）；配置后要求携带有效 Bearer token。
    与 Flask 侧的 :func:`create_api_auth_middleware` 语义一致，供分布式
    节点/主控（:mod:`src.distributed.node_server` /
    :mod:`src.distributed.master_http_server`）复用，避免鉴权逻辑分叉。
    """
    from aiohttp import web

    @web.middleware
    async def _auth_middleware(request, handler):
        if not request.path.startswith(protected_prefix) or not token_configured():
            return await handler(request)
        token = extract_bearer(get_header_value(request.headers.get, "Authorization"))
        if not token or not check_bearer(f"Bearer {token}"):
            return web.json_response(
                {"success": False, "error": {"code": "UNAUTHORIZED", "message": "Authentication required"}},
                status=401,
            )
        return await handler(request)

    return _auth_middleware


def ensure_safe_listen_binding(host: str) -> None:
    """网络边界默认拒绝：非回环地址绑定要求已配置 API token。

    分布式节点/主控是 TTS 控制面（任意路径读写、触发合成），暴露到非回环
    地址却未配置 :data:`API_TOKEN_ENV` 时直接拒绝启动，避免无鉴权暴露在
    局域网/公网。
    """
    if not is_loopback_host(host) and not token_configured():
        raise RuntimeError(
            f"Refusing to bind to non-loopback address '{host}' without "
            f"{API_TOKEN_ENV} configured. Set the token (or bind to 127.0.0.1) to continue."
        )
