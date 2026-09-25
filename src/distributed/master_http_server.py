"""MasterHttpServer: thin HTTP wrapper around a MasterUnit.

从 :mod:`src.distributed.processing_unit` 拆出（Wave 2 规模约束），
类行为与公共 API 不变；``processing_unit`` 模块仍再导出该类以
保持 ``from src.distributed.processing_unit import MasterHttpServer``
等既有导入路径可用。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiohttp import web

    from src.distributed.processing_unit import MasterUnit

from src.web.auth import create_aiohttp_auth_middleware, ensure_safe_listen_binding

logger = logging.getLogger(__name__)


class MasterHttpServer:
    """将 :class:`MasterUnit` 暴露为 HTTP 服务的薄壳。

    启动后接收 ``POST /api/v1/convert``，调用
    :meth:`MasterUnit.handle_convert_payload`，再把结果 JSON 化返回。
    同时提供 ``GET /api/v1/health``、``GET /api/v1/stats`` 与
    ``POST /api/v1/workers``（动态注册/摘除 worker）。
    之所以把 HTTP 监听放到独立类，是因为
    :class:`MasterUnit` 本身只负责"转发/兜底"业务，
    把传输层拆开后，将来切换到 gRPC/消息队列都不会污染业务逻辑。
    """

    def __init__(self, master: MasterUnit):
        self.master = master
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    # ----------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """启动 HTTP 监听。"""
        ensure_safe_listen_binding(self.master.host)
        try:
            from aiohttp import web  # type: ignore
        except ImportError as e:  # pragma: no cover - hard dep
            raise RuntimeError("aiohttp is required for MasterHttpServer") from e

        # 与 TTSNodeService 一致：对 /api/v1/* 执行可选 Bearer Token 鉴权，
        # 避免主控的 convert/workers 控制面无鉴权暴露。
        self._app = web.Application(middlewares=[create_aiohttp_auth_middleware()])
        self._app.router.add_post("/api/v1/convert", self._handle_convert)
        self._app.router.add_get("/api/v1/health", self._handle_health)
        self._app.router.add_get("/api/v1/stats", self._handle_stats)
        self._app.router.add_post("/api/v1/workers", self._handle_add_worker)
        self._app.router.add_delete("/api/v1/workers", self._handle_remove_worker)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.master.host, self.master.port)
        await self._site.start()
        logger.info(
            "MasterHttpServer %s listening on http://%s:%s",
            self.master.node_id,
            self.master.host,
            self.master.port,
        )

    async def stop(self) -> None:
        """关闭 HTTP 监听。"""
        if self._site is not None:
            try:
                await self._site.stop()
            except Exception as e:  # noqa: BLE001
                logger.debug("MasterHttpServer site stop: %s", e)
            self._site = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception as e:  # noqa: BLE001
                logger.debug("MasterHttpServer runner cleanup: %s", e)
            self._runner = None
        self._app = None

    # ----------------------------------------------------------- routes

    async def _handle_convert(self, request) -> Any:
        """处理 ``POST /api/v1/convert``。"""
        from aiohttp import web  # type: ignore

        try:
            try:
                payload = await request.json()
            except Exception as e:  # noqa: BLE001
                return web.json_response({"error": f"invalid JSON body: {e}"}, status=400)
            if not isinstance(payload, dict):
                return web.json_response({"error": "request body must be a JSON object"}, status=400)

            data = await self.master.handle_convert_payload(payload)
            status = 200 if data.get("success") else 500
            return web.json_response(
                data,
                status=status,
                headers={"X-Node-Id": self.master.node_id},
            )
        except Exception as e:  # noqa: BLE001
            logger.error("master convert failed: %s", e)
            return web.json_response({"error": str(e), "node_id": self.master.node_id}, status=500)

    async def _handle_health(self, request) -> Any:
        """健康检查端点。"""
        from aiohttp import web  # type: ignore

        return web.json_response(
            {
                "status": "healthy",
                "node_id": self.master.node_id,
                "role": self.master.role.value,
            }
        )

    async def _handle_stats(self, request) -> Any:
        """统计端点。"""
        from aiohttp import web  # type: ignore

        return web.json_response(await self.master.get_stats())

    async def _handle_add_worker(self, request) -> Any:
        """``POST /api/v1/workers``，body 为 ``{"address": "http://..."}``。"""
        from aiohttp import web  # type: ignore

        try:
            data = await request.json()
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": f"invalid JSON body: {e}"}, status=400)
        address = (data or {}).get("address", "")
        if not address:
            return web.json_response({"error": "address is required"}, status=400)
        self.master.add_worker(address)
        return web.json_response(
            {
                "status": "added",
                "address": address,
                "workers": list(self.master.worker_addresses),
            }
        )

    async def _handle_remove_worker(self, request) -> Any:
        """``DELETE /api/v1/workers``，body 为 ``{"address": "http://..."}``。"""
        from aiohttp import web  # type: ignore

        try:
            data = await request.json()
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": f"invalid JSON body: {e}"}, status=400)
        address = (data or {}).get("address", "")
        if not address:
            return web.json_response({"error": "address is required"}, status=400)
        removed = self.master.remove_worker(address)
        return web.json_response(
            {
                "status": "removed" if removed else "not_found",
                "address": address,
                "workers": list(self.master.worker_addresses),
            },
            status=200 if removed else 404,
        )
