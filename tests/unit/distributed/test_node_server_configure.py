"""TTSNodeService.handle_configure 并发数验证测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.config import get_preset
from src.distributed.node_server import TTSNodeService


def _make_service() -> TTSNodeService:
    return TTSNodeService(
        config=get_preset("balanced"),
        host="127.0.0.1",
        port=0,  # 不绑定端口（仅构造，不 start）
        max_concurrency=4,
    )


def _req(payload: dict):
    req = MagicMock()
    req.json = AsyncMock(return_value=payload)
    return req


def test_configure_rejects_zero_max_concurrency():
    """``max_concurrency=0`` 必须被拒绝（400），不得应用。

    Bug: ``handle_configure`` 直接 ``int(data["max_concurrency"])`` 且**无下限
    校验**。``0`` 会让 ``asyncio.Semaphore(0)`` 永久零许可——所有 ``/synthesize``
    请求在 ``async with self._semaphore`` 上**无限阻塞**，节点“活着”却无法合成
    （自锁/假活）。``<0`` 由 asyncio 抛 ValueError → 400，但 ``0`` 恰好合法。
    修复: ``new_concurrency < 1`` 时返回 400 且不应用。
    """
    service = _make_service()
    resp = asyncio.run(service.handle_configure(_req({"max_concurrency": 0})))
    assert resp.status == 400, f"max_concurrency=0 应被拒绝，却 status={resp.status}"
    assert service.max_concurrency == 4, "并发数不应被改成 0"


def test_configure_accepts_positive_concurrency():
    """合法并发数（>=1）正常应用。"""
    service = _make_service()
    resp = asyncio.run(service.handle_configure(_req({"max_concurrency": 8})))
    assert resp.status == 200
    assert service.max_concurrency == 8


def test_configure_rejects_non_integer():
    """非整数并发数返回 400。"""
    service = _make_service()
    resp = asyncio.run(service.handle_configure(_req({"max_concurrency": "abc"})))
    assert resp.status == 400
