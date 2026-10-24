"""P2-17: ``/api/v1/convert`` 必须正确维护 ``current_concurrency``。

旧实现只在 ``/synthesize`` 路径增减计数，convert 路径恒为 0，导致
``NodeInfo.is_available`` 永不饱和、基于连接的负载均衡退化、drain 立即通过。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.config import get_preset
from src.distributed.node_server import TTSNodeService
from src.distributed.processing_unit import ConvertResult


class _SlowUnit:
    """处理 convert 时短暂挂起，便于观察并发峰值。"""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def handle_convert_request(self, request) -> ConvertResult:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.05)
            return ConvertResult(success=True, total=1, completed=1, failed=0)
        finally:
            self.active -= 1


def _request(payload: dict):
    req = MagicMock()
    req.json = AsyncMock(return_value=payload)
    return req


def test_convert_tracks_concurrency_peak_and_resets() -> None:
    unit = _SlowUnit()
    service = TTSNodeService(
        config=get_preset("balanced"),
        host="127.0.0.1",
        port=0,
        max_concurrency=8,
        unit=unit,
    )

    payload = {"input_dir": "in", "output_dir": "out"}

    async def scenario():
        responses = await asyncio.gather(
            *(service.handle_convert(_request(payload)) for _ in range(5))
        )
        return responses

    responses = asyncio.run(scenario())

    assert all(resp.status == 200 for resp in responses)
    # 5 个请求同时处理 → 峰值必须大于 0（旧实现恒为 0）
    assert service.get_stats()["stats"]["peak_concurrency"] >= 2
    assert unit.peak >= 2
    # 全部结束后归零（异常/正常路径都必须回退）
    assert service.get_stats()["stats"]["current_concurrency"] == 0
    assert service._current_concurrency == 0
    # 响应头把 worker 侧实时并发回传给 master
    for resp in responses:
        assert "X-Current-Concurrency" in resp.headers


def test_convert_concurrency_decrements_on_exception() -> None:
    class _BoomUnit:
        async def handle_convert_request(self, request):
            raise RuntimeError("boom")

    service = TTSNodeService(
        config=get_preset("balanced"),
        host="127.0.0.1",
        port=0,
        max_concurrency=4,
        unit=_BoomUnit(),
    )

    resp = asyncio.run(service.handle_convert(_request({"input_dir": "in", "output_dir": "out"})))

    assert resp.status == 500
    assert service.get_stats()["stats"]["current_concurrency"] == 0
    assert service._current_concurrency == 0
