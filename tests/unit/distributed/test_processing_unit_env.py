"""Task 4 - 环境变量覆盖测试。

验证 ``WorkerUnit.start()`` / ``MasterUnit.synthesize_text()`` /
``TTSNodeService.start()`` 能正确从 ``PPC_TTS_BACKEND`` /
``PPC_TTS_PROXY`` 环境变量读取 TTS 覆盖参数并透传到底层
``TTSExecutor`` / ``TTSEngine`` 构造器。

通过 ``monkeypatch.setattr`` 将 ``src.executors.TTSExecutor`` /
``src.engines.tts_engine.TTSEngine`` 替换为 stub 类，捕获构造参数。
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.distributed.node_server import TTSNodeService
from src.distributed.processing_unit import (
    ConvertResult,
    MasterUnit,
    WorkerUnit,
)


# ---------------------------------------------------------------------------
# Stub 类：捕获 TTSExecutor / TTSEngine 构造参数
# ---------------------------------------------------------------------------


class _StubTTSExecutor:
    """``TTSExecutor`` 替身，捕获 ``tts_backend`` / ``tts_proxy`` kwargs。"""

    instances: list[_StubTTSExecutor] = []

    def __init__(
        self,
        config: Any,
        retry_policy: Any = None,
        *,
        tts_backend: str | None = None,
        tts_proxy: str | None = None,
    ) -> None:
        self.config = config
        self.retry_policy = retry_policy
        self.tts_backend = tts_backend
        self.tts_proxy = tts_proxy
        _StubTTSExecutor.instances.append(self)

    async def initialize(self) -> None:
        return None

    async def cleanup(self) -> None:
        return None


class _StubTTSEngine:
    """``TTSEngine`` 替身，捕获 ``backend_override`` / ``proxy_override`` kwargs。"""

    instances: list[_StubTTSEngine] = []

    def __init__(
        self,
        config: Any,
        *,
        backend_override: str | None = None,
        proxy_override: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config
        self.backend_override = backend_override
        self.proxy_override = proxy_override
        _StubTTSEngine.instances.append(self)

    async def initialize(self) -> None:
        return None

    async def cleanup(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_stubs():
    """每个测试前后清空 stub 实例列表，避免相互污染。"""
    _StubTTSExecutor.instances.clear()
    _StubTTSEngine.instances.clear()
    yield
    _StubTTSExecutor.instances.clear()
    _StubTTSEngine.instances.clear()


def _make_config() -> MagicMock:
    """构造一个最小可用的 MagicMock 配置对象。"""
    cfg = MagicMock()
    cfg.reliability.tts_retry.max_retries = 1
    cfg.reliability.tts_retry.base_delay = 0.01
    cfg.reliability.tts_retry.max_delay = 0.01
    cfg.reliability.tts_retry.exponential_base = 2.0
    cfg.reliability.tts_retry.jitter = 0.0
    return cfg


# ---------------------------------------------------------------------------
# Test 1: WorkerUnit.start() 从环境变量读取覆盖
# ---------------------------------------------------------------------------


def test_worker_unit_reads_env(monkeypatch):
    """``WorkerUnit.start()`` 应从环境变量读取并透传到 ``TTSExecutor``。"""
    monkeypatch.setenv("PPC_TTS_BACKEND", "env")
    monkeypatch.setenv("PPC_TTS_PROXY", "http://proxy.example.com:8080")

    import src.executors as executors_module

    monkeypatch.setattr(executors_module, "TTSExecutor", _StubTTSExecutor)

    cfg = _make_config()
    unit = WorkerUnit(host="127.0.0.1", port=0, config=cfg)

    asyncio.run(unit.start())

    assert len(_StubTTSExecutor.instances) == 1
    inst = _StubTTSExecutor.instances[0]
    assert inst.tts_backend == "env"
    assert inst.tts_proxy == "http://proxy.example.com:8080"
    assert inst.retry_policy is not None

    # Cleanup
    asyncio.run(unit.stop())


# ---------------------------------------------------------------------------
# Test 2: MasterUnit.synthesize_text() 从环境变量读取覆盖
# ---------------------------------------------------------------------------


def test_master_unit_reads_env(monkeypatch):
    """``MasterUnit`` 通过继承的 ``synthesize_text`` 懒加载路径读取环境变量。

    注：spec 描述的 ``MasterUnit._execute_locally()`` 在当前代码中由
    ``ProcessingUnit.synthesize_text()`` 的懒加载分支承担。MasterUnit
    继承该方法，调用时会触发 ``TTSExecutor`` 构造。
    """
    monkeypatch.setenv("PPC_TTS_BACKEND", "env")
    monkeypatch.setenv("PPC_TTS_PROXY", "http://proxy.example.com:8080")

    import src.executors as executors_module

    monkeypatch.setattr(executors_module, "TTSExecutor", _StubTTSExecutor)

    cfg = _make_config()
    master = MasterUnit(
        host="127.0.0.1",
        port=0,
        config=cfg,
        local_fallback=False,
    )

    # synthesize_text 会调用 handle_convert_request；这里 stub 一个
    # 失败结果以快速返回，但 TTSExecutor 构造已发生在调用之前。
    async def _stub_handle_convert_request(request):
        return ConvertResult(
            success=False,
            total=0,
            completed=0,
            failed=0,
            error="stubbed for env test",
        )

    master.handle_convert_request = _stub_handle_convert_request

    # synthesize_text 会因 stub 返回 success=False 而 raise RuntimeError，
    # 这是预期的 —— 我们只关心 TTSExecutor 是否被构造。
    with pytest.raises(RuntimeError, match="stubbed for env test"):
        asyncio.run(master.synthesize_text("hello"))

    assert len(_StubTTSExecutor.instances) == 1
    inst = _StubTTSExecutor.instances[0]
    assert inst.tts_backend == "env"
    assert inst.tts_proxy == "http://proxy.example.com:8080"


# ---------------------------------------------------------------------------
# Test 3: 不设置环境变量时，stub 收到 None
# ---------------------------------------------------------------------------


def test_no_env_no_override(monkeypatch):
    """未设置环境变量时，``tts_backend`` / ``tts_proxy`` 应为 ``None``。"""
    monkeypatch.delenv("PPC_TTS_BACKEND", raising=False)
    monkeypatch.delenv("PPC_TTS_PROXY", raising=False)

    import src.executors as executors_module

    monkeypatch.setattr(executors_module, "TTSExecutor", _StubTTSExecutor)

    cfg = _make_config()
    unit = WorkerUnit(host="127.0.0.1", port=0, config=cfg)

    asyncio.run(unit.start())

    assert len(_StubTTSExecutor.instances) == 1
    inst = _StubTTSExecutor.instances[0]
    assert inst.tts_backend is None
    assert inst.tts_proxy is None

    asyncio.run(unit.stop())


# ---------------------------------------------------------------------------
# Test 4: TTSNodeService.start() 从环境变量读取并透传到 TTSEngine
# ---------------------------------------------------------------------------


def test_node_service_reads_env(monkeypatch):
    """``TTSNodeService.start()`` 应从环境变量读取并透传到 ``TTSEngine``。"""
    monkeypatch.setenv("PPC_TTS_BACKEND", "env")
    monkeypatch.setenv("PPC_TTS_PROXY", "http://proxy.example.com:8080")

    import aiohttp

    import src.engines.tts_engine as engine_module

    monkeypatch.setattr(engine_module, "TTSEngine", _StubTTSEngine)

    cfg = _make_config()
    service = TTSNodeService(
        config=cfg,
        host="127.0.0.1",
        port=0,
        max_concurrency=1,
    )

    # ``aiohttp.web.AppRunner`` 使用 ``__slots__``，部分属性 read-only，
    # 因此直接替换整个 ``_runner`` 实例而非 patch 其方法。
    mock_runner = MagicMock()
    mock_runner.setup = AsyncMock()
    mock_runner.cleanup = AsyncMock()
    service._runner = mock_runner

    # Mock ``TCPSite`` 避免真正绑定端口
    mock_site = MagicMock()
    mock_site.start = AsyncMock()
    mock_site.stop = AsyncMock()
    monkeypatch.setattr(aiohttp.web, "TCPSite", lambda *a, **kw: mock_site)

    # 避免触发真实的 ``WorkerUnit.start()``（也会构造 TTSExecutor）
    monkeypatch.setattr(service.unit, "start", AsyncMock())
    monkeypatch.setattr(service.unit, "stop", AsyncMock())

    asyncio.run(service.start())

    assert len(_StubTTSEngine.instances) == 1
    inst = _StubTTSEngine.instances[0]
    assert inst.backend_override == "env"
    assert inst.proxy_override == "http://proxy.example.com:8080"

    asyncio.run(service.stop())
