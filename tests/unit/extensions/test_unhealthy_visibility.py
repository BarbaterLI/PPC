"""P2-20: ``adapter._failed`` 必须对调用方可见，失败扩展可被移除。

旧实现中资源监控置 ``_failed = True`` 后无人读取，``process()`` 静默返回
``None``，被终止的扩展仍留在 ``_loaded_extensions``。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.extensions.base import Extension, ExtensionMetadata, SecurityLevel
from src.extensions.executable_adapter import (
    ExecutableExtensionAdapter,
    ExtensionUnhealthyError,
)
from src.extensions.loader import ExtensionLoader
from src.extensions.package import ExtensionManifest
from src.extensions.sandbox import SandboxConfig, SandboxContext
from src.extensions.security import AuditLogger


class _FakeProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.pid = 12345

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def _adapter(tmp_path: Path) -> ExecutableExtensionAdapter:
    manifest = ExtensionManifest(
        name="unhealthy-ext",
        version="1.0.0",
        package_type="executable",
        entry_point="ext.exe",
        sha256="a" * 64,
    )
    sandbox = SandboxContext(
        SandboxConfig(
            extension_name="unhealthy-ext",
            permissions=[],
            audit_logger=AuditLogger(log_dir=tmp_path / "audit"),
            security_level=SecurityLevel.STRICT,
            data_dir=tmp_path / "data",
            package_type="executable",
        )
    )
    return ExecutableExtensionAdapter(
        manifest=manifest,
        process=_FakeProcess(),
        sandbox_context=sandbox,
        audit_logger=AuditLogger(log_dir=tmp_path / "audit2"),
        ipc_client=None,
    )


def test_process_raises_when_adapter_failed(tmp_path: Path) -> None:
    """模拟资源超限终止（``_failed=True``）后，process 必须抛明确异常。"""
    adapter = _adapter(tmp_path)
    adapter._failed = True  # ResourceMonitor 超限终止时就是这样标记的

    try:
        adapter.process("hello")
    except ExtensionUnhealthyError as exc:
        assert "unhealthy" in str(exc)
        assert "unhealthy-ext" in str(exc)
    else:  # pragma: no cover - 失败路径
        raise AssertionError("process() 静默返回 None，失败对上层不可见")


def test_loader_mark_unhealthy_removes_extension(tmp_path: Path) -> None:
    """loader.mark_unhealthy 应记录失败并移除扩展（触发 cleanup）。"""

    class _DummyExt(Extension):
        def __init__(self) -> None:
            super().__init__(ExtensionMetadata(name="dummy-ext", version="1.0.0"))
            self.cleanup_called = False

        async def initialize(self) -> None:
            self._initialized = True

        async def cleanup(self) -> None:
            self.cleanup_called = True

    loader = ExtensionLoader()
    ext = _DummyExt()
    loader._loaded_extensions["dummy-ext"] = ext

    removed = asyncio.run(loader.mark_unhealthy("dummy-ext", "resource limit exceeded"))

    assert removed is True
    assert "dummy-ext" not in loader.get_loaded_extensions()
    assert ext.cleanup_called is True
    assert any(
        "unhealthy" in item["error"] for item in loader.get_failed_extensions()
    )
