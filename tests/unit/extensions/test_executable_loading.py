"""Task 6 - executable 类型扩展加载分派与适配器测试。

覆盖 SubTask 6.1 / 6.2 / 6.3 / 6.4 / 6.5 / 6.6：

* ``package_type`` 分派：python → ``_load_extension_common``，executable → ``_load_executable_extension``
* executable 非平台跳过 / exe 缺失 / 沙箱创建 / 进程启动 / 进程立即退出
* :class:`ExecutableExtensionAdapter` 的 ``cleanup`` / ``health_check`` / ``initialize`` 行为
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.extensions.base import SecurityLevel
from src.extensions.executable_adapter import ExecutableExtensionAdapter
from src.extensions.loader import ExtensionLoader
from src.extensions.package import ExtensionManifest, read_manifest_from_dir
from src.extensions.sandbox import SandboxConfig, SandboxContext
from src.extensions.security import AuditLogger

# ---------------------------------------------------------------------------
# 异步运行辅助
# ---------------------------------------------------------------------------


def _run(coro):
    """运行异步协程。"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("loop is running")
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fake subprocess.Popen 替身
# ---------------------------------------------------------------------------


class _FakeProcess:
    """subprocess.Popen 的替身，用于测试。

    参数:
        returncode: poll() 的返回值；None 表示进程仍在运行
        pid: 进程 PID（用于日志记录）
    """

    def __init__(self, *, returncode=None, pid=12345):
        self.returncode = returncode
        self.pid = pid
        self.terminate_called = False
        self.kill_called = False
        self.wait_called = False
        self.wait_timeout = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_called = True
        self.returncode = -15

    def kill(self):
        self.kill_called = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.wait_called = True
        self.wait_timeout = timeout
        return self.returncode


# ---------------------------------------------------------------------------
# 包构造辅助
# ---------------------------------------------------------------------------


def _make_executable_manifest_dict(
    name: str = "test-exe-ext",
    *,
    entry_point: str = "ext.exe",
    sha256: str = "a" * 64,
    allowed_hosts: list[str] | None = None,
) -> dict:
    """生成 executable 类型的 manifest 字典。"""
    manifest = {
        "name": name,
        "version": "1.0.0",
        "description": "Test executable extension",
        "author": "tester",
        "extension_type": "tool_integration",
        "package_type": "executable",
        "entry_point": entry_point,
        "sha256": sha256,
    }
    if allowed_hosts is not None:
        manifest["allowed_hosts"] = allowed_hosts
    return manifest


def _make_executable_package(
    tmp_path: Path,
    name: str = "test-exe-ext",
    *,
    entry_point: str = "ext.exe",
    sha256: str | None = None,
    allowed_hosts: list[str] | None = None,
    create_exe: bool = True,
) -> Path:
    """创建测试用 executable 扩展包目录（含 manifest.yml + 可选的 .exe 文件）。

    ``sha256=None`` 时：若 ``create_exe=True`` 则从 exe 内容计算真实 SHA-256
    （使哈希校验通过），否则使用占位值 ``"a" * 64``（exe 不存在时不会校验）。
    显式传入 ``sha256`` 则直接使用该值（用于测试哈希不匹配场景）。
    """
    pkg_dir = tmp_path / name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    exe_content = b"FAKE EXE CONTENT"
    if sha256 is None:
        sha256 = hashlib.sha256(exe_content).hexdigest() if create_exe else "a" * 64
    manifest = _make_executable_manifest_dict(
        name, entry_point=entry_point, sha256=sha256, allowed_hosts=allowed_hosts
    )
    (pkg_dir / "manifest.yml").write_text(
        yaml.dump(manifest, allow_unicode=True), encoding="utf-8"
    )
    if create_exe:
        (pkg_dir / entry_point).write_bytes(exe_content)
    return pkg_dir


def _make_python_package(tmp_path: Path, name: str = "test-py-ext") -> Path:
    """创建测试用 python 扩展包目录（仅含 manifest.yml，用于分派测试）。"""
    pkg_dir = tmp_path / name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "version": "1.0.0",
        "package_type": "python",
        "entry_point": "extension:TestExtension",
    }
    (pkg_dir / "manifest.yml").write_text(
        yaml.dump(manifest, allow_unicode=True), encoding="utf-8"
    )
    return pkg_dir


def _make_manifest(**kwargs) -> ExtensionManifest:
    """直接构造 ExtensionManifest 对象（用于适配器单元测试）。"""
    defaults = {
        "name": "test-exe-ext",
        "version": "1.0.0",
        "package_type": "executable",
        "entry_point": "ext.exe",
        "sha256": "a" * 64,
    }
    defaults.update(kwargs)
    return ExtensionManifest(**defaults)


def _make_sandbox(tmp_path: Path) -> SandboxContext:
    """构造 executable 类型的 SandboxContext（用于适配器单元测试）。"""
    return SandboxContext(SandboxConfig(
        extension_name="test-exe-ext",
        permissions=[],
        audit_logger=AuditLogger(log_dir=tmp_path / "audit"),
        security_level=SecurityLevel.STRICT,
        data_dir=tmp_path / "data",
        package_type="executable",
        allowed_hosts=["api.example.com"],
    ))


# ---------------------------------------------------------------------------
# SubTask 6.1: package_type 分派测试
# ---------------------------------------------------------------------------


def test_dispatch_python_calls_common(tmp_path: Path):
    """manifest.package_type="python" 时调用 _load_extension_common。"""
    pkg_dir = _make_python_package(tmp_path)
    loader = ExtensionLoader()
    try:
        with patch.object(
            loader, "_load_extension_common", return_value=None
        ) as mock_common:
            _run(loader._load_extension_from_package(pkg_dir))

        assert mock_common.called, "_load_extension_common should be called for python type"
    finally:
        _run(loader.clear())


def test_dispatch_executable_calls_executable_loader(tmp_path: Path):
    """manifest.package_type="executable" 时调用 _load_executable_extension。"""
    pkg_dir = _make_executable_package(tmp_path)
    loader = ExtensionLoader()
    try:
        with patch.object(
            loader, "_load_executable_extension", return_value=None
        ) as mock_exe:
            _run(loader._load_extension_from_package(pkg_dir))

        assert mock_exe.called, "_load_executable_extension should be called for executable type"
        # Verify package_path and manifest passed correctly
        call_args = mock_exe.call_args
        assert call_args.args[0] == pkg_dir
        assert call_args.args[1].package_type == "executable"
        assert call_args.args[1].name == "test-exe-ext"
    finally:
        _run(loader.clear())


# ---------------------------------------------------------------------------
# SubTask 6.2: _load_executable_extension 行为测试
# ---------------------------------------------------------------------------


def test_executable_skipped_on_non_windows(
    tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
):
    """非 Windows 平台 executable 类型 WARNING 并跳过，返回 None。"""
    pkg_dir = _make_executable_package(tmp_path)
    manifest = read_manifest_from_dir(pkg_dir)
    assert manifest is not None

    loader = ExtensionLoader()
    monkeypatch.setattr(sys, "platform", "linux")
    try:
        with caplog.at_level(logging.WARNING, logger="src.extensions.loader"):
            result = _run(loader._load_executable_extension(pkg_dir, manifest))

        assert result is None
        assert any(
            "only supported on Windows" in rec.message and "test-exe-ext" in rec.message
            for rec in caplog.records
        ), f"expected a WARNING about Windows-only support, got: {[r.message for r in caplog.records]}"
    finally:
        _run(loader.clear())


def test_executable_missing_exe_returns_none(
    tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
):
    """entry_point 指向的 exe 不存在时返回 None 并记录 ERROR。"""
    pkg_dir = _make_executable_package(
        tmp_path, entry_point="missing.exe", create_exe=False
    )
    manifest = read_manifest_from_dir(pkg_dir)
    assert manifest is not None

    loader = ExtensionLoader()
    monkeypatch.setattr(sys, "platform", "win32")
    try:
        with caplog.at_level(logging.ERROR, logger="src.extensions.loader"):
            result = _run(loader._load_executable_extension(pkg_dir, manifest))

        assert result is None
        assert any(
            "file not found" in rec.message and "missing.exe" in rec.message
            for rec in caplog.records
        ), f"expected an ERROR about missing exe, got: {[r.message for r in caplog.records]}"
        # Failure should be recorded
        failures = loader.get_failed_extensions()
        assert len(failures) >= 1
        assert any("file not found" in f["error"] for f in failures)
    finally:
        _run(loader.clear())


def test_executable_creates_sandbox_context(
    tmp_path: Path, monkeypatch
):
    """executable 类型加载时创建 SandboxContext（package_type="executable" + STRICT）。"""
    pkg_dir = _make_executable_package(
        tmp_path, allowed_hosts=["api.example.com"]
    )
    manifest = read_manifest_from_dir(pkg_dir)
    assert manifest is not None

    loader = ExtensionLoader()
    monkeypatch.setattr(sys, "platform", "win32")
    fake_process = _FakeProcess(returncode=None)
    try:
        with patch("src.extensions.process_manager.subprocess.Popen", return_value=fake_process), \
             patch("src.extensions.process_manager.time.sleep"):
            result = _run(loader._load_executable_extension(pkg_dir, manifest))

        assert result is not None
        assert isinstance(result, ExecutableExtensionAdapter)
        # Verify sandbox context created with executable package_type
        sandbox = result._sandbox
        assert sandbox._package_type == "executable"
        # executable type forces STRICT
        assert sandbox.security_level == SecurityLevel.STRICT
        # allowed_hosts propagated from manifest
        assert sandbox.is_host_allowed("api.example.com") is True
        assert sandbox.is_host_allowed("evil.com") is False
    finally:
        _run(loader.clear())


def test_executable_spawns_process(
    tmp_path: Path, monkeypatch
):
    """subprocess.Popen 以正确的 exe 路径被调用。"""
    pkg_dir = _make_executable_package(tmp_path, entry_point="ext.exe")
    manifest = read_manifest_from_dir(pkg_dir)
    assert manifest is not None

    loader = ExtensionLoader()
    monkeypatch.setattr(sys, "platform", "win32")
    fake_process = _FakeProcess(returncode=None)
    try:
        with patch(
            "src.extensions.process_manager.subprocess.Popen", return_value=fake_process
        ) as mock_popen, \
             patch("src.extensions.process_manager.time.sleep"):
            result = _run(loader._load_executable_extension(pkg_dir, manifest))

        assert result is not None
        mock_popen.assert_called_once_with([str(pkg_dir / "ext.exe")])
    finally:
        _run(loader.clear())


def test_executable_process_dies_returns_none(
    tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
):
    """进程启动后立即退出时返回 None 并记录 ERROR。"""
    pkg_dir = _make_executable_package(tmp_path)
    manifest = read_manifest_from_dir(pkg_dir)
    assert manifest is not None

    loader = ExtensionLoader()
    monkeypatch.setattr(sys, "platform", "win32")
    # poll() returns 1 → process exited immediately
    fake_process = _FakeProcess(returncode=1)
    try:
        with caplog.at_level(logging.ERROR, logger="src.extensions.loader"), \
             patch("src.extensions.process_manager.subprocess.Popen", return_value=fake_process), \
             patch("src.extensions.process_manager.time.sleep"):
            result = _run(loader._load_executable_extension(pkg_dir, manifest))

        assert result is None
        assert any(
            "exited immediately" in rec.message and "test-exe-ext" in rec.message
            for rec in caplog.records
        ), f"expected an ERROR about immediate exit, got: {[r.message for r in caplog.records]}"
        failures = loader.get_failed_extensions()
        assert any("exited immediately" in f["error"] for f in failures)
    finally:
        _run(loader.clear())


# ---------------------------------------------------------------------------
# SubTask 6.3: ExecutableExtensionAdapter 行为测试
# ---------------------------------------------------------------------------


def test_executable_adapter_cleanup_terminates(tmp_path: Path):
    """ExecutableExtensionAdapter.cleanup() 调用 process.terminate() 和 process.wait()。"""
    fake_process = _FakeProcess(returncode=None)  # running
    manifest = _make_manifest()
    sandbox = _make_sandbox(tmp_path)
    audit_logger = AuditLogger(log_dir=tmp_path / "audit")

    adapter = ExecutableExtensionAdapter(
        manifest=manifest,
        process=fake_process,
        sandbox_context=sandbox,
        audit_logger=audit_logger,
    )

    assert fake_process.terminate_called is False
    assert fake_process.wait_called is False

    _run(adapter.cleanup())

    assert fake_process.terminate_called is True, "terminate() should be called on cleanup"
    assert fake_process.wait_called is True, "wait() should be called on cleanup"
    assert fake_process.wait_timeout == 5, "wait() should be called with timeout=5"
    assert adapter._initialized is False


def test_executable_adapter_cleanup_skips_already_dead_process(tmp_path: Path):
    """进程已退出时 cleanup() 不调用 terminate（避免对已死进程操作）。"""
    fake_process = _FakeProcess(returncode=0)  # already exited
    manifest = _make_manifest()
    sandbox = _make_sandbox(tmp_path)
    audit_logger = AuditLogger(log_dir=tmp_path / "audit")

    adapter = ExecutableExtensionAdapter(
        manifest=manifest,
        process=fake_process,
        sandbox_context=sandbox,
        audit_logger=audit_logger,
    )

    _run(adapter.cleanup())

    assert fake_process.terminate_called is False, "terminate() should NOT be called for dead process"
    assert adapter._initialized is False


def test_executable_adapter_health_check(tmp_path: Path):
    """health_check() 返回 {"status": "ok"} 当进程运行，{"status": "dead"} 当进程退出。"""
    manifest = _make_manifest()
    sandbox = _make_sandbox(tmp_path)
    audit_logger = AuditLogger(log_dir=tmp_path / "audit")

    # Running process
    running_process = _FakeProcess(returncode=None)
    adapter_running = ExecutableExtensionAdapter(
        manifest=manifest, process=running_process,
        sandbox_context=sandbox, audit_logger=audit_logger,
    )
    assert adapter_running.health_check() == {"status": "ok"}

    # Dead process
    dead_process = _FakeProcess(returncode=1)
    adapter_dead = ExecutableExtensionAdapter(
        manifest=manifest, process=dead_process,
        sandbox_context=sandbox, audit_logger=audit_logger,
    )
    assert adapter_dead.health_check() == {"status": "dead"}


def test_executable_adapter_initialize_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """initialize() 记录 info 日志（Task 9 将替换为 IPC 调用）。"""
    fake_process = _FakeProcess(returncode=None, pid=99999)
    manifest = _make_manifest(name="my-exe-ext")
    sandbox = _make_sandbox(tmp_path)
    audit_logger = AuditLogger(log_dir=tmp_path / "audit")

    adapter = ExecutableExtensionAdapter(
        manifest=manifest,
        process=fake_process,
        sandbox_context=sandbox,
        audit_logger=audit_logger,
    )

    assert adapter._initialized is False

    with caplog.at_level(logging.INFO, logger="src.extensions.executable_adapter"):
        _run(adapter.initialize())

    assert adapter._initialized is True
    assert any(
        "initialized" in rec.message and "my-exe-ext" in rec.message and "99999" in rec.message
        for rec in caplog.records
    ), f"expected an INFO log about initialization with pid, got: {[r.message for r in caplog.records]}"


def test_executable_adapter_on_error_marks_failed(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """on_error() 记录错误并标记扩展为失败状态。"""
    fake_process = _FakeProcess(returncode=None)
    manifest = _make_manifest(name="err-exe-ext")
    sandbox = _make_sandbox(tmp_path)
    audit_logger = AuditLogger(log_dir=tmp_path / "audit")

    adapter = ExecutableExtensionAdapter(
        manifest=manifest,
        process=fake_process,
        sandbox_context=sandbox,
        audit_logger=audit_logger,
    )

    assert adapter._failed is False

    error = RuntimeError("IPC pipe broken")
    with caplog.at_level(logging.ERROR, logger="src.extensions.executable_adapter"):
        _run(adapter.on_error(adapter._context, error))

    assert adapter._failed is True
    assert any(
        "err-exe-ext" in rec.message and "IPC pipe broken" in rec.message
        for rec in caplog.records
    ), f"expected an ERROR log about the error, got: {[r.message for r in caplog.records]}"


def test_executable_adapter_process_stub(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """process() stub 记录日志并返回 None（Task 9 替换为 IPC 调用）。"""
    fake_process = _FakeProcess(returncode=None)
    manifest = _make_manifest(name="proc-exe-ext")
    sandbox = _make_sandbox(tmp_path)
    audit_logger = AuditLogger(log_dir=tmp_path / "audit")

    adapter = ExecutableExtensionAdapter(
        manifest=manifest,
        process=fake_process,
        sandbox_context=sandbox,
        audit_logger=audit_logger,
    )

    with caplog.at_level(logging.INFO, logger="src.extensions.executable_adapter"):
        result = adapter.process("some text", context={"key": "value"})

    assert result is None
    assert any(
        "process called" in rec.message and "proc-exe-ext" in rec.message
        for rec in caplog.records
    ), f"expected an INFO log about process call, got: {[r.message for r in caplog.records]}"
