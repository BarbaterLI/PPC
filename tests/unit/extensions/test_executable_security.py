"""Task 7 - executable 类型扩展安全限制测试。

覆盖 SubTask 7.1-7.7:

* SHA-256 哈希校验（匹配 / 不匹配 / 空值 WARNING）
* 资源监控（CPU / 内存 / 超时，超限终止进程 + WARNING）
* IPC 审计日志 stub（``log_ipc_call`` 在 initialize / process / cleanup 中被调用）
* :meth:`AuditLogger.log_ipc_call` 方法存在且写入审计日志
* :meth:`ExecutableExtensionAdapter._check_network_host_for_ipc` 委托给 sandbox
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from src.extensions.base import SecurityLevel
from src.extensions.executable_adapter import ExecutableExtensionAdapter
from src.extensions.loader import ExtensionLoader
from src.extensions.package import ExtensionManifest
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
    """subprocess.Popen 的替身，用于测试。"""

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


_EXE_CONTENT = b"FAKE EXE CONTENT"


def _make_exe_package(
    tmp_path: Path,
    name: str = "sec-exe-ext",
    *,
    sha256: str | None = None,
    exe_content: bytes = _EXE_CONTENT,
) -> Path:
    """创建测试用 executable 扩展包目录。

    ``sha256=None`` 时从 exe 内容计算真实 SHA-256（使校验通过）。
    传入 ``""`` 测试空值 WARNING，传入其他值测试不匹配。
    """
    pkg_dir = tmp_path / name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    actual_sha = hashlib.sha256(exe_content).hexdigest()
    manifest_sha = sha256 if sha256 is not None else actual_sha
    manifest = {
        "name": name,
        "version": "1.0.0",
        "package_type": "executable",
        "entry_point": "ext.exe",
        "sha256": manifest_sha,
    }
    (pkg_dir / "manifest.yml").write_text(
        yaml.dump(manifest, allow_unicode=True), encoding="utf-8"
    )
    (pkg_dir / "ext.exe").write_bytes(exe_content)
    return pkg_dir


def _make_manifest(**kwargs) -> ExtensionManifest:
    """直接构造 ExtensionManifest 对象（用于适配器单元测试）。"""
    defaults = {
        "name": "sec-exe-ext",
        "version": "1.0.0",
        "package_type": "executable",
        "entry_point": "ext.exe",
        "sha256": "a" * 64,
    }
    defaults.update(kwargs)
    return ExtensionManifest(**defaults)


def _make_sandbox(tmp_path: Path, *, allowed_hosts: list[str] | None = None) -> SandboxContext:
    """构造 executable 类型的 SandboxContext（用于适配器单元测试）。"""
    return SandboxContext(SandboxConfig(
        extension_name="sec-exe-ext",
        permissions=[],
        audit_logger=AuditLogger(log_dir=tmp_path / "audit"),
        security_level=SecurityLevel.STRICT,
        data_dir=tmp_path / "data",
        package_type="executable",
        allowed_hosts=allowed_hosts if allowed_hosts is not None else ["api.example.com"],
    ))


def _make_adapter(
    tmp_path: Path,
    *,
    manifest: ExtensionManifest | None = None,
    process: _FakeProcess | None = None,
    sandbox: SandboxContext | None = None,
    audit_logger: AuditLogger | Mock | None = None,
) -> tuple[ExecutableExtensionAdapter, _FakeProcess]:
    """构造 ExecutableExtensionAdapter 及其 FakeProcess。"""
    if manifest is None:
        manifest = _make_manifest()
    if process is None:
        process = _FakeProcess(returncode=None)
    if sandbox is None:
        sandbox = _make_sandbox(tmp_path)
    if audit_logger is None:
        audit_logger = AuditLogger(log_dir=tmp_path / "audit")
    adapter = ExecutableExtensionAdapter(
        manifest=manifest,
        process=process,
        sandbox_context=sandbox,
        audit_logger=audit_logger,
    )
    return adapter, process


# ---------------------------------------------------------------------------
# SubTask 7.1: SHA-256 哈希校验
# ---------------------------------------------------------------------------


def test_sha256_match_loads(tmp_path: Path, monkeypatch):
    """exe 的 SHA-256 与 manifest 一致时正常加载。"""
    pkg_dir = _make_exe_package(tmp_path)
    from src.extensions.package import read_manifest_from_dir

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
    finally:
        _run(loader.clear())


def test_sha256_mismatch_rejected(
    tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
):
    """exe 的 SHA-256 与 manifest 不一致时拒绝加载并记录 ERROR。"""
    pkg_dir = _make_exe_package(tmp_path, sha256="b" * 64)
    from src.extensions.package import read_manifest_from_dir

    manifest = read_manifest_from_dir(pkg_dir)
    assert manifest is not None

    loader = ExtensionLoader()
    monkeypatch.setattr(sys, "platform", "win32")
    try:
        with caplog.at_level(logging.ERROR, logger="src.extensions.loader"), \
             patch("src.extensions.process_manager.subprocess.Popen") as mock_popen:
            result = _run(loader._load_executable_extension(pkg_dir, manifest))

        assert result is None, "mismatched sha256 should reject loading"
        # 进程不应被启动
        assert mock_popen.called is False, "process should NOT be spawned on hash mismatch"
        # ERROR 日志
        assert any(
            "hash mismatch" in rec.message.lower() and "sec-exe-ext" in rec.message
            for rec in caplog.records
        ), f"expected ERROR about hash mismatch, got: {[r.message for r in caplog.records]}"
        # 失败被记录
        failures = loader.get_failed_extensions()
        assert any("hash mismatch" in f["error"].lower() for f in failures)
    finally:
        _run(loader.clear())


def test_sha256_empty_warns_but_loads(
    tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
):
    """manifest.sha256 为空时 WARNING 但继续加载（不阻塞）。"""
    pkg_dir = _make_exe_package(tmp_path, sha256="")
    from src.extensions.package import read_manifest_from_dir

    manifest = read_manifest_from_dir(pkg_dir)
    assert manifest is not None
    assert manifest.sha256 == ""

    loader = ExtensionLoader()
    monkeypatch.setattr(sys, "platform", "win32")
    fake_process = _FakeProcess(returncode=None)
    try:
        with caplog.at_level(logging.WARNING, logger="src.extensions.loader"), \
             patch("src.extensions.process_manager.subprocess.Popen", return_value=fake_process), \
             patch("src.extensions.process_manager.time.sleep"):
            result = _run(loader._load_executable_extension(pkg_dir, manifest))

        assert result is not None, "empty sha256 should not block loading"
        assert isinstance(result, ExecutableExtensionAdapter)
        assert any(
            "no sha256" in rec.message.lower() and "insecure" in rec.message.lower()
            for rec in caplog.records
        ), f"expected WARNING about missing sha256, got: {[r.message for r in caplog.records]}"
    finally:
        _run(loader.clear())


# ---------------------------------------------------------------------------
# SubTask 7.2: 强制 STRICT（由 Task 8 SandboxContext 实现，此处验证）
# ---------------------------------------------------------------------------


def test_executable_load_forces_strict(tmp_path: Path, monkeypatch):
    """executable 加载时 SandboxContext 强制 STRICT（即使 loader 设为 WARN）。"""
    pkg_dir = _make_exe_package(tmp_path)
    from src.extensions.package import read_manifest_from_dir

    manifest = read_manifest_from_dir(pkg_dir)
    assert manifest is not None

    loader = ExtensionLoader(security_level=SecurityLevel.WARN)
    monkeypatch.setattr(sys, "platform", "win32")
    fake_process = _FakeProcess(returncode=None)
    try:
        with patch("src.extensions.process_manager.subprocess.Popen", return_value=fake_process), \
             patch("src.extensions.process_manager.time.sleep"):
            result = _run(loader._load_executable_extension(pkg_dir, manifest))

        assert result is not None
        assert result._sandbox.security_level == SecurityLevel.STRICT
    finally:
        _run(loader.clear())


# ---------------------------------------------------------------------------
# SubTask 7.3: 文件系统沙箱限制（由 Task 8 实现，此处验证）
# ---------------------------------------------------------------------------


def test_executable_load_sandbox_dir_restricted(tmp_path: Path, monkeypatch):
    """executable 加载时 _allowed_paths 仅含 extensions_data/<name>/sandbox/。"""
    pkg_dir = _make_exe_package(tmp_path)
    from src.extensions.package import read_manifest_from_dir

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
        sandbox = result._sandbox
        assert sandbox._package_type == "executable"
        # 仅允许 sandbox 目录
        assert len(sandbox._allowed_paths) == 1
        sandbox_dir = sandbox._allowed_paths[0]
        assert sandbox_dir.name == "sandbox"
        assert sandbox_dir.parent.name == manifest.name
    finally:
        _run(loader.clear())


# ---------------------------------------------------------------------------
# SubTask 7.4: 网络白名单委托
# ---------------------------------------------------------------------------


def test_check_network_host_for_ipc_delegates(tmp_path: Path):
    """_check_network_host_for_ipc 委托给 sandbox_context.check_network_host。"""
    manifest = _make_manifest()
    sandbox = Mock(spec=SandboxContext)
    audit_logger = AuditLogger(log_dir=tmp_path / "audit")
    fake_process = _FakeProcess(returncode=None)

    adapter = ExecutableExtensionAdapter(
        manifest=manifest,
        process=fake_process,
        sandbox_context=sandbox,
        audit_logger=audit_logger,
    )

    adapter._check_network_host_for_ipc("api.example.com")

    sandbox.check_network_host.assert_called_once_with("api.example.com")


def test_check_network_host_for_ipc_raises_on_disallowed(tmp_path: Path):
    """_check_network_host_for_ipc 在主机不在白名单时抛出 PermissionError。"""
    manifest = _make_manifest()
    sandbox = _make_sandbox(tmp_path, allowed_hosts=["api.example.com"])
    audit_logger = AuditLogger(log_dir=tmp_path / "audit")
    fake_process = _FakeProcess(returncode=None)

    adapter = ExecutableExtensionAdapter(
        manifest=manifest,
        process=fake_process,
        sandbox_context=sandbox,
        audit_logger=audit_logger,
    )

    with pytest.raises(PermissionError) as exc_info:
        adapter._check_network_host_for_ipc("evil.com")
    assert "evil.com" in str(exc_info.value)


# ---------------------------------------------------------------------------
# SubTask 7.5: 资源限制监控
# ---------------------------------------------------------------------------


def test_resource_monitor_terminates_on_cpu_exceed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """CPU 超限时终止进程并记录 WARNING。"""
    fake_process = _FakeProcess(returncode=None)
    manifest = _make_manifest(resource_limits={"max_cpu_percent": 50})
    adapter, _ = _make_adapter(
        tmp_path, manifest=manifest, process=fake_process
    )

    loader = ExtensionLoader()
    with patch("src.extensions.resource_monitor.psutil.Process") as mock_proc_cls, \
         patch("src.extensions.process_manager.time.sleep"), \
         caplog.at_level(logging.WARNING, logger="src.extensions.loader"):
        mock_proc = mock_proc_cls.return_value
        mock_proc.cpu_percent.return_value = 80.0
        loader._monitor_resources(fake_process, manifest, adapter)

    assert fake_process.terminate_called, "process should be terminated on CPU exceed"
    assert adapter._failed is True
    assert any(
        "exceeded CPU limit" in rec.message and "sec-exe-ext" in rec.message
        for rec in caplog.records
    ), f"expected WARNING about CPU limit, got: {[r.message for r in caplog.records]}"


def test_resource_monitor_terminates_on_memory_exceed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """内存超限时终止进程并记录 WARNING。"""
    fake_process = _FakeProcess(returncode=None)
    manifest = _make_manifest(resource_limits={"max_memory_mb": 512})
    adapter, _ = _make_adapter(
        tmp_path, manifest=manifest, process=fake_process
    )

    loader = ExtensionLoader()
    with patch("src.extensions.resource_monitor.psutil.Process") as mock_proc_cls, \
         patch("src.extensions.process_manager.time.sleep"), \
         caplog.at_level(logging.WARNING, logger="src.extensions.loader"):
        mock_proc = mock_proc_cls.return_value
        # CPU 在限内，不触发 CPU 检查
        mock_proc.cpu_percent.return_value = 0.0
        # 1GB RSS > 512MB 限制
        mock_mem = mock_proc.memory_info.return_value
        mock_mem.rss = 1024 * 1024 * 1024
        loader._monitor_resources(fake_process, manifest, adapter)

    assert fake_process.terminate_called, "process should be terminated on memory exceed"
    assert adapter._failed is True
    assert any(
        "exceeded memory limit" in rec.message and "sec-exe-ext" in rec.message
        for rec in caplog.records
    ), f"expected WARNING about memory limit, got: {[r.message for r in caplog.records]}"


def test_resource_monitor_terminates_on_timeout(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """超时时终止进程并记录 WARNING。"""
    fake_process = _FakeProcess(returncode=None)
    manifest = _make_manifest(resource_limits={"timeout_seconds": 1})
    adapter, _ = _make_adapter(
        tmp_path, manifest=manifest, process=fake_process
    )

    loader = ExtensionLoader()
    # Mock time.time: start=0.0, all subsequent calls (elapsed + logging) = 2.0
    #   → elapsed = 2.0 - 0.0 = 2.0 > 1s timeout
    # Mock psutil.Process to raise (fake pid), so proc=None → 跳过 CPU/内存检查
    time_values = [0.0] + [2.0] * 50
    with patch(
        "src.extensions.resource_monitor.time.time", side_effect=time_values
    ), \
         patch("src.extensions.process_manager.time.sleep"), \
         patch(
             "src.extensions.resource_monitor.psutil.Process",
             side_effect=Exception("no such process"),
         ), \
         caplog.at_level(logging.WARNING, logger="src.extensions.loader"):
        loader._monitor_resources(fake_process, manifest, adapter)

    assert fake_process.terminate_called, "process should be terminated on timeout"
    assert adapter._failed is True
    assert any(
        "exceeded timeout" in rec.message and "sec-exe-ext" in rec.message
        for rec in caplog.records
    ), f"expected WARNING about timeout, got: {[r.message for r in caplog.records]}"


def test_resource_monitor_exits_when_process_dies(tmp_path: Path):
    """进程已退出时监控线程正常退出（不调用 terminate）。"""
    fake_process = _FakeProcess(returncode=0)  # already exited
    manifest = _make_manifest(resource_limits={"timeout_seconds": 100})
    adapter, _ = _make_adapter(
        tmp_path, manifest=manifest, process=fake_process
    )

    loader = ExtensionLoader()
    with patch("src.extensions.process_manager.time.sleep"):
        loader._monitor_resources(fake_process, manifest, adapter)

    assert fake_process.terminate_called is False
    assert adapter._failed is False


# ---------------------------------------------------------------------------
# SubTask 7.6: IPC 审计日志
# ---------------------------------------------------------------------------


def test_audit_log_ipc_call_called_in_initialize(tmp_path: Path):
    """initialize() 调用 audit_logger.log_ipc_call。"""
    manifest = _make_manifest(name="audit-init-ext")
    audit_logger = Mock(spec=AuditLogger)
    adapter, _ = _make_adapter(
        tmp_path, manifest=manifest, audit_logger=audit_logger
    )

    _run(adapter.initialize())

    audit_logger.log_ipc_call.assert_called_once_with(
        "audit-init-ext", "initialize", "", "stub"
    )


def test_audit_log_ipc_call_called_in_process(tmp_path: Path):
    """process() 调用 audit_logger.log_ipc_call。"""
    manifest = _make_manifest(name="audit-proc-ext")
    audit_logger = Mock(spec=AuditLogger)
    adapter, _ = _make_adapter(
        tmp_path, manifest=manifest, audit_logger=audit_logger
    )

    adapter.process("some text")

    audit_logger.log_ipc_call.assert_called_once_with(
        "audit-proc-ext", "process", "", "stub"
    )


def test_audit_log_ipc_call_called_in_cleanup(tmp_path: Path):
    """cleanup() 调用 audit_logger.log_ipc_call。"""
    manifest = _make_manifest(name="audit-clean-ext")
    audit_logger = Mock(spec=AuditLogger)
    adapter, _ = _make_adapter(
        tmp_path, manifest=manifest, audit_logger=audit_logger
    )

    _run(adapter.cleanup())

    audit_logger.log_ipc_call.assert_called_once_with(
        "audit-clean-ext", "cleanup", "", "stub"
    )


def test_audit_log_ipc_call_method_exists(tmp_path: Path):
    """AuditLogger.log_ipc_call 方法存在且写入审计日志文件。"""
    audit_logger = AuditLogger(log_dir=tmp_path / "audit")

    audit_logger.log_ipc_call(
        "test-ext", "initialize", '{"key": "value"}', "ok"
    )

    logs = audit_logger.read_recent_logs()
    assert any(
        log["action"] == "ipc_call"
        and log["extension_name"] == "test-ext"
        and log["target"] == "initialize"
        and log["result"] == "ok"
        and log["details"].get("data_summary") == '{"key": "value"}'
        for log in logs
    ), f"expected an ipc_call audit entry, got: {logs}"


def test_audit_log_ipc_call_with_details(tmp_path: Path):
    """log_ipc_call 接受 details 参数并写入审计日志。"""
    audit_logger = AuditLogger(log_dir=tmp_path / "audit")

    audit_logger.log_ipc_call(
        "test-ext", "process", "text-data", "error",
        details={"error_code": 500, "duration_ms": 120},
    )

    logs = audit_logger.read_recent_logs()
    assert any(
        log["action"] == "ipc_call"
        and log["result"] == "error"
        and log["details"].get("error_code") == 500
        and log["details"].get("duration_ms") == 120
        and log["details"].get("data_summary") == "text-data"
        for log in logs
    ), f"expected an ipc_call audit entry with details, got: {logs}"
