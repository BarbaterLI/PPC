"""Unit tests for executable-type strengthening of :mod:`src.extensions.sandbox`.

Covers the executable-specific hardening of :class:`SandboxContext`:
forced STRICT security level, sandbox directory containment, host whitelist
filtering via ``is_host_allowed`` / ``check_network_host``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.extensions.base import ExtensionPermission, SecurityLevel
from src.extensions.sandbox import SandboxConfig, SandboxContext
from src.extensions.security import AuditLogger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def audit_logger(tmp_path: Path) -> AuditLogger:
    """Fresh AuditLogger writing to an isolated tmp directory."""
    return AuditLogger(log_dir=tmp_path / "audit")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_executable_sandbox(
    data_dir: Path,
    audit_logger: AuditLogger,
    *,
    permissions: list[ExtensionPermission] | None = None,
    security_level: SecurityLevel = SecurityLevel.STRICT,
    allowed_paths: list[str | Path] | None = None,
    allowed_hosts: list[str] | None = None,
) -> SandboxContext:
    return SandboxContext(SandboxConfig(
        extension_name="test_exe_ext",
        permissions=permissions or [],
        audit_logger=audit_logger,
        security_level=security_level,
        data_dir=data_dir,
        allowed_paths=allowed_paths,
        package_type="executable",
        allowed_hosts=allowed_hosts,
    ))


# ---------------------------------------------------------------------------
# SubTask 8.2: executable forces STRICT
# ---------------------------------------------------------------------------


def test_executable_forces_strict(data_dir: Path, audit_logger: AuditLogger) -> None:
    """executable type with WARN declared is forced to STRICT."""
    sandbox = _make_executable_sandbox(
        data_dir, audit_logger, security_level=SecurityLevel.WARN
    )
    assert sandbox.security_level == SecurityLevel.STRICT


def test_executable_strict_warning_logged(
    data_dir: Path, audit_logger: AuditLogger, caplog: pytest.LogCaptureFixture
) -> None:
    """executable type forcing STRICT emits a WARNING explaining the override."""
    with caplog.at_level(logging.WARNING, logger="src.extensions.sandbox"):
        _make_executable_sandbox(
            data_dir, audit_logger, security_level=SecurityLevel.WARN
        )
    assert any(
        "executable type" in rec.message and "forcing SecurityLevel.STRICT" in rec.message
        for rec in caplog.records
    ), "expected a WARNING explaining STRICT override for executable type"


def test_executable_keeps_strict(
    data_dir: Path, audit_logger: AuditLogger, caplog: pytest.LogCaptureFixture
) -> None:
    """executable type already STRICT does not emit an override WARNING."""
    with caplog.at_level(logging.WARNING, logger="src.extensions.sandbox"):
        sandbox = _make_executable_sandbox(
            data_dir, audit_logger, security_level=SecurityLevel.STRICT
        )
    assert sandbox.security_level == SecurityLevel.STRICT
    assert not any(
        "forcing SecurityLevel.STRICT" in rec.message for rec in caplog.records
    ), "no override WARNING should be emitted when already STRICT"


# ---------------------------------------------------------------------------
# SubTask 8.3: sandbox dir only, ignores allowed_paths, creates dir
# ---------------------------------------------------------------------------


def test_executable_sandbox_dir_only(
    data_dir: Path, audit_logger: AuditLogger, tmp_path: Path
) -> None:
    """executable type: paths inside sandbox/ are allowed, outside are blocked."""
    sandbox = _make_executable_sandbox(
        data_dir, audit_logger,
        permissions=[ExtensionPermission.FILESYSTEM_READ, ExtensionPermission.FILESYSTEM_WRITE],
    )
    sandbox_dir = data_dir / "sandbox"

    # Inside sandbox: allowed
    inside_file = sandbox_dir / "data.txt"
    inside_file.write_text("hello", encoding="utf-8")
    with sandbox.safe_open(inside_file, "r") as f:
        assert f.read() == "hello"

    # Outside sandbox (in data_dir root but not in sandbox/): blocked
    outside_file = data_dir / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")
    with pytest.raises(PermissionError):
        sandbox.safe_open(outside_file, "r")


def test_executable_ignores_allowed_paths_param(
    data_dir: Path, audit_logger: AuditLogger, tmp_path: Path
) -> None:
    """executable type ignores the allowed_paths parameter."""
    extra_dir = tmp_path / "extra"
    extra_dir.mkdir()
    extra_file = extra_dir / "file.txt"
    extra_file.write_text("x", encoding="utf-8")

    sandbox = _make_executable_sandbox(
        data_dir, audit_logger, allowed_paths=[extra_dir]
    )
    # extra_dir must NOT be in _allowed_paths after executable reset
    assert extra_dir.resolve() not in sandbox._allowed_paths
    assert sandbox._is_path_allowed(extra_file) is False


def test_executable_creates_sandbox_dir(
    data_dir: Path, audit_logger: AuditLogger
) -> None:
    """executable type creates extensions_data/<name>/sandbox/ directory."""
    sandbox = _make_executable_sandbox(data_dir, audit_logger)
    sandbox_dir = data_dir / "sandbox"
    assert sandbox_dir.exists() and sandbox_dir.is_dir()
    # The sandbox dir (resolved) is the only allowed path
    assert sandbox._allowed_paths == [sandbox_dir.resolve()]


# ---------------------------------------------------------------------------
# SubTask 8.4: is_host_allowed
# ---------------------------------------------------------------------------


def test_is_host_allowed_python_always_true(
    data_dir: Path, audit_logger: AuditLogger
) -> None:
    """python type: any host returns True (no host filtering)."""
    sandbox = SandboxContext(SandboxConfig(
        extension_name="test_py_ext",
        permissions=[],
        audit_logger=audit_logger,
        security_level=SecurityLevel.STRICT,
        data_dir=data_dir,
        package_type="python",
    ))
    assert sandbox.is_host_allowed("anything.com") is True
    assert sandbox.is_host_allowed("evil.com") is True
    assert sandbox.is_host_allowed("api.example.com:443") is True


def test_is_host_allowed_executable_whitelist(
    data_dir: Path, audit_logger: AuditLogger
) -> None:
    """executable type: only whitelisted hosts return True."""
    sandbox = _make_executable_sandbox(
        data_dir, audit_logger, allowed_hosts=["api.example.com"]
    )
    assert sandbox.is_host_allowed("api.example.com") is True
    assert sandbox.is_host_allowed("evil.com") is False


def test_is_host_allowed_wildcard(
    data_dir: Path, audit_logger: AuditLogger
) -> None:
    """executable type: wildcard '*.example.com' matches subdomains only."""
    sandbox = _make_executable_sandbox(
        data_dir, audit_logger, allowed_hosts=["*.example.com"]
    )
    assert sandbox.is_host_allowed("sub.example.com") is True
    assert sandbox.is_host_allowed("notexample.com") is False


def test_is_host_allowed_wildcard_rejects_spoofed_suffix(
    data_dir: Path, audit_logger: AuditLogger
) -> None:
    """wildcard '*.example.com' rejects host with spoofed dot-suffix boundary.

    Regression: ``evil.example.com.evil.com`` must NOT match ``*.example.com``
    because the wildcard is anchored to a strict dot boundary.
    """
    sandbox = _make_executable_sandbox(
        data_dir, audit_logger, allowed_hosts=["*.example.com"]
    )
    assert sandbox.is_host_allowed("evil.example.com.evil.com") is False
    assert sandbox.is_host_allowed("sub.example.com") is True
    assert sandbox.is_host_allowed("example.com.evil.com") is False


def test_is_host_allowed_executable_strips_port(
    data_dir: Path, audit_logger: AuditLogger
) -> None:
    """executable type: port is stripped before whitelist check."""
    sandbox = _make_executable_sandbox(
        data_dir, audit_logger, allowed_hosts=["api.example.com"]
    )
    assert sandbox.is_host_allowed("api.example.com:443") is True


def test_is_host_allowed_executable_empty_denies_all(
    data_dir: Path, audit_logger: AuditLogger
) -> None:
    """executable type with empty allowed_hosts denies all hosts."""
    sandbox = _make_executable_sandbox(data_dir, audit_logger, allowed_hosts=None)
    assert sandbox.is_host_allowed("api.example.com") is False
    assert sandbox.is_host_allowed("any.host.com") is False


# ---------------------------------------------------------------------------
# SubTask 8.5: check_network_host
# ---------------------------------------------------------------------------


def test_check_network_host_raises(
    data_dir: Path, audit_logger: AuditLogger
) -> None:
    """executable type: disallowed host raises PermissionError and logs audit."""
    sandbox = _make_executable_sandbox(
        data_dir, audit_logger, allowed_hosts=["api.example.com"]
    )
    with pytest.raises(PermissionError) as exc_info:
        sandbox.check_network_host("evil.com")
    assert "cannot access host" in str(exc_info.value)
    assert "evil.com" in str(exc_info.value)

    # Audit log records the permission denial
    logs = audit_logger.read_recent_logs()
    assert any(
        log["action"] == "permission_denied"
        and log["target"] == "evil.com"
        and log["details"].get("reason") == "host_not_allowed"
        for log in logs
    ), f"expected a permission_denied audit entry for evil.com, got: {logs}"


def test_check_network_host_allowed_no_raise(
    data_dir: Path, audit_logger: AuditLogger
) -> None:
    """executable type: whitelisted host does not raise."""
    sandbox = _make_executable_sandbox(
        data_dir, audit_logger, allowed_hosts=["api.example.com"]
    )
    # Should not raise
    sandbox.check_network_host("api.example.com")
