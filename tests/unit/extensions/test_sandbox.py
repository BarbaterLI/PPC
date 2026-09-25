"""Unit tests for :mod:`src.extensions.sandbox`.

Covers the :class:`SandboxContext` permission enforcement, path containment,
file/subprocess/network/config proxies and audit-log integration.
"""

from __future__ import annotations

import logging
import sys
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


@pytest.fixture
def strict_sandbox(data_dir: Path, audit_logger: AuditLogger) -> SandboxContext:
    """SandboxContext in STRICT mode with NO permissions."""
    return SandboxContext(SandboxConfig(
        extension_name="test_ext",
        permissions=[],
        audit_logger=audit_logger,
        security_level=SecurityLevel.STRICT,
        data_dir=data_dir,
    ))


def _make_sandbox(
    data_dir: Path,
    audit_logger: AuditLogger,
    permissions: list[ExtensionPermission],
    *,
    security_level: SecurityLevel = SecurityLevel.STRICT,
    allowed_paths: list[str | Path] | None = None,
) -> SandboxContext:
    return SandboxContext(SandboxConfig(
        extension_name="test_ext",
        permissions=permissions,
        audit_logger=audit_logger,
        security_level=security_level,
        data_dir=data_dir,
        allowed_paths=allowed_paths,
    ))


# ---------------------------------------------------------------------------
# _check_permission across security levels
# ---------------------------------------------------------------------------


def test_check_permission_strict_raises(strict_sandbox: SandboxContext) -> None:
    """STRICT mode raises PermissionError when permission is missing."""
    with pytest.raises(PermissionError):
        strict_sandbox._check_permission(ExtensionPermission.NETWORK, "test_target")


def test_check_permission_warn_allows(
    data_dir: Path, audit_logger: AuditLogger, caplog: pytest.LogCaptureFixture
) -> None:
    """WARN mode allows the operation but logs a warning."""
    sandbox = _make_sandbox(
        data_dir, audit_logger, [], security_level=SecurityLevel.WARN
    )
    with caplog.at_level(logging.WARNING):
        result = sandbox._check_permission(ExtensionPermission.NETWORK, "test_target")
    assert result is True
    assert any(
        "missing permission" in rec.message for rec in caplog.records
    ), "WARN mode should emit a missing-permission warning"


def test_check_permission_allow_allows(
    data_dir: Path, audit_logger: AuditLogger
) -> None:
    """ALLOW mode allows the operation without raising."""
    sandbox = _make_sandbox(
        data_dir, audit_logger, [], security_level=SecurityLevel.ALLOW
    )
    result = sandbox._check_permission(ExtensionPermission.NETWORK, "test_target")
    assert result is True


def test_warning_dedup(
    data_dir: Path, audit_logger: AuditLogger, caplog: pytest.LogCaptureFixture
) -> None:
    """WARN mode only warns once per permission:target key."""
    sandbox = _make_sandbox(
        data_dir, audit_logger, [], security_level=SecurityLevel.WARN
    )
    with caplog.at_level(logging.WARNING):
        sandbox._check_permission(ExtensionPermission.NETWORK, "test_target")
        sandbox._check_permission(ExtensionPermission.NETWORK, "test_target")
        sandbox._check_permission(ExtensionPermission.NETWORK, "test_target")

    warning_count = sum(
        1 for rec in caplog.records if "missing permission" in rec.message
    )
    assert warning_count == 1, f"expected exactly one warning, got {warning_count}"


# ---------------------------------------------------------------------------
# Path containment
# ---------------------------------------------------------------------------


def test_is_path_allowed(
    strict_sandbox: SandboxContext, data_dir: Path, tmp_path: Path
) -> None:
    """Path inside data_dir is allowed; path outside is blocked."""
    assert strict_sandbox._is_path_allowed(data_dir / "file.txt") is True
    outside = tmp_path / "outside.txt"
    assert strict_sandbox._is_path_allowed(outside) is False


def test_path_traversal_blocked(strict_sandbox: SandboxContext) -> None:
    """Relative path traversal (../../etc/passwd) is blocked."""
    with pytest.raises(PermissionError):
        strict_sandbox.safe_open("../../etc/passwd", "r")


# ---------------------------------------------------------------------------
# safe_open / safe_read_file / safe_write_file
# ---------------------------------------------------------------------------


def test_safe_open_read(data_dir: Path, audit_logger: AuditLogger) -> None:
    """Reading a file with FILESYSTEM_READ permission works."""
    sandbox = _make_sandbox(data_dir, audit_logger, [ExtensionPermission.FILESYSTEM_READ])
    test_file = data_dir / "test.txt"
    test_file.write_text("hello", encoding="utf-8")
    with sandbox.safe_open(test_file, "r") as f:
        content = f.read()
    assert content == "hello"


def test_safe_open_write_blocked_without_permission(
    strict_sandbox: SandboxContext, data_dir: Path
) -> None:
    """Writing without FILESYSTEM_WRITE raises PermissionError."""
    with pytest.raises(PermissionError):
        strict_sandbox.safe_open(data_dir / "test.txt", "w")


def test_safe_open_write_with_permission(
    data_dir: Path, audit_logger: AuditLogger
) -> None:
    """Writing with FILESYSTEM_WRITE permission works."""
    sandbox = _make_sandbox(
        data_dir, audit_logger, [ExtensionPermission.FILESYSTEM_WRITE]
    )
    with sandbox.safe_open(data_dir / "test.txt", "w") as f:
        f.write("hello")
    assert (data_dir / "test.txt").read_text(encoding="utf-8") == "hello"


def test_safe_write_file_creates_directory_after_permission(
    strict_sandbox: SandboxContext, data_dir: Path
) -> None:
    """Directory creation happens AFTER the permission check (regression)."""
    new_subdir = data_dir / "new_subdir"
    target_file = new_subdir / "file.txt"
    with pytest.raises(PermissionError):
        strict_sandbox.safe_write_file(target_file, "content")
    # The directory must NOT have been created (permission check ran first).
    assert not new_subdir.exists(), (
        "Directory was created before the permission check succeeded - "
        "this is the bug we fixed"
    )


def test_safe_read_file(data_dir: Path, audit_logger: AuditLogger) -> None:
    """Reading file content works with permission."""
    sandbox = _make_sandbox(data_dir, audit_logger, [ExtensionPermission.FILESYSTEM_READ])
    test_file = data_dir / "test.txt"
    test_file.write_text("hello world", encoding="utf-8")
    content = sandbox.safe_read_file(test_file)
    assert content == "hello world"


def test_safe_write_file(data_dir: Path, audit_logger: AuditLogger) -> None:
    """Writing file content works with permission."""
    sandbox = _make_sandbox(
        data_dir, audit_logger, [ExtensionPermission.FILESYSTEM_WRITE]
    )
    target = data_dir / "out.txt"
    sandbox.safe_write_file(target, "hello world")
    assert target.read_text(encoding="utf-8") == "hello world"


def test_safe_list_dir(data_dir: Path, audit_logger: AuditLogger) -> None:
    """Listing directory works with FILESYSTEM_READ permission."""
    sandbox = _make_sandbox(data_dir, audit_logger, [ExtensionPermission.FILESYSTEM_READ])
    (data_dir / "a.txt").write_text("a", encoding="utf-8")
    (data_dir / "b.txt").write_text("b", encoding="utf-8")
    entries = sandbox.safe_list_dir()
    names = sorted(p.name for p in entries)
    assert "a.txt" in names
    assert "b.txt" in names


def test_safe_delete_file(data_dir: Path, audit_logger: AuditLogger) -> None:
    """Deleting requires FILESYSTEM_DELETE permission."""
    sandbox = _make_sandbox(
        data_dir, audit_logger, [ExtensionPermission.FILESYSTEM_DELETE]
    )
    target = data_dir / "to_delete.txt"
    target.write_text("delete me", encoding="utf-8")
    assert target.exists()
    sandbox.safe_delete_file(target)
    assert not target.exists()

    # Without permission, raises PermissionError
    sandbox_no_perm = _make_sandbox(data_dir, audit_logger, [])
    target2 = data_dir / "to_delete2.txt"
    target2.write_text("delete me", encoding="utf-8")
    with pytest.raises(PermissionError):
        sandbox_no_perm.safe_delete_file(target2)
    assert target2.exists()


# ---------------------------------------------------------------------------
# safe_subprocess
# ---------------------------------------------------------------------------


def test_safe_subprocess_with_permission(
    data_dir: Path, audit_logger: AuditLogger
) -> None:
    """Subprocess works with SUBPROCESS permission."""
    sandbox = _make_sandbox(data_dir, audit_logger, [ExtensionPermission.SUBPROCESS])
    result = sandbox.safe_subprocess([sys.executable, "-c", "print('hello')"])
    assert result.returncode == 0


def test_safe_subprocess_without_permission(strict_sandbox: SandboxContext) -> None:
    """Subprocess blocked without SUBPROCESS permission."""
    with pytest.raises(PermissionError):
        strict_sandbox.safe_subprocess([sys.executable, "--version"])


def test_safe_subprocess_shell_true_logs_warning(
    data_dir: Path, audit_logger: AuditLogger, caplog: pytest.LogCaptureFixture
) -> None:
    """shell=True logs a critical warning."""
    sandbox = _make_sandbox(data_dir, audit_logger, [ExtensionPermission.SUBPROCESS])
    with caplog.at_level(logging.CRITICAL, logger="src.extensions.sandbox"):
        sandbox.safe_subprocess("echo hello", shell=True)
    assert any(
        "shell=True" in rec.message for rec in caplog.records
    ), "expected a critical log mentioning shell=True"


# ---------------------------------------------------------------------------
# Network / config checks
# ---------------------------------------------------------------------------


def test_check_network_access(
    data_dir: Path, audit_logger: AuditLogger
) -> None:
    """Network access check works."""
    # With permission -> True
    sandbox = _make_sandbox(data_dir, audit_logger, [ExtensionPermission.NETWORK])
    assert sandbox.check_network_access("http://example.com") is True

    # STRICT without permission -> raises
    strict = _make_sandbox(data_dir, audit_logger, [])
    with pytest.raises(PermissionError):
        strict.check_network_access("http://example.com")

    # WARN without permission -> True
    warn = _make_sandbox(data_dir, audit_logger, [], security_level=SecurityLevel.WARN)
    assert warn.check_network_access("http://example.com", method="POST") is True


def test_check_config_read_write(data_dir: Path, audit_logger: AuditLogger) -> None:
    """Config read/write checks work."""
    sandbox_read = _make_sandbox(data_dir, audit_logger, [ExtensionPermission.CONFIG_READ])
    assert sandbox_read.check_config_read("key") is True

    sandbox_write = _make_sandbox(
        data_dir, audit_logger, [ExtensionPermission.CONFIG_WRITE]
    )
    assert sandbox_write.check_config_write("key") is True

    sandbox_none = _make_sandbox(data_dir, audit_logger, [])
    with pytest.raises(PermissionError):
        sandbox_none.check_config_read("key")
    with pytest.raises(PermissionError):
        sandbox_none.check_config_write("key")


# ---------------------------------------------------------------------------
# add_allowed_path
# ---------------------------------------------------------------------------


def test_add_allowed_path(
    data_dir: Path, audit_logger: AuditLogger, tmp_path: Path
) -> None:
    """Additional allowed paths work."""
    sandbox = _make_sandbox(data_dir, audit_logger, [ExtensionPermission.FILESYSTEM_READ])
    extra_dir = tmp_path / "extra"
    extra_dir.mkdir()
    extra_file = extra_dir / "file.txt"
    extra_file.write_text("hello", encoding="utf-8")

    # Initially the extra path is not allowed
    assert sandbox._is_path_allowed(extra_file) is False
    sandbox.add_allowed_path(extra_dir)
    assert sandbox._is_path_allowed(extra_file) is True


# ---------------------------------------------------------------------------
# Audit log integration
# ---------------------------------------------------------------------------


def test_audit_log_on_success(data_dir: Path, tmp_path: Path) -> None:
    """Audit log records successful operations."""
    audit_logger = AuditLogger(log_dir=tmp_path / "audit")
    sandbox = _make_sandbox(
        data_dir, audit_logger, [ExtensionPermission.FILESYSTEM_READ]
    )
    test_file = data_dir / "test.txt"
    test_file.write_text("hello", encoding="utf-8")
    with sandbox.safe_open(test_file, "r") as f:
        f.read()

    logs = audit_logger.read_recent_logs()
    assert any(
        log["action"] == "file_read" and log["result"] == "success" for log in logs
    ), f"expected a successful file_read entry, got: {logs}"


def test_audit_log_on_denied(data_dir: Path, tmp_path: Path) -> None:
    """Audit log records denied operations."""
    audit_logger = AuditLogger(log_dir=tmp_path / "audit")
    sandbox = _make_sandbox(data_dir, audit_logger, [])
    with pytest.raises(PermissionError):
        sandbox.safe_open(data_dir / "test.txt", "w")

    logs = audit_logger.read_recent_logs()
    assert any(
        log["result"] == "denied" for log in logs
    ), f"expected a denied entry, got: {logs}"
