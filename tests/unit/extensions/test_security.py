"""Unit tests for :mod:`src.extensions.security`.

Covers the :class:`AuditLogger` (constructor injection, file/network/subprocess/config/
permission-denied log helpers, recent-log reader robustness) and the
:class:`AuditEntry` dataclass serialisation helpers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from src.extensions.security import (
    AuditAction,
    AuditEntry,
    AuditLogger,
    AuditResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def audit_logger(tmp_path: Path) -> AuditLogger:
    """Fresh AuditLogger writing to an isolated tmp directory.

    Uses the public constructor (constructor injection, no shared state)
    so tests don't interfere with other instances.
    """
    return AuditLogger(log_dir=tmp_path / "audit")


# ---------------------------------------------------------------------------
# Constructor injection
# ---------------------------------------------------------------------------


def test_independent_instances(tmp_path: Path) -> None:
    """Each AuditLogger() constructs an independent instance (no singleton)."""
    i1 = AuditLogger(log_dir=tmp_path / "single")
    i2 = AuditLogger(log_dir=tmp_path / "single")
    assert i1 is not i2
    assert not hasattr(AuditLogger, "get_instance")


# ---------------------------------------------------------------------------
# log_file_access
# ---------------------------------------------------------------------------


def test_log_file_access_read(audit_logger: AuditLogger) -> None:
    """log_file_access with "read" operation creates FILE_READ entry."""
    audit_logger.log_file_access("ext", "/path/file", "read", success=True)
    logs = audit_logger.read_recent_logs()
    assert len(logs) == 1
    entry = logs[0]
    assert entry["action"] == AuditAction.FILE_READ.value
    assert entry["extension_name"] == "ext"
    assert entry["target"] == "/path/file"
    assert entry["result"] == AuditResult.SUCCESS.value


def test_log_file_access_write(audit_logger: AuditLogger) -> None:
    """log_file_access with "write" creates FILE_WRITE entry."""
    audit_logger.log_file_access("ext", "/path/file", "write", success=True)
    logs = audit_logger.read_recent_logs()
    assert logs[0]["action"] == AuditAction.FILE_WRITE.value


def test_log_file_access_delete(audit_logger: AuditLogger) -> None:
    """log_file_access with "delete" creates FILE_DELETE entry."""
    audit_logger.log_file_access("ext", "/path/file", "delete", success=True)
    logs = audit_logger.read_recent_logs()
    assert logs[0]["action"] == AuditAction.FILE_DELETE.value


def test_log_file_access_unknown_op(audit_logger: AuditLogger) -> None:
    """Unknown operation defaults to FILE_READ."""
    audit_logger.log_file_access("ext", "/path/file", "unknown_op", success=True)
    logs = audit_logger.read_recent_logs()
    assert logs[0]["action"] == AuditAction.FILE_READ.value


# ---------------------------------------------------------------------------
# log_network_access / log_subprocess / log_config_access
# ---------------------------------------------------------------------------


def test_log_network_access(audit_logger: AuditLogger) -> None:
    """log_network_access creates entry with method in details."""
    audit_logger.log_network_access(
        "ext", "http://example.com", method="POST", success=True
    )
    logs = audit_logger.read_recent_logs()
    assert len(logs) == 1
    entry = logs[0]
    assert entry["action"] == AuditAction.NETWORK_ACCESS.value
    assert entry["target"] == "http://example.com"
    assert entry["details"]["method"] == "POST"


def test_log_subprocess(audit_logger: AuditLogger) -> None:
    """log_subprocess creates entry."""
    audit_logger.log_subprocess("ext", "ls -la", success=True, details={"rc": 0})
    logs = audit_logger.read_recent_logs()
    assert len(logs) == 1
    entry = logs[0]
    assert entry["action"] == AuditAction.SUBPROCESS_EXEC.value
    assert entry["target"] == "ls -la"
    assert entry["details"]["rc"] == 0


def test_log_config_access_read(audit_logger: AuditLogger) -> None:
    """log_config_access with "read" creates CONFIG_READ."""
    audit_logger.log_config_access("ext", "my_key", operation="read", success=True)
    logs = audit_logger.read_recent_logs()
    assert logs[0]["action"] == AuditAction.CONFIG_READ.value


def test_log_config_access_write(audit_logger: AuditLogger) -> None:
    """log_config_access with "write" creates CONFIG_WRITE."""
    audit_logger.log_config_access("ext", "my_key", operation="write", success=True)
    logs = audit_logger.read_recent_logs()
    assert logs[0]["action"] == AuditAction.CONFIG_WRITE.value


# ---------------------------------------------------------------------------
# log_permission_denied / log_event
# ---------------------------------------------------------------------------


def test_log_permission_denied(audit_logger: AuditLogger) -> None:
    """log_permission_denied creates DENIED entry."""
    audit_logger.log_permission_denied(
        "ext", "network", "http://example.com", details={"method": "GET"}
    )
    logs = audit_logger.read_recent_logs()
    assert len(logs) == 1
    entry = logs[0]
    assert entry["action"] == AuditAction.PERMISSION_DENIED.value
    assert entry["result"] == AuditResult.DENIED.value
    assert entry["details"]["permission"] == "network"
    assert entry["details"]["method"] == "GET"


def test_log_event(audit_logger: AuditLogger) -> None:
    """log_event creates generic entry."""
    audit_logger.log_event(
        "ext", "EXTENSION_LOAD", "ext:v1", AuditResult.SUCCESS, details={"k": "v"}
    )
    logs = audit_logger.read_recent_logs()
    assert len(logs) == 1
    entry = logs[0]
    assert entry["action"] == "EXTENSION_LOAD"
    assert entry["target"] == "ext:v1"
    assert entry["result"] == AuditResult.SUCCESS.value
    assert entry["details"]["k"] == "v"


# ---------------------------------------------------------------------------
# read_recent_logs
# ---------------------------------------------------------------------------


def test_read_recent_logs(audit_logger: AuditLogger) -> None:
    """read_recent_logs returns entries in reverse order."""
    for i in range(3):
        audit_logger.log_event(
            "ext", f"ACT_{i}", f"target_{i}", AuditResult.SUCCESS
        )
    logs = audit_logger.read_recent_logs()
    assert len(logs) == 3
    # newest first
    assert logs[0]["action"] == "ACT_2"
    assert logs[1]["action"] == "ACT_1"
    assert logs[2]["action"] == "ACT_0"


def test_read_recent_logs_missing_file(tmp_path: Path) -> None:
    """Missing log file returns empty list."""
    logger = AuditLogger(log_dir=tmp_path / "fresh")
    # Don't write anything; log file doesn't exist yet
    assert not logger.get_log_file_path().exists()
    assert logger.read_recent_logs() == []


def test_read_recent_logs_malformed_json(audit_logger: AuditLogger) -> None:
    """Malformed JSON lines are skipped."""
    log_file = audit_logger.get_log_file_path()
    with open(log_file, "w", encoding="utf-8") as f:
        f.write('{"action": "A"}\n')
        f.write("THIS IS NOT JSON\n")
        f.write('{"action": "B"}\n')
        f.write("{broken json\n")
    logs = audit_logger.read_recent_logs()
    # Only 2 valid lines, in reverse order
    assert len(logs) == 2
    assert logs[0]["action"] == "B"
    assert logs[1]["action"] == "A"


# ---------------------------------------------------------------------------
# AuditEntry serialisation
# ---------------------------------------------------------------------------


def test_audit_entry_to_dict() -> None:
    """AuditEntry.to_dict() works."""
    entry = AuditEntry(
        timestamp="2024-01-01T00:00:00Z",
        extension_name="ext",
        action="file_read",
        target="/path/file",
        result="success",
        details={"k": "v"},
    )
    d = entry.to_dict()
    assert d["timestamp"] == "2024-01-01T00:00:00Z"
    assert d["extension_name"] == "ext"
    assert d["action"] == "file_read"
    assert d["target"] == "/path/file"
    assert d["result"] == "success"
    assert d["details"] == {"k": "v"}


def test_audit_entry_to_json() -> None:
    """AuditEntry.to_json() produces valid JSON."""
    entry = AuditEntry(
        timestamp="2024-01-01T00:00:00Z",
        extension_name="ext",
        action="file_read",
        target="/path/file",
        result="success",
        details={"k": "v"},
    )
    s = entry.to_json()
    d: dict[str, Any] = json.loads(s)
    assert d["action"] == "file_read"
    assert d["extension_name"] == "ext"
    assert d["target"] == "/path/file"
    assert d["result"] == "success"
    assert d["details"]["k"] == "v"


# ---------------------------------------------------------------------------
# Denied-warning emission
# ---------------------------------------------------------------------------


def test_denied_warning_emitted(
    audit_logger: AuditLogger, caplog: pytest.LogCaptureFixture
) -> None:
    """DENIED result triggers a warning log."""
    with caplog.at_level(logging.WARNING, logger="ppc.extensions.audit"):
        audit_logger.log_permission_denied("ext", "network", "http://example.com")
    assert any(
        "denied" in rec.message.lower() and "Audit" in rec.message
        for rec in caplog.records
    ), f"expected an Audit denied warning, got: {[r.message for r in caplog.records]}"
