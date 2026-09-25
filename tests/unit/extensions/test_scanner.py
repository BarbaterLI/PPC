"""Unit tests for :mod:`src.extensions.scanner`.

Covers the :class:`ExtensionSecurityScanner` AST/regex rules, the
:class:`SecurityScanResult` risk-score and permission-tracking helpers, and
the file/directory scanning entry-points.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.extensions.scanner import (
    ExtensionSecurityScanner,
    SecurityIssue,
    SecurityScanResult,
    SecuritySeverity,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def scanner() -> ExtensionSecurityScanner:
    return ExtensionSecurityScanner()


def _write(tmp_path: Path, name: str, source: str) -> Path:
    f = tmp_path / name
    f.write_text(source, encoding="utf-8")
    return f


def _has_issue(
    result: SecurityScanResult,
    *,
    rule_id: str | None = None,
    severity: SecuritySeverity | None = None,
) -> bool:
    for issue in result.issues:
        if rule_id is not None and issue.rule_id != rule_id:
            continue
        if severity is not None and issue.severity != severity:
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# Dangerous imports
# ---------------------------------------------------------------------------


def test_scan_pickle_import(tmp_path: Path, scanner: ExtensionSecurityScanner) -> None:
    """`import pickle` detected as CRITICAL."""
    f = _write(tmp_path, "test.py", "import pickle\n")
    result = scanner.scan_file(f)
    assert _has_issue(result, rule_id="DANGEROUS_IMPORT_PICKLE", severity=SecuritySeverity.CRITICAL)


def test_scan_pickle_loads(tmp_path: Path, scanner: ExtensionSecurityScanner) -> None:
    """`pickle.loads()` detected as CRITICAL."""
    f = _write(tmp_path, "test.py", "import pickle\npickle.loads(data)\n")
    result = scanner.scan_file(f)
    assert _has_issue(result, rule_id="PICKLE_LOADS", severity=SecuritySeverity.CRITICAL)


def test_scan_os_system(tmp_path: Path, scanner: ExtensionSecurityScanner) -> None:
    """`os.system()` detected as HIGH."""
    f = _write(tmp_path, "test.py", "import os\nos.system('ls')\n")
    result = scanner.scan_file(f)
    assert _has_issue(result, rule_id="OS_SYSTEM", severity=SecuritySeverity.HIGH)


def test_scan_eval(tmp_path: Path, scanner: ExtensionSecurityScanner) -> None:
    """`eval()` detected as CRITICAL."""
    f = _write(tmp_path, "test.py", "x = eval('1+1')\n")
    result = scanner.scan_file(f)
    assert _has_issue(result, rule_id="EVAL", severity=SecuritySeverity.CRITICAL)


def test_scan_exec(tmp_path: Path, scanner: ExtensionSecurityScanner) -> None:
    """`exec()` detected as CRITICAL."""
    f = _write(tmp_path, "test.py", "exec('x = 1')\n")
    result = scanner.scan_file(f)
    assert _has_issue(result, rule_id="EXEC", severity=SecuritySeverity.CRITICAL)


# ---------------------------------------------------------------------------
# Subprocess rules
# ---------------------------------------------------------------------------


def test_scan_subprocess_shell_true(
    tmp_path: Path, scanner: ExtensionSecurityScanner
) -> None:
    """`subprocess.run(..., shell=True)` detected as CRITICAL."""
    f = _write(
        tmp_path, "test.py", "import subprocess\nsubprocess.run('ls', shell=True)\n"
    )
    result = scanner.scan_file(f)
    assert _has_issue(
        result, rule_id="SUBPROCESS_SHELL_TRUE", severity=SecuritySeverity.CRITICAL
    )


def test_scan_subprocess_without_shell(
    tmp_path: Path, scanner: ExtensionSecurityScanner
) -> None:
    """`subprocess.run()` without shell detected as MEDIUM."""
    f = _write(tmp_path, "test.py", "import subprocess\nsubprocess.run(['ls'])\n")
    result = scanner.scan_file(f)
    assert _has_issue(
        result, rule_id="SUBPROCESS_CALL", severity=SecuritySeverity.MEDIUM
    )


# ---------------------------------------------------------------------------
# Other dangerous patterns
# ---------------------------------------------------------------------------


def test_scan_unsafe_yaml_load(
    tmp_path: Path, scanner: ExtensionSecurityScanner
) -> None:
    """`yaml.load()` without Loader detected as HIGH."""
    f = _write(tmp_path, "test.py", "import yaml\nyaml.load(data)\n")
    result = scanner.scan_file(f)
    assert _has_issue(result, rule_id="UNSAFE_YAML_LOAD", severity=SecuritySeverity.HIGH)


def test_scan_ctypes_import(
    tmp_path: Path, scanner: ExtensionSecurityScanner
) -> None:
    """`import ctypes` detected as HIGH."""
    f = _write(tmp_path, "test.py", "import ctypes\n")
    result = scanner.scan_file(f)
    assert _has_issue(result, severity=SecuritySeverity.HIGH)


def test_scan_socket_import(
    tmp_path: Path, scanner: ExtensionSecurityScanner
) -> None:
    """`import socket` detected as MEDIUM and requires NETWORK permission."""
    f = _write(tmp_path, "test.py", "import socket\n")
    result = scanner.scan_file(f)
    assert _has_issue(result, severity=SecuritySeverity.MEDIUM)
    assert "NETWORK" in result.required_permissions


def test_scan_os_remove(tmp_path: Path, scanner: ExtensionSecurityScanner) -> None:
    """`os.remove()` detected as MEDIUM."""
    f = _write(tmp_path, "test.py", "import os\nos.remove('foo')\n")
    result = scanner.scan_file(f)
    assert _has_issue(result, rule_id="FILE_DELETE", severity=SecuritySeverity.MEDIUM)


def test_scan_shutil_rmtree(
    tmp_path: Path, scanner: ExtensionSecurityScanner
) -> None:
    """`shutil.rmtree()` detected as HIGH."""
    f = _write(tmp_path, "test.py", "import shutil\nshutil.rmtree('foo')\n")
    result = scanner.scan_file(f)
    assert _has_issue(
        result, rule_id="DIRECTORY_DELETE", severity=SecuritySeverity.HIGH
    )


def test_scan_os_execv(tmp_path: Path, scanner: ExtensionSecurityScanner) -> None:
    """`os.execv()` detected as HIGH."""
    f = _write(tmp_path, "test.py", "import os\nos.execv('/bin/ls', ['ls'])\n")
    result = scanner.scan_file(f)
    assert _has_issue(
        result, rule_id="PROCESS_REPLACE", severity=SecuritySeverity.HIGH
    )


def test_scan_pty_spawn(tmp_path: Path, scanner: ExtensionSecurityScanner) -> None:
    """`pty.spawn()` detected as HIGH."""
    f = _write(tmp_path, "test.py", "import pty\npty.spawn(['ls'])\n")
    result = scanner.scan_file(f)
    assert _has_issue(
        result, rule_id="SUBPROCESS_SPAWN", severity=SecuritySeverity.HIGH
    )


def test_scan_code_interact(
    tmp_path: Path, scanner: ExtensionSecurityScanner
) -> None:
    """`code.interact()` detected as HIGH."""
    f = _write(tmp_path, "test.py", "import code\ncode.interact()\n")
    result = scanner.scan_file(f)
    assert _has_issue(
        result, rule_id="INTERACTIVE_EXEC", severity=SecuritySeverity.HIGH
    )


def test_scan_importlib_import_module(
    tmp_path: Path, scanner: ExtensionSecurityScanner
) -> None:
    """`importlib.import_module()` detected as MEDIUM."""
    f = _write(
        tmp_path, "test.py", "import importlib\nimportlib.import_module('os')\n"
    )
    result = scanner.scan_file(f)
    assert _has_issue(
        result, rule_id="DYNAMIC_IMPORT", severity=SecuritySeverity.MEDIUM
    )


# ---------------------------------------------------------------------------
# File-write detection
# ---------------------------------------------------------------------------


def test_scan_file_write_modes(
    tmp_path: Path, scanner: ExtensionSecurityScanner
) -> None:
    """open() with write mode requires FILESYSTEM_WRITE permission."""
    f = _write(tmp_path, "test.py", "with open('foo.txt', 'w') as f:\n    f.write('hi')\n")
    result = scanner.scan_file(f)
    assert _has_issue(result, rule_id="FILE_WRITE", severity=SecuritySeverity.MEDIUM)
    assert "FILESYSTEM_WRITE" in result.required_permissions


def test_scan_file_write_binary_modes(
    tmp_path: Path, scanner: ExtensionSecurityScanner
) -> None:
    """open() with binary write modes (xb, x+, w+b) detected."""
    f = _write(
        tmp_path,
        "test.py",
        "open('a', 'xb')\nopen('b', 'x+')\nopen('c', 'w+b')\n",
    )
    result = scanner.scan_file(f)
    file_write_issues = [i for i in result.issues if i.rule_id == "FILE_WRITE"]
    assert len(file_write_issues) == 3, (
        f"expected 3 FILE_WRITE issues, got {len(file_write_issues)}"
    )


# ---------------------------------------------------------------------------
# Clean code
# ---------------------------------------------------------------------------


def test_scan_clean_code(
    tmp_path: Path, scanner: ExtensionSecurityScanner
) -> None:
    """Clean code with no dangerous patterns produces empty results."""
    f = _write(tmp_path, "test.py", "def add(a, b):\n    return a + b\n")
    result = scanner.scan_file(f)
    assert result.issues == []
    assert result.risk_score == 0


# ---------------------------------------------------------------------------
# Risk score / has_critical_or_high
# ---------------------------------------------------------------------------


def test_calculate_risk_score() -> None:
    """Risk score calculation works correctly."""
    result = SecurityScanResult(file_path=Path("test.py"))
    result.issues = [
        SecurityIssue(SecuritySeverity.LOW, "low", "low", "loc", "sug"),
        SecurityIssue(SecuritySeverity.MEDIUM, "med", "med", "loc", "sug"),
        SecurityIssue(SecuritySeverity.HIGH, "high", "high", "loc", "sug"),
        SecurityIssue(SecuritySeverity.CRITICAL, "crit", "crit", "loc", "sug"),
    ]
    result.calculate_risk_score()
    # weights: LOW=1, MEDIUM=5, HIGH=20, CRITICAL=50  ->  76
    assert result.risk_score == 76
    assert result.summary["low"] == 1
    assert result.summary["medium"] == 1
    assert result.summary["high"] == 1
    assert result.summary["critical"] == 1


def test_has_critical_or_high() -> None:
    """has_critical_or_high returns True when CRITICAL/HIGH present, False otherwise."""
    # Only LOW -> False
    result_low = SecurityScanResult(file_path=Path("test.py"))
    result_low.issues = [
        SecurityIssue(SecuritySeverity.LOW, "low", "low", "loc", "sug"),
    ]
    result_low.calculate_risk_score()
    assert result_low.has_critical_or_high() is False

    # Has HIGH -> True
    result_high = SecurityScanResult(file_path=Path("test.py"))
    result_high.issues = [
        SecurityIssue(SecuritySeverity.HIGH, "high", "high", "loc", "sug"),
        SecurityIssue(SecuritySeverity.LOW, "low", "low", "loc", "sug"),
    ]
    result_high.calculate_risk_score()
    assert result_high.has_critical_or_high() is True

    # Has CRITICAL -> True
    result_crit = SecurityScanResult(file_path=Path("test.py"))
    result_crit.issues = [
        SecurityIssue(SecuritySeverity.CRITICAL, "crit", "crit", "loc", "sug"),
    ]
    result_crit.calculate_risk_score()
    assert result_crit.has_critical_or_high() is True


# ---------------------------------------------------------------------------
# scan_file / scan_directory
# ---------------------------------------------------------------------------


def test_scan_file(tmp_path: Path, scanner: ExtensionSecurityScanner) -> None:
    """scan_file reads and scans a file."""
    f = _write(tmp_path, "test.py", "import os\nos.system('ls')\n")
    result = scanner.scan_file(f)
    assert result.file_path == f
    assert len(result.issues) > 0
    assert result.risk_score > 0


def test_scan_file_missing(
    tmp_path: Path, scanner: ExtensionSecurityScanner
) -> None:
    """scan_file handles missing file gracefully."""
    result = scanner.scan_file(tmp_path / "nonexistent.py")
    assert result.issues == []
    assert result.risk_score == 0


def test_scan_directory(tmp_path: Path, scanner: ExtensionSecurityScanner) -> None:
    """scan_directory scans all .py files (skips underscore-prefixed)."""
    _write(tmp_path, "a.py", "import pickle\n")
    _write(tmp_path, "b.py", "import os\nos.system('ls')\n")
    # Underscore-prefixed files should be skipped by scan_directory
    _write(tmp_path, "_skip.py", "import pickle\n")
    results = scanner.scan_directory(tmp_path)
    file_names = sorted(r.file_path.name for r in results)
    assert file_names == ["a.py", "b.py"], f"unexpected files: {file_names}"


# ---------------------------------------------------------------------------
# Required permissions tracking
# ---------------------------------------------------------------------------


def test_required_permissions_network(
    tmp_path: Path, scanner: ExtensionSecurityScanner
) -> None:
    """Network imports add NETWORK to required_permissions."""
    f = _write(tmp_path, "test.py", "import socket\n")
    result = scanner.scan_file(f)
    assert "NETWORK" in result.required_permissions


def test_required_permissions_subprocess(
    tmp_path: Path, scanner: ExtensionSecurityScanner
) -> None:
    """subprocess adds SUBPROCESS to required_permissions."""
    f = _write(tmp_path, "test.py", "import subprocess\nsubprocess.run(['ls'])\n")
    result = scanner.scan_file(f)
    assert "SUBPROCESS" in result.required_permissions


def test_required_permissions_filesystem_write(
    tmp_path: Path, scanner: ExtensionSecurityScanner
) -> None:
    """File write adds FILESYSTEM_WRITE to required_permissions."""
    f = _write(tmp_path, "test.py", "open('foo.txt', 'w')\n")
    result = scanner.scan_file(f)
    assert "FILESYSTEM_WRITE" in result.required_permissions
