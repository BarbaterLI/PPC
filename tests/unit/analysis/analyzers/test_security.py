"""Unit tests for the security analyzer."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

import pytest

from src.analysis.analyzers.security import (
    SecurityAnalyzer,
    _scan_string_literal,
    _shannon_entropy,
)
from src.analysis.models import AnalysisCategory, Severity


def _run(coro):
    """在 Python 3.10+ 中 get_event_loop 已被弃用，使用 asyncio.run 代替。"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("loop is running")
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


def test_shannon_entropy_uniform_is_max():
    # "abcd" (4 chars) has entropy 2.0
    assert _shannon_entropy("abcd") == pytest.approx(2.0)


def test_shannon_entropy_repeated_is_zero():
    assert _shannon_entropy("aaaa") == 0.0


def test_shannon_entropy_empty():
    assert _shannon_entropy("") == 0.0


def test_scan_string_literal_high_entropy_flagged():
    # 32 chars, all unique -> high entropy
    res = _scan_string_literal("Aa1Bb2Cc3Dd4Ee5Ff6Gg7Hh8Jj9Kk")
    assert res is not None
    assert res[0] == "high_entropy"
    assert res[1] > 4.0


def test_scan_string_literal_low_entropy_not_flagged():
    res = _scan_string_literal("aaaaaaaaaaaaaaaaaaaa")
    assert res is None


def test_scan_string_literal_short_not_flagged():
    res = _scan_string_literal("abc")
    assert res is None


# ---------------------------------------------------------------------------
# Analyzer tests
# ---------------------------------------------------------------------------


def _build_analyzer(tmp_path: Path) -> SecurityAnalyzer:
    # Use tmp_path as scan_root so the analyzer doesn't walk the full src
    # tree (which may not contain secrets but adds latency).
    return SecurityAnalyzer(scan_root=str(tmp_path))


def test_analyzer_detects_api_key_assignment(tmp_path: Path):
    sample = tmp_path / "leak.py"
    sample.write_text(
        textwrap.dedent(
            """\
            config = {
                'api_key': 'sk_live_4eC39HqLyjWDarjtT1zdp7dc',
                'normal': 'hello-world',
            }
            """
        ),
        encoding="utf-8",
    )
    analyzer = _build_analyzer(tmp_path)
    issues = _run(analyzer.analyze())
    matches = [i for i in issues if i.details.get("pattern") == "api_key_assignment"]
    assert matches, "Expected api_key_assignment to be flagged"
    assert matches[0].severity == Severity.HIGH
    assert matches[0].category == AnalysisCategory.SECURITY


def test_analyzer_detects_pickle_load(tmp_path: Path):
    sample = tmp_path / "unsafe.py"
    sample.write_text(
        textwrap.dedent(
            """\
            import pickle
            data = pickle.load(open('file.bin', 'rb'))
            """
        ),
        encoding="utf-8",
    )
    analyzer = _build_analyzer(tmp_path)
    issues = _run(analyzer.analyze())
    matches = [i for i in issues if i.details.get("pattern") == "pickle_load"]
    assert matches
    assert matches[0].severity == Severity.CRITICAL


def test_analyzer_detects_yaml_load_without_loader(tmp_path: Path):
    sample = tmp_path / "yaml_unsafe.py"
    sample.write_text(
        textwrap.dedent(
            """\
            import yaml
            data = yaml.load(stream)
            """
        ),
        encoding="utf-8",
    )
    analyzer = _build_analyzer(tmp_path)
    issues = _run(analyzer.analyze())
    matches = [i for i in issues if i.details.get("pattern") == "yaml_load_unsafe"]
    assert matches


def test_analyzer_safe_yaml_load_not_flagged(tmp_path: Path):
    sample = tmp_path / "yaml_safe.py"
    sample.write_text(
        textwrap.dedent(
            """\
            import yaml
            data = yaml.load(stream, Loader=yaml.SafeLoader)
            """
        ),
        encoding="utf-8",
    )
    analyzer = _build_analyzer(tmp_path)
    issues = _run(analyzer.analyze())
    matches = [i for i in issues if i.details.get("pattern") == "yaml_load_unsafe"]
    assert matches == []


def test_analyzer_detects_subprocess_shell_true(tmp_path: Path):
    sample = tmp_path / "cmd.py"
    sample.write_text(
        textwrap.dedent(
            """\
            import subprocess
            subprocess.run(f"echo {user_input}", shell=True)
            """
        ),
        encoding="utf-8",
    )
    analyzer = _build_analyzer(tmp_path)
    issues = _run(analyzer.analyze())
    matches = [i for i in issues if i.details.get("pattern") == "subprocess_shell"]
    assert matches
    assert matches[0].severity == Severity.CRITICAL


def test_analyzer_detects_subprocess_string_command(tmp_path: Path):
    sample = tmp_path / "cmd2.py"
    sample.write_text(
        textwrap.dedent(
            """\
            import subprocess
            subprocess.run("rm -rf /")
            """
        ),
        encoding="utf-8",
    )
    analyzer = _build_analyzer(tmp_path)
    issues = _run(analyzer.analyze())
    matches = [i for i in issues if i.details.get("pattern") == "subprocess_string_cmd"]
    assert matches


def test_analyzer_detects_pem_private_key(tmp_path: Path):
    sample = tmp_path / "key.py"
    sample.write_text(
        "CERT = '-----BEGIN RSA PRIVATE KEY-----\\nABC'",
        encoding="utf-8",
    )
    analyzer = _build_analyzer(tmp_path)
    issues = _run(analyzer.analyze())
    matches = [i for i in issues if i.details.get("pattern") == "private_key_pem"]
    assert matches


def test_analyzer_detects_high_entropy_string(tmp_path: Path):
    # A long random-looking string literal should be flagged by entropy.
    secret = "x9Q" + "abcdef0123456789ABCDEF" * 4
    sample = tmp_path / "entropy.py"
    sample.write_text(f'TOKEN = "{secret}"\n', encoding="utf-8")
    analyzer = _build_analyzer(tmp_path)
    issues = _run(analyzer.analyze())
    matches = [i for i in issues if i.details.get("kind") == "entropy"]
    assert matches


def test_analyzer_empty_directory_returns_no_issues(tmp_path: Path):
    analyzer = _build_analyzer(tmp_path)
    issues = _run(analyzer.analyze())
    assert issues == []


def test_analyzer_handles_missing_scan_root(tmp_path: Path):
    analyzer = SecurityAnalyzer(scan_root=str(tmp_path / "nope"))
    issues = _run(analyzer.analyze())
    assert issues == []


def test_analyzer_inline_sources(tmp_path: Path):
    analyzer = SecurityAnalyzer(scan_root=str(tmp_path / "nope"))
    issues = _run(
        analyzer.analyze(context={"inline_sources": {"virtual.py": "password = 'hunter2-supersecret-12345'\n"}})
    )
    matches = [i for i in issues if i.details.get("pattern") == "password_assignment"]
    assert matches


def test_analyzer_ignores_hex_blob_in_comment(tmp_path: Path):
    sample = tmp_path / "comment.py"
    sample.write_text("# some long hex: 0123456789abcdef0123456789abcdef\n", encoding="utf-8")
    analyzer = _build_analyzer(tmp_path)
    issues = _run(analyzer.analyze())
    matches = [i for i in issues if i.details.get("pattern") == "hex_blob"]
    assert matches == []
