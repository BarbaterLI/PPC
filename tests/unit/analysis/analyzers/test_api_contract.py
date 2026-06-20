"""Unit tests for the API contract analyzer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.analysis.analyzers.api_contract import (
    APIContractAnalyzer,
    MethodSignature,
    _diff_signatures,
    _safe_import,
    _signature_dict,
    _snapshot_class,
)
from src.analysis.models import Severity


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class _SampleV1:
    def synthesize(self, text, voice, rate="+0%", volume="+0%"):
        return None

    async def async_method(self, payload):
        return payload


class _SampleV2Broken:
    """Drop the ``voice`` parameter; change sync -> async."""

    def synthesize(self, text, rate="+0%", volume="+0%"):
        return None

    async def async_method(self, payload, new_arg):
        return payload


def test_signature_dict_captures_public_methods():
    sigs = _signature_dict(_SampleV1)
    assert "synthesize" in sigs
    assert "async_method" in sigs
    assert sigs["synthesize"].params == ["text", "voice", "rate", "volume"]
    assert sigs["synthesize"].defaults.get("rate") == "'+0%'"
    assert sigs["async_method"].is_async is True


def test_diff_signatures_removed_method_is_critical():
    base = {
        "synthesize": MethodSignature("synthesize", ["text", "voice"]),
    }
    cur: dict = {}
    issues = _diff_signatures("cls", cur, base)
    kinds = [i.details["kind"] for i in issues]
    assert "method_removed" in kinds
    issue = next(i for i in issues if i.details["kind"] == "method_removed")
    assert issue.severity == Severity.CRITICAL


def test_diff_signatures_param_removed_is_breaking():
    base = {
        "synthesize": MethodSignature("synthesize", ["text", "voice"]),
    }
    cur = {
        "synthesize": MethodSignature("synthesize", ["text"]),
    }
    issues = _diff_signatures("cls", cur, base)
    sig_issues = [i for i in issues if i.details.get("kind") == "signature_changed"]
    assert sig_issues
    assert sig_issues[0].severity == Severity.HIGH


def test_diff_signatures_async_signature_change_is_breaking():
    base = {
        "method": MethodSignature("method", ["x"], is_async=False),
    }
    cur = {
        "method": MethodSignature("method", ["x"], is_async=True),
    }
    issues = _diff_signatures("cls", cur, base)
    sig_issues = [i for i in issues if i.details.get("kind") == "signature_changed"]
    assert sig_issues


def test_diff_signatures_new_method_is_info():
    base: dict = {}
    cur = {
        "new_method": MethodSignature("new_method", ["x"]),
    }
    issues = _diff_signatures("cls", cur, base)
    kinds = [i.details["kind"] for i in issues]
    assert "method_added" in kinds
    issue = next(i for i in issues if i.details["kind"] == "method_added")
    assert issue.severity == Severity.INFO


def test_safe_import_returns_none_for_missing():
    assert _safe_import("definitely.not.a.Module") is None


# ---------------------------------------------------------------------------
# Analyzer tests
# ---------------------------------------------------------------------------


def _build_baseline(path: Path, classes: dict) -> Path:
    payload = {"timestamp": "2024-01-01T00:00:00+00:00", "classes": classes}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_analyzer_creates_baseline_when_missing(tmp_path: Path):
    baseline = tmp_path / "api_baseline.json"
    analyzer = APIContractAnalyzer(
        baseline_path=baseline,
        targets=[],  # empty to keep the test focused
        auto_create_baseline=True,
    )
    issues = _run(analyzer.analyze())
    kinds = [i.details.get("kind") for i in issues]
    assert "baseline_created" in kinds
    assert baseline.is_file()


def test_analyzer_returns_no_issues_when_unchanged(tmp_path: Path):
    # Use a real, stable class to build the baseline.
    snapshot = {"src.engines.tts_engine.TTSEngine": _snapshot_class("src.engines.tts_engine.TTSEngine", "engine")}
    baseline = tmp_path / "api_baseline.json"
    _build_baseline(baseline, snapshot)

    analyzer = APIContractAnalyzer(
        baseline_path=baseline,
        targets=[("src.engines.tts_engine.TTSEngine", "engine")],
    )
    issues = _run(analyzer.analyze())
    breaking = [i for i in issues if i.severity in {Severity.CRITICAL, Severity.HIGH}]
    assert breaking == []


def test_analyzer_detects_breaking_change_in_baseline(tmp_path: Path):
    # Fake a baseline that includes a "removed" method.
    baseline_data = {
        "src.engines.tts_engine.TTSEngine": {
            "id": "src.engines.tts_engine.TTSEngine",
            "kind": "engine",
            "module": "src.engines.tts_engine",
            "qualname": "TTSEngine",
            "found": True,
            "methods": {
                "totally_made_up_method": {
                    "params": ["a", "b"],
                    "defaults": {},
                    "return_annotation": "Any",
                    "is_async": False,
                }
            },
        }
    }
    baseline = tmp_path / "api_baseline.json"
    _build_baseline(baseline, baseline_data)

    analyzer = APIContractAnalyzer(
        baseline_path=baseline,
        targets=[("src.engines.tts_engine.TTSEngine", "engine")],
    )
    issues = _run(analyzer.analyze())
    removed = [i for i in issues if i.details.get("kind") == "method_removed"]
    assert removed
    assert removed[0].severity == Severity.CRITICAL


def test_analyzer_detects_new_class(tmp_path: Path):
    baseline = tmp_path / "api_baseline.json"
    _build_baseline(baseline, {})

    analyzer = APIContractAnalyzer(
        baseline_path=baseline,
        targets=[("src.engines.tts_engine.TTSEngine", "engine")],
    )
    issues = _run(analyzer.analyze())
    new = [i for i in issues if i.details.get("kind") == "new_class"]
    assert new


def test_create_baseline_helper(tmp_path: Path):
    baseline = tmp_path / "api_baseline.json"
    analyzer = APIContractAnalyzer(
        baseline_path=baseline,
        targets=[("src.engines.tts_engine.TTSEngine", "engine")],
    )
    path = analyzer.create_baseline()
    assert path == baseline
    assert baseline.is_file()
    data = json.loads(baseline.read_text(encoding="utf-8"))
    assert "classes" in data


def test_analyzer_handles_corrupt_baseline(tmp_path: Path):
    baseline = tmp_path / "api_baseline.json"
    baseline.write_text("{ this is not json", encoding="utf-8")
    analyzer = APIContractAnalyzer(
        baseline_path=baseline,
        targets=[],
        auto_create_baseline=True,
    )
    issues = _run(analyzer.analyze())
    kinds = [i.details.get("kind") for i in issues]
    assert "baseline_created" in kinds
