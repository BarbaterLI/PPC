"""API contract analyzer.

Detects breaking changes in the public API of PPC10 by introspecting
``Engine``/``Executor`` classes with the :mod:`inspect`
module and diffing the current signatures against a baseline snapshot.

A baseline snapshot is a JSON file mapping a class identifier to a
list of method signatures, for example::

    {
        "src.engines.tts_engine.TTSEngine": {
            "module": "src.engines.tts_engine",
            "qualname": "TTSEngine",
            "methods": {
                "synthesize": {
                    "params": ["text", "voice", "rate", "volume"],
                    "defaults": {"rate": "+0%", "volume": "+0%"},
                    "return_annotation": "None",
                },
                ...
            }
        }
    }

If the baseline file is missing the analyzer creates it on first run so
the user has a stable reference for future comparisons.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..engine import BaseAnalyzer
from ..models import AnalysisCategory, AnalysisIssue, Severity

# Default location of the API baseline snapshot.
DEFAULT_BASELINE_PATH = Path(__file__).resolve().parent.parent.parent.parent / ".trae" / "api_baseline.json"

# Classes that we always look at when scanning for Engine/Executor
# contracts.  The "id" used in the baseline is a dotted path.
DEFAULT_TARGETS: list[tuple[str, str]] = [
    # (dotted_path, kind)
    ("src.engines.tts_engine.TTSEngine", "engine"),
    ("src.engines.chapter_engine.ChapterEngine", "engine"),
    ("src.engines.epub_engine.EPUBEngine", "engine"),
    ("src.executors.base.BaseExecutor", "executor"),
    ("src.executors.tts.TTSExecutor", "executor"),
    ("src.executors.splitter.SplitterExecutor", "executor"),
    ("src.executors.batcher.BatcherExecutor", "executor"),
]


@dataclass
class MethodSignature:
    """A minimal representation of a callable's public signature."""

    name: str
    params: list[str] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)
    return_annotation: str = "Any"
    is_async: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "defaults": self.defaults,
            "return_annotation": self.return_annotation,
            "is_async": self.is_async,
        }

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> MethodSignature:
        return cls(
            name=name,
            params=list(data.get("params", [])),
            defaults=dict(data.get("defaults", {})),
            return_annotation=str(data.get("return_annotation", "Any")),
            is_async=bool(data.get("is_async", False)),
        )


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def _safe_import(dotted_path: str) -> type[Any] | None:
    """Import *dotted_path* and return the resolved object or ``None``."""
    try:
        module_name, _, attr = dotted_path.rpartition(".")
        if not module_name:
            return None
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, attr, None)
    except Exception:
        return None


def _signature_dict(obj: Any) -> dict[str, MethodSignature]:
    """Return a mapping of method-name to :class:`MethodSignature` for *obj*."""
    methods: dict[str, MethodSignature] = {}
    if obj is None:
        return methods

    target = obj if inspect.isclass(obj) else type(obj)
    try:
        members = inspect.getmembers(target, predicate=inspect.isfunction)
    except Exception:
        return methods

    for name, func in members:
        if name.startswith("_") and name not in {"__call__"}:
            continue
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            continue

        params: list[str] = []
        defaults: dict[str, Any] = {}
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            params.append(pname)
            if param.default is not inspect.Parameter.empty:
                try:
                    defaults[pname] = repr(param.default)
                except Exception:
                    defaults[pname] = "<unrepresentable>"

        ret_ann = "Any"
        if sig.return_annotation is not inspect.Signature.empty:
            try:
                ret_ann = str(sig.return_annotation)
            except Exception:
                ret_ann = "Any"

        methods[name] = MethodSignature(
            name=name,
            params=params,
            defaults=defaults,
            return_annotation=ret_ann,
            is_async=asyncio.iscoroutinefunction(func),
        )
    return methods


def _snapshot_class(dotted_path: str, kind: str) -> dict[str, Any]:
    """Build a snapshot dict for the class at *dotted_path*."""
    obj = _safe_import(dotted_path)
    module_name = dotted_path.rpartition(".")[0]
    qualname = dotted_path.rsplit(".", 1)[-1]
    return {
        "id": dotted_path,
        "kind": kind,
        "module": module_name,
        "qualname": qualname,
        "found": obj is not None,
        "methods": {n: m.to_dict() for n, m in _signature_dict(obj).items()},
    }


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------


def _diff_signatures(
    class_id: str,
    current: dict[str, MethodSignature],
    baseline: dict[str, MethodSignature],
) -> list[AnalysisIssue]:
    """Diff two signature dicts and return a list of breaking-change issues."""
    issues: list[AnalysisIssue] = []

    # Detect removed methods
    for name in sorted(set(baseline) - set(current)):
        base = baseline[name]
        issues.append(
            AnalysisIssue(
                severity=Severity.CRITICAL,
                category=AnalysisCategory.CODE_QUALITY,
                description=f"[API 破坏] {class_id}.{name}() 已被移除",
                suggestion=("考虑在父类/兼容层保留该方法或明确记录到 CHANGELOG；若确认移除，请在基线文件中同步更新"),
                location=f"{class_id}.{name}",
                details={
                    "kind": "method_removed",
                    "method": name,
                    "class": class_id,
                    "previous_signature": base.to_dict(),
                },
            )
        )

    # Detect added methods (info-level, non-breaking)
    for name in sorted(set(current) - set(baseline)):
        cur = current[name]
        issues.append(
            AnalysisIssue(
                severity=Severity.INFO,
                category=AnalysisCategory.CODE_QUALITY,
                description=f"[API 新增] {class_id}.{name}() 新增",
                suggestion="新增方法向后兼容，但需要更新文档",
                location=f"{class_id}.{name}",
                details={
                    "kind": "method_added",
                    "method": name,
                    "class": class_id,
                    "signature": cur.to_dict(),
                },
            )
        )

    # Detect signature changes in shared methods
    for name in sorted(set(current) & set(baseline)):
        cur = current[name]
        base = baseline[name]
        changes: list[str] = []

        if cur.params != base.params:
            if len(cur.params) < len(base.params):
                changes.append(f"参数数量减少: {base.params} -> {cur.params}")
            else:
                # Adding parameters is potentially breaking if positional.
                added = [p for p in cur.params if p not in base.params]
                if added:
                    changes.append(f"新增位置参数 {added}; 已有调用方传入的位置实参会被错位")
                # Removing parameters is breaking.
                removed = [p for p in base.params if p not in cur.params]
                if removed:
                    changes.append(f"移除参数 {removed}")

        # Return type change is a contract change.
        if cur.return_annotation != base.return_annotation:
            changes.append(f"返回类型变化: {base.return_annotation} -> {cur.return_annotation}")

        # Sync -> async or vice versa is breaking.
        if cur.is_async != base.is_async:
            changes.append(f"同步/异步签名变化: async={base.is_async} -> async={cur.is_async}")

        if changes:
            issues.append(
                AnalysisIssue(
                    severity=Severity.HIGH,
                    category=AnalysisCategory.CODE_QUALITY,
                    description=(f"[API 破坏] {class_id}.{name}() 签名变化: " + "; ".join(changes)),
                    suggestion=("保持向后兼容：新增参数应提供默认值；如确需修改返回类型，请通过新增方法或兼容层过渡"),
                    location=f"{class_id}.{name}",
                    details={
                        "kind": "signature_changed",
                        "method": name,
                        "class": class_id,
                        "current_signature": cur.to_dict(),
                        "baseline_signature": base.to_dict(),
                        "changes": changes,
                    },
                )
            )

    return issues


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class APIContractAnalyzer(BaseAnalyzer):
    """Analyzer for Engine/Executor public API breaking changes."""

    def __init__(
        self,
        baseline_path: Path | None = None,
        targets: Iterable[tuple[str, str]] | None = None,
        auto_create_baseline: bool = True,
    ) -> None:
        super().__init__(name="APIContractAnalyzer")
        self._baseline_path = Path(baseline_path) if baseline_path else DEFAULT_BASELINE_PATH
        self._targets: list[tuple[str, str]] = list(targets) if targets else list(DEFAULT_TARGETS)
        self._auto_create = auto_create_baseline

    # ------------------------------------------------------------------
    # BaseAnalyzer
    # ------------------------------------------------------------------

    def get_categories(self) -> list[AnalysisCategory]:
        return [AnalysisCategory.CODE_QUALITY]

    async def analyze(self, context: dict[str, Any] | None = None) -> list[AnalysisIssue]:
        # Optionally override path / targets via context.
        if context:
            bp = context.get("baseline_path")
            if bp:
                self._baseline_path = Path(bp)
            if "targets" in context and isinstance(context["targets"], list):
                self._targets = list(context["targets"])
            self._auto_create = bool(context.get("auto_create_baseline", self._auto_create))

        # 1) Snapshot current public API
        current_snapshot: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "classes": {dotted: _snapshot_class(dotted, kind) for dotted, kind in self._targets},
        }

        # 2) Load baseline (or create one)
        baseline = self._load_baseline()
        is_first_run = baseline is None

        if is_first_run:
            if self._auto_create:
                self._save_baseline(current_snapshot)
                return [
                    AnalysisIssue(
                        severity=Severity.INFO,
                        category=AnalysisCategory.CODE_QUALITY,
                        description=(f"已生成 API 基线快照 (首次运行)，路径: {self._baseline_path}"),
                        suggestion=("下次执行将自动与该基线对比以检测破坏性变更"),
                        location=str(self._baseline_path),
                        details={"kind": "baseline_created", "classes": len(current_snapshot["classes"])},
                    )
                ]
            return [
                AnalysisIssue(
                    severity=Severity.MEDIUM,
                    category=AnalysisCategory.CODE_QUALITY,
                    description="API 基线快照不存在",
                    suggestion=("使用 'ppc10 analyze --api-contract --api-baseline-init' 生成基线快照"),
                    location=str(self._baseline_path),
                    details={"kind": "baseline_missing"},
                )
            ]

        # 3) Diff baseline vs current.
        assert baseline is not None
        issues: list[AnalysisIssue] = []
        baseline_classes: dict[str, dict[str, Any]] = baseline.get("classes", {})

        for class_id, snap in current_snapshot["classes"].items():
            base_class = baseline_classes.get(class_id)
            if base_class is None:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.INFO,
                        category=AnalysisCategory.CODE_QUALITY,
                        description=f"新增目标类: {class_id}",
                        suggestion="将该类加入基线或从 targets 中移除",
                        location=class_id,
                        details={"kind": "new_class", "class": class_id},
                    )
                )
                continue
            if not snap.get("found", False):
                issues.append(
                    AnalysisIssue(
                        severity=Severity.CRITICAL,
                        category=AnalysisCategory.CODE_QUALITY,
                        description=f"目标类已无法导入: {class_id}",
                        suggestion="检查模块路径或重命名后更新基线",
                        location=class_id,
                        details={"kind": "missing_class", "class": class_id},
                    )
                )
                continue
            cur_methods: dict[str, MethodSignature] = {
                n: MethodSignature.from_dict(n, m) for n, m in snap.get("methods", {}).items()
            }
            base_methods: dict[str, MethodSignature] = {
                n: MethodSignature.from_dict(n, m) for n, m in base_class.get("methods", {}).items()
            }
            issues.extend(_diff_signatures(class_id, cur_methods, base_methods))

        # Detect classes that disappeared from targets.
        for class_id in sorted(set(baseline_classes) - set(current_snapshot["classes"])):
            issues.append(
                AnalysisIssue(
                    severity=Severity.MEDIUM,
                    category=AnalysisCategory.CODE_QUALITY,
                    description=f"基线中存在的类已不在目标列表: {class_id}",
                    suggestion="更新 targets 或恢复该类",
                    location=class_id,
                    details={"kind": "target_dropped", "class": class_id},
                )
            )

        return issues

    # ------------------------------------------------------------------
    # Baseline persistence
    # ------------------------------------------------------------------

    def _load_baseline(self) -> dict[str, Any] | None:
        if not self._baseline_path.is_file():
            return None
        try:
            text = self._baseline_path.read_text(encoding="utf-8")
            data = json.loads(text)
            if not isinstance(data, dict) or "classes" not in data:
                return None
            return data
        except (json.JSONDecodeError, OSError):
            return None

    def _save_baseline(self, snapshot: dict[str, Any]) -> None:
        self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self._baseline_path.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def create_baseline(self) -> Path:
        """Create / overwrite the API baseline with the current snapshot."""
        snapshot = {
            "timestamp": datetime.now(UTC).isoformat(),
            "classes": {dotted: _snapshot_class(dotted, kind) for dotted, kind in self._targets},
        }
        self._save_baseline(snapshot)
        return self._baseline_path


__all__ = [
    "APIContractAnalyzer",
    "MethodSignature",
    "DEFAULT_BASELINE_PATH",
    "DEFAULT_TARGETS",
]
