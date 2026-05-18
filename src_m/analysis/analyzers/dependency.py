"""Dependency version compatibility analyzer.

Checks requirements.txt for dependency version compatibility, Python version
requirements, and reports outdated or incompatible dependencies.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..engine import BaseAnalyzer
from ..models import AnalysisCategory, AnalysisIssue, Severity


class DependencyAnalyzer(BaseAnalyzer):
    """Analyzer for dependency version compatibility."""

    def __init__(self) -> None:
        super().__init__(name="DependencyAnalyzer")

    def get_categories(self) -> List[AnalysisCategory]:
        return [AnalysisCategory.DEPENDENCY]

    async def analyze(self, context: Optional[Dict[str, Any]] = None) -> List[AnalysisIssue]:
        issues: List[AnalysisIssue] = []

        requirements_path = Path(__file__).parent.parent.parent.parent / "requirements.txt"

        try:
            text = requirements_path.read_text(encoding="utf-8")
        except Exception:
            return issues

        current_python = sys.version_info
        if current_python.major < 3 or (current_python.major == 3 and current_python.minor < 10):
            issues.append(
                AnalysisIssue(
                    severity=Severity.HIGH,
                    category=AnalysisCategory.DEPENDENCY,
                    description=f"Python 版本 {sys.version} 低于要求的 3.10",
                    suggestion="升级 Python 至 3.10 或更高版本",
                    location=str(requirements_path),
                    details={
                        "current_version": f"{current_python.major}.{current_python.minor}.{current_python.micro}",
                        "required_version": ">=3.10",
                    },
                )
            )

        version_spec_pattern = re.compile(r"([a-zA-Z_][\w.-]*)\s*([><=!~]+)\s*([\d.]+[\w.]*)")
        python_version_pattern = re.compile(r"python_version\s*(==|!=|>=|<=|>|<)\s*[\"']?(\d+\.\d+)[\"']?", re.IGNORECASE)

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue

            if ";" in line:
                dep_part, env_part = line.split(";", 1)
                env_match = python_version_pattern.search(env_part)
                if env_match:
                    op = env_match.group(1)
                    ver_str = env_match.group(2)
                    req_ver = tuple(int(x) for x in ver_str.split("."))
                    if not self._check_version_compat(current_python[:2], op, req_ver):
                        continue

            dep_part = line.split(";")[0].strip()
            dep_match = version_spec_pattern.match(dep_part)
            if dep_match:
                dep_name = dep_match.group(1)
                dep_op = dep_match.group(2)
                dep_ver_str = dep_match.group(3)
                self._check_dependency_version(issues, dep_name, dep_op, dep_ver_str, requirements_path)

        return issues

    def _check_python_version(
        self,
        current: tuple[int, int],
        op: str,
        required: tuple[int, int],
    ) -> bool:
        return self._check_version_compat(current, op, required)

    def _check_version_compat(
        self,
        current: tuple[int, ...],
        op: str,
        required: tuple[int, ...],
    ) -> bool:
        if op == "==":
            return current == required
        elif op == "!=":
            return current != required
        elif op == ">=":
            return current >= required
        elif op == "<=":
            return current <= required
        elif op == ">":
            return current > required
        elif op == "<":
            return current < required
        elif op == "~=":
            if len(required) < 2:
                return current >= required
            return current >= required and current[0] == required[0]
        return True

    def _check_dependency_version(
        self,
        issues: List[AnalysisIssue],
        dep_name: str,
        dep_op: str,
        dep_ver_str: str,
        requirements_path: Path,
    ) -> None:
        if dep_op in (">=", ">") and dep_op == ">=":
            issues.append(
                AnalysisIssue(
                    severity=Severity.MEDIUM,
                    category=AnalysisCategory.DEPENDENCY,
                    description=f"依赖 '{dep_name}' 指定宽松版本 ({dep_op}{dep_ver_str})，建议锁定精确版本",
                    suggestion=f"将 '{dep_name}' 的版本固定为具体版本号以避免意外升级",
                    location=str(requirements_path),
                    details={
                        "dependency": dep_name,
                        "specified_version": f"{dep_op}{dep_ver_str}",
                    },
                )
            )

        if dep_op == "<=" or dep_op == "<":
            issues.append(
                AnalysisIssue(
                    severity=Severity.HIGH,
                    category=AnalysisCategory.DEPENDENCY,
                    description=f"依赖 '{dep_name}' 版本上限 ({dep_op}{dep_ver_str}) 可能导致兼容性问题",
                    suggestion=f"更新 '{dep_name}' 以支持最新版本或移除版本上限",
                    location=str(requirements_path),
                    details={
                        "dependency": dep_name,
                        "specified_version": f"{dep_op}{dep_ver_str}",
                    },
                )
            )
