"""Code quality analyzer.

Detects code quality issues by scanning Python source files for
TODO, FIXME, XXX, HACK, and WORKAROUND comments.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..engine import BaseAnalyzer
from ..models import AnalysisCategory, AnalysisIssue, Severity

_COMMENT_PATTERN = re.compile(r'#.*\b(TODO|FIXME|XXX|HACK|WORKAROUND)\b\s*:?\s*(.*)')


class CodeQualityAnalyzer(BaseAnalyzer):
    """Analyzer for code quality issues such as incomplete code markers."""

    def __init__(self) -> None:
        super().__init__(name="CodeQualityAnalyzer")

    def get_categories(self) -> List[AnalysisCategory]:
        return [AnalysisCategory.CODE_QUALITY]

    async def analyze(self, context: Optional[Dict[str, Any]] = None) -> List[AnalysisIssue]:
        issues: List[AnalysisIssue] = []

        try:
            src_m_dir = Path(__file__).resolve().parent.parent.parent
            if not src_m_dir.is_dir():
                return issues

            files_findings: Dict[str, List[tuple[int, str]]] = defaultdict(list)

            for py_file in sorted(src_m_dir.rglob("*.py")):
                if py_file.resolve() == Path(__file__).resolve():
                    continue
                try:
                    text = py_file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                for line_no, line in enumerate(text.splitlines(), 1):
                    match = _COMMENT_PATTERN.search(line)
                    if match:
                        tag = match.group(1)
                        comment_text = (match.group(2) or "").strip()
                        full_text = f"{tag}: {comment_text}" if comment_text else tag
                        files_findings[str(py_file)].append((line_no, full_text))

            total_findings = sum(len(findings) for findings in files_findings.values())
            severity = Severity.HIGH if total_findings > 20 else Severity.LOW

            for file_path, findings in sorted(files_findings.items()):
                for line_no, comment_text in findings:
                    issues.append(
                        AnalysisIssue(
                            severity=severity,
                            category=AnalysisCategory.CODE_QUALITY,
                            description=f"发现代码标记: {comment_text}",
                            suggestion="在发布前解决标记的问题并移除注释",
                            location=file_path,
                            details={
                                "line": line_no,
                                "comment": comment_text,
                                "file": file_path,
                            },
                        )
                    )

        except Exception:
            pass

        return issues
