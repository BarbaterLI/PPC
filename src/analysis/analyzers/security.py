"""Security analyzer.

Detects common security issues in the source tree:

* Sensitive information leakage (API key / token / password / private key)
  via both well-known regexes and Shannon-entropy based heuristics.
* Unsafe deserialization (``pickle.load`` and ``yaml.load`` without a
  ``Loader`` argument).
* Command injection sinks (``subprocess`` invocations using
  ``shell=True`` or string commands).

The analyzer is pure-Python, has no third-party dependencies, and operates
on ``*.py`` files.  The set of roots scanned is configurable through the
analysis ``context`` dictionary.  When the context is absent the analyzer
defaults to scanning the bundled ``src`` tree.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from ..engine import BaseAnalyzer
from ..models import AnalysisCategory, AnalysisIssue, Severity

# ---------------------------------------------------------------------------
# Heuristics / patterns
# ---------------------------------------------------------------------------

# Regex based high-precision patterns.  The named "key" is used to label the
# match in the produced issue.
_SENSITIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Generic API key / secret / token assignment.  This matches both plain
    # ``api_key = "..."`` and dict-literal style ``'api_key': '...'.``
    (
        "api_key_assignment",
        re.compile(
            r"""(?ix)
        (?P<name>api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|
           client[_-]?secret|app[_-]?secret|private[_-]?key)
        \b
        \s*['"]?\s*[:=]\s*
        ['"](?P<value>[A-Za-z0-9_+\-/\.=]{12,})['"]
        """
        ),
    ),
    # Bearer tokens in source
    ("bearer_token", re.compile(r"""(?i)bearer\s+[A-Za-z0-9\-_\.=]{16,}""")),
    # AWS access key
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    # AWS secret access key (heuristic - 40 base64-ish chars after assignment)
    (
        "aws_secret_key",
        re.compile(r"""(?i)aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*['"](?P<value>[A-Za-z0-9/+=]{40})['"]"""),
    ),
    # GitHub personal access token (classic & fine-grained)
    ("github_pat", re.compile(r"\bghp_[A-Za-z0-9]{30,}\b")),
    # Slack token
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    # Generic password assignment
    (
        "password_assignment",
        re.compile(
            r"""(?ix)
        (?P<name>password|passwd|pwd)
        \b
        \s*['"]?\s*[:=]\s*
        ['"](?P<value>[^'"\s]{6,})['"]
        """
        ),
    ),
    # PEM private key header
    ("private_key_pem", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    # High-entropy "secret" string
    ("hex_blob", re.compile(r"\b[0-9a-f]{32,}\b", re.IGNORECASE)),
    # JWT
    ("jwt_token", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
]

# Unsafe deserialization sinks.
_UNSAFE_DESER_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("pickle_load", re.compile(r"\bpickle\.load\s*\(")),
    ("pickle_loads", re.compile(r"\bpickle\.loads\s*\(")),
    ("cPickle_load", re.compile(r"\bcPickle\.load\s*\(")),
    ("yaml_load_unsafe", re.compile(r"\byaml\.load\s*\((?![^)]*Loader\s*=)")),
    ("yaml_load_full", re.compile(r"\byaml\.load_all\s*\((?![^)]*Loader\s*=)")),
    ("marshal_loads", re.compile(r"\bmarshal\.loads\s*\(")),
]

# Command injection sinks.  We are conservative: the analyzer only flags
# subprocess calls that pass ``shell=True`` or use ``os.system``/
# ``os.popen`` which always run in a shell.
_CMD_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "subprocess_shell",
        re.compile(
            r"""(?x)
        \b(subprocess\.(?:call|run|Popen|check_output|check_call)|
            subprocess\.getoutput|
            os\.system|os\.popen)
        \s*\([^)]*shell\s*=\s*True
        """
        ),
    ),
    ("os_system", re.compile(r"\bos\.(?:system|popen)\s*\(")),
    (
        "subprocess_string_cmd",
        re.compile(
            r"""(?x)
        \bsubprocess\.(?:call|run|Popen|check_output|check_call)\s*\(
        \s*['"](?P<cmd>[^'"]+)['"]
        """
        ),
    ),
]

# Minimum string length to even consider computing entropy.  Shorter strings
# are almost never "secret" content.
_ENTROPY_MIN_LEN = 20
# Entropy threshold for flagging a high-entropy string literal.
_ENTROPY_THRESHOLD = 4.5

# String literal regex (single/double quoted, multi-line triple-quoted).
_STRING_LITERAL = re.compile(
    r"""
    (\"\"\"[^\"]{8,}?\"\"\")    # triple double
    | (\'[^\']{8,}?\')            # triple single (not strictly but close)
    | (\"(?:[^\"\\]|\\.){8,}\")   # single double
    | (\'(?:[^\'\\]|\\.){8,}\')   # single single
    """,
    re.VERBOSE,
)

# Project markers used to identify "first-party" sources.
_PYTHON_SUFFIX = ".py"

# Default scan root: src tree.
_DEFAULT_SCAN_ROOT = "src"

# Cap on the number of files scanned to avoid runaway cost.
_MAX_SCAN_FILES = 5000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shannon_entropy(value: str) -> float:
    """Return the Shannon entropy (in bits) of *value*."""
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _scan_string_literal(value: str) -> tuple[str, float] | None:
    """Return ``(reason, entropy)`` if a string looks like a secret."""
    if len(value) < _ENTROPY_MIN_LEN:
        return None
    entropy = _shannon_entropy(value)
    if entropy >= _ENTROPY_THRESHOLD:
        return ("high_entropy", entropy)
    return None


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class SecurityAnalyzer(BaseAnalyzer):
    """Analyzer for security issues: secrets, deserialization, command injection."""

    def __init__(
        self,
        scan_root: str | None = None,
        patterns: dict[str, list[tuple[str, re.Pattern]]] | None = None,
        max_files: int = _MAX_SCAN_FILES,
    ) -> None:
        super().__init__(name="SecurityAnalyzer")
        self._scan_root = scan_root or _DEFAULT_SCAN_ROOT
        self._max_files = max_files
        # Allow tests to inject custom pattern dictionaries.
        self._patterns = patterns or {
            "sensitive": list(_SENSITIVE_PATTERNS),
            "deserialization": list(_UNSAFE_DESER_PATTERNS),
            "command_injection": list(_CMD_INJECTION_PATTERNS),
        }
        # Used by tests to force a single-threaded scan when desired.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # BaseAnalyzer
    # ------------------------------------------------------------------

    def get_categories(self) -> list[AnalysisCategory]:
        return [AnalysisCategory.SECURITY]

    async def analyze(self, context: dict[str, Any] | None = None) -> list[AnalysisIssue]:
        issues: list[AnalysisIssue] = []

        scan_root_str = self._scan_root
        if context:
            scan_root_str = context.get("scan_root", scan_root_str)
            extra_files = context.get("files")
        else:
            extra_files = None

        # 1) Scan project files
        root = Path(scan_root_str)
        if not root.is_absolute():
            root = (Path(__file__).resolve().parent.parent.parent / scan_root_str).resolve()

        files: list[Path] = []
        if root.is_dir():
            for py_file in sorted(root.rglob(f"*{_PYTHON_SUFFIX}")):
                if not py_file.is_file():
                    continue
                files.append(py_file)
                if len(files) >= self._max_files:
                    break

        if extra_files:
            files.extend(Path(f) for f in extra_files)

        for path in files:
            self._scan_file(path, issues)

        # 2) Allow context to inject a synthetic source for direct unit testing.
        if context:
            inline = context.get("inline_sources")
            if isinstance(inline, dict):
                for virtual_path, text in inline.items():
                    self._scan_text(virtual_path, text or "", issues)

        return issues

    # ------------------------------------------------------------------
    # File scanning
    # ------------------------------------------------------------------

    def _scan_file(self, path: Path, issues: list[AnalysisIssue]) -> None:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            return
        self._scan_text(str(path), text, issues)

    def _scan_text(self, location: str, text: str, issues: list[AnalysisIssue]) -> None:
        for line_no, line in enumerate(text.splitlines(), 1):
            self._scan_line(location, line_no, line, issues)

    def _scan_line(
        self,
        location: str,
        line_no: int,
        line: str,
        issues: list[AnalysisIssue],
    ) -> None:
        # Skip obvious comment-only lines for sensitive regex matches.
        stripped = line.lstrip()
        in_comment = stripped.startswith("#")

        # 1) Sensitive patterns (regex based)
        for label, pattern in self._patterns["sensitive"]:
            if in_comment and label in {"hex_blob"}:
                # High-entropy hashes inside comments are noise.
                continue
            for match in pattern.finditer(line):
                snippet = match.group(0)
                issues.append(
                    AnalysisIssue(
                        severity=Severity.HIGH,
                        category=AnalysisCategory.SECURITY,
                        description=(f"疑似敏感信息: {label} (长度={len(snippet)})"),
                        suggestion=("将密钥迁移到环境变量、密钥管理服务或 .env 文件（已加入 .gitignore）"),
                        location=location,
                        details={
                            "kind": "sensitive",
                            "pattern": label,
                            "line": line_no,
                            "snippet": snippet[:80],
                        },
                    )
                )

        # 2) Unsafe deserialization
        for label, pattern in self._patterns["deserialization"]:
            if pattern.search(line):
                severity = (
                    Severity.CRITICAL if label in {"pickle_load", "pickle_loads", "cPickle_load"} else Severity.HIGH
                )
                issues.append(
                    AnalysisIssue(
                        severity=severity,
                        category=AnalysisCategory.SECURITY,
                        description=f"检测到不安全的反序列化调用: {label}",
                        suggestion=(
                            "对 pickle 使用来源可信的字节流或改用 JSON；yaml.load 必须显式传入 Loader=SafeLoader"
                        ),
                        location=location,
                        details={"kind": "deserialization", "pattern": label, "line": line_no},
                    )
                )

        # 3) Command injection
        for label, pattern in self._patterns["command_injection"]:
            if pattern.search(line):
                severity = Severity.CRITICAL if label == "subprocess_shell" else Severity.HIGH
                issues.append(
                    AnalysisIssue(
                        severity=severity,
                        category=AnalysisCategory.SECURITY,
                        description=f"检测到潜在命令注入点: {label}",
                        suggestion=(
                            "避免 shell=True；改用 shell=False + 列表参数并校验输入，或使用 shlex.quote 包裹用户输入"
                        ),
                        location=location,
                        details={"kind": "command_injection", "pattern": label, "line": line_no},
                    )
                )

        # 4) Entropy based heuristic on string literals
        if not in_comment:
            for match in _STRING_LITERAL.finditer(line):
                match.group(0)[1:-1] if match.group(0)[0] != '"' or len(match.group(0)) < 4 else match.group(0)
                # Strip quotes for the entropy check.
                text_quote = match.group(0)
                inner = text_quote[3:-3] if text_quote.startswith(('"""', "'''")) else text_quote[1:-1]
                result = _scan_string_literal(inner)
                if result is not None:
                    reason, entropy = result
                    issues.append(
                        AnalysisIssue(
                            severity=Severity.MEDIUM,
                            category=AnalysisCategory.SECURITY,
                            description=(f"字符串字面量具有高 Shannon 熵 ({entropy:.2f} bits)，疑似密钥/Token"),
                            suggestion=("若该字符串为凭证，请迁移到环境变量/密钥管理服务，并在代码中仅引用其名称"),
                            location=location,
                            details={
                                "kind": "entropy",
                                "pattern": reason,
                                "line": line_no,
                                "entropy": round(entropy, 3),
                                "length": len(inner),
                            },
                        )
                    )


# Public re-exports for testing helpers
__all__ = [
    "SecurityAnalyzer",
    "_shannon_entropy",
    "_scan_string_literal",
]
