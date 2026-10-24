"""T1 回归测试：导入别名 / getattr 字符串属性访问不得绕过静态扫描。

修复前的绕过方式（`import subprocess as sp` / `import os as o` /
`from os import system`）只命中 low/medium 的导入规则，STRICT 模式下不会
阻断任意命令执行。修复后扫描器把本地绑定名还原为真实点分名再做规则匹配。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.extensions.base import SecurityLevel
from src.extensions.loader import ExtensionLoader
from src.extensions.scanner import (
    ExtensionSecurityScanner,
    SecurityScanResult,
    SecuritySeverity,
)


def _scan(tmp_path: Path, source: str) -> SecurityScanResult:
    target = tmp_path / "mod.py"
    target.write_text(source, encoding="utf-8")
    return ExtensionSecurityScanner().scan_file(target)


# ---------------------------------------------------------------------------
# 三种别名导入形式都必须被判为 HIGH / CRITICAL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected_rule"),
    [
        (
            "import subprocess as sp\n\n\ndef f(c):\n    sp.run(c, shell=True)\n",
            "SUBPROCESS_SHELL_TRUE",
        ),
        (
            "import os as o\n\n\ndef f(c):\n    o.system(c)\n",
            "OS_SYSTEM",
        ),
        (
            "from os import system\n\n\ndef f(c):\n    system(c)\n",
            "OS_SYSTEM",
        ),
        (
            "from subprocess import run as r\n\n\ndef f(c):\n    r(c, shell=True)\n",
            "SUBPROCESS_SHELL_TRUE",
        ),
    ],
)
def test_alias_import_bypass_is_flagged(
    tmp_path: Path, source: str, expected_rule: str
) -> None:
    """`import X as a` / `from X import Y` / `from X import Y as a` 均被识别。"""
    result = _scan(tmp_path, source)

    assert result.has_critical_or_high(), (
        f"alias bypass was not blocked, issues={[(i.severity.value, i.rule_id) for i in result.issues]}"
    )
    assert any(
        issue.rule_id == expected_rule
        and issue.severity in (SecuritySeverity.HIGH, SecuritySeverity.CRITICAL)
        for issue in result.issues
    ), f"expected {expected_rule}, got {[i.rule_id for i in result.issues]}"


def test_alias_applies_to_attribute_chain(tmp_path: Path) -> None:
    """`import os as o` 后 `o.path.join` 不误报，`o.system` 仍命中。"""
    clean = _scan(tmp_path, "import os as o\n\n\ndef f():\n    return o.path.join('a', 'b')\n")
    assert clean.issues == []

    dirty = _scan(tmp_path, "import os as o\n\n\ndef f(c):\n    return o.system(c)\n")
    assert any(i.rule_id == "OS_SYSTEM" for i in dirty.issues)


@pytest.mark.parametrize(
    "source",
    [
        "import os\n\n\ndef f(c):\n    getattr(os, 'system')(c)\n",
        "import os as o\n\n\ndef f(c):\n    getattr(o, 'system')(c)\n",
        "import subprocess as sp\n\n\ndef f(c):\n    getattr(sp, 'run')(c, shell=True)\n",
    ],
)
def test_getattr_string_attribute_is_flagged(tmp_path: Path, source: str) -> None:
    """`getattr(os, "system")` 这类字符串属性访问给出保守告警。"""
    result = _scan(tmp_path, source)
    assert result.has_critical_or_high(), [i.rule_id for i in result.issues]
    assert any(i.rule_id.startswith("GETATTR_") for i in result.issues)


# ---------------------------------------------------------------------------
# 正常写法不得误报
# ---------------------------------------------------------------------------


def test_plain_imports_do_not_false_positive(tmp_path: Path) -> None:
    """`import os` + `os.path.join` 不应产生任何 issue。"""
    result = _scan(
        tmp_path,
        "import os\nimport os.path as osp\n\n\ndef f():\n"
        "    return os.path.join('a', 'b'), osp.basename('x/y')\n",
    )
    assert result.issues == []
    assert result.risk_score == 0


def test_relative_import_is_not_resolved_as_stdlib(tmp_path: Path) -> None:
    """相对导入（from .os import system）不得被当成 os.system 误报。"""
    result = _scan(
        tmp_path,
        "from .os import system\n\n\ndef f(c):\n    return system(c)\n",
    )
    assert not any(i.rule_id == "OS_SYSTEM" for i in result.issues)


# ---------------------------------------------------------------------------
# 端到端：STRICT 模式下加载被拒绝
# ---------------------------------------------------------------------------

ALIAS_BYPASS_EXT_SOURCE = '''"""Extension that hides os.system behind an import alias."""
import os as o

from src.extensions.base import Extension, ExtensionMetadata


class AliasBypassExtension(Extension):
    def __init__(self):
        super().__init__(ExtensionMetadata(
            name="alias-bypass-ext",
            version="1.0.0",
            description="alias bypass test extension",
        ))

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False

    def run(self, cmd):
        return o.system(cmd)


extension = AliasBypassExtension()
'''


def test_strict_mode_blocks_alias_bypass_extension(tmp_path: Path) -> None:
    """STRICT 模式下，用 `import os as o` 绕过导入规则仍必须被阻断。"""
    ext_path = tmp_path / "alias_bypass.py"
    ext_path.write_text(ALIAS_BYPASS_EXT_SOURCE, encoding="utf-8")

    loader = ExtensionLoader(security_level=SecurityLevel.STRICT)
    try:
        asyncio.run(loader._load_extension_from_file(ext_path))
        assert "alias-bypass-ext" not in loader.get_loaded_extensions()
        failed = loader.get_failed_extensions()
        assert len(failed) == 1
        assert "Security scan blocked" in failed[0]["error"]
    finally:
        asyncio.run(loader.clear())
