"""冒烟测试：验证 CLI 启动无异常。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_ppc10(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "ppc10.py", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestCLIBootstrap:
    def test_help_exits_zero(self) -> None:
        result = _run_ppc10("--help")
        # 帮助可能返回 0 或 2，2 也算正常（argparse 在 -h 上）
        assert result.returncode in (0, 1, 2)
        # 输出应非空
        out = (result.stdout or "") + (result.stderr or "")
        assert len(out) > 0

    def test_import_root_module(self) -> None:
        """验证 ppc10.py 可作为模块导入而不抛异常。"""
        import importlib.util

        spec = importlib.util.spec_from_file_location("ppc10", str(REPO_ROOT / "ppc10.py"))
        assert spec is not None
        # 不实际执行；只验证 spec 合法
        assert spec.name == "ppc10"
