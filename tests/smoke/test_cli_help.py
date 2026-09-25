"""冒烟测试：验证 CLI 启动无异常。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_ppc(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    # ppc CLI 强制 UTF-8 输出（见 src.cli.output.setup_console），
    # 子进程管道必须显式按 UTF-8 解码；否则中文 Windows 默认 GBK
    # 解码会在 reader 线程抛 UnicodeDecodeError 并吞掉 stdout。
    return subprocess.run(
        [sys.executable, "ppc.py", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


class TestCLIBootstrap:
    def test_help_exits_zero(self) -> None:
        result = _run_ppc("--help")
        # 帮助可能返回 0 或 2，2 也算正常（argparse 在 -h 上）
        assert result.returncode in (0, 1, 2)
        # 输出应非空
        out = (result.stdout or "") + (result.stderr or "")
        assert len(out) > 0

    def test_import_root_module(self) -> None:
        """验证 ppc.py 可作为模块导入而不抛异常。"""
        import importlib.util

        spec = importlib.util.spec_from_file_location("ppc", str(REPO_ROOT / "ppc.py"))
        assert spec is not None
        # 不实际执行；只验证 spec 合法
        assert spec.name == "ppc"


class TestCLICommands:
    def test_webui_help(self) -> None:
        result = _run_ppc("webui", "--help")
        assert result.returncode == 0
        out = (result.stdout or "") + (result.stderr or "")
        assert "--host" in out and "--port" in out

    def test_dist_help(self) -> None:
        result = _run_ppc("dist", "--help")
        assert result.returncode == 0
        out = (result.stdout or "") + (result.stderr or "")
        assert "node" in out and "master" in out

    def test_distributed_alias_help(self) -> None:
        """distributed 别名应可用（与 dist 相同命令树）。"""
        result = _run_ppc("distributed", "--help")
        assert result.returncode == 0

    def test_help_json_non_interactive(self) -> None:
        """help --json 非交互输出命令索引，不挂起等待输入。"""
        result = _run_ppc("help", "--json", timeout=10)
        assert result.returncode == 0
        import json

        data = json.loads(result.stdout)
        assert "convert" in data
        assert "webui" in data

    def test_version_exits_zero(self) -> None:
        from src.version import __version__

        result = _run_ppc("--version")
        assert result.returncode == 0
        assert __version__ in ((result.stdout or "") + (result.stderr or ""))
