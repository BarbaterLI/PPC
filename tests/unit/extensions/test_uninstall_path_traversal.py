"""卸载扩展时的路径穿越（安全回归）。

旧实现直接 `extensions_dir / name` 再 `rmtree`：`..` 会解析到项目根并递归删除，
`C:/Windows` 这类绝对路径在 Windows 上直接覆盖左侧路径。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.extensions.package_ops import ExtensionUninstaller


@pytest.mark.parametrize("name", ["..", "../victim", "C:/Windows", "a/b", "", "."])
def test_uninstall_rejects_traversal(tmp_path, name: str) -> None:
    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("x", encoding="utf-8")

    result = ExtensionUninstaller(ext_dir).uninstall(name, force=True)

    assert result["success"] is False
    assert (victim / "keep.txt").exists()


def test_uninstall_still_removes_legit_extension(tmp_path) -> None:
    ext_dir = tmp_path / "extensions"
    target = ext_dir / "demo_ext"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("", encoding="utf-8")

    result = ExtensionUninstaller(ext_dir).uninstall("demo_ext", force=True)

    assert result["success"] is True
    assert not target.exists()
