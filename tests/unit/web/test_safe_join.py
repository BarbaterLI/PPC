"""下载接口的文件名拼接（安全回归）。"""

from __future__ import annotations

import pytest

from src.web.validation import safe_join


def test_safe_join_accepts_plain_name(tmp_path) -> None:
    target = tmp_path / "a.mp3"
    target.write_text("x", encoding="utf-8")
    assert safe_join(tmp_path, "a.mp3") == target.resolve()


def test_safe_join_accepts_nested_relative(tmp_path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.mp3").write_text("x", encoding="utf-8")
    assert safe_join(tmp_path, "sub/b.mp3") == (sub / "b.mp3").resolve()


@pytest.mark.parametrize(
    "name",
    ["../victim/keep.txt", "C:/Windows/win.ini", "\\\\srv\\x", "/etc/passwd", "", ".."],
)
def test_safe_join_rejects_escape(tmp_path, name: str) -> None:
    assert safe_join(tmp_path, name) is None
