"""`--pitch` 必须与 `--rate` 走同一套校验：非法值在 CLI 边界就报错。"""

from __future__ import annotations

import pytest

from src.cli.commands.convert import _normalize_pitch
from src.core.exceptions import CLIError


def test_pitch_normalized() -> None:
    assert _normalize_pitch("20Hz") == "+20Hz"
    assert _normalize_pitch("-30hz") == "-30Hz"


@pytest.mark.parametrize("bad", ["abc", "20", "+20%", ""])
def test_pitch_invalid_rejected(bad: str) -> None:
    with pytest.raises(CLIError):
        _normalize_pitch(bad)


def test_pitch_out_of_range_rejected() -> None:
    # 单音色路径不会像 cast 路径那样静默夹取，所以必须在入口拦住
    with pytest.raises(CLIError):
        _normalize_pitch("+200Hz")
