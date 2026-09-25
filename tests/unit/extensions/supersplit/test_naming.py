"""Unit tests for the SuperSplit naming module."""

from __future__ import annotations

from pathlib import Path

from src.extensions.supersplit.models import ConflictStrategy, SplitSegment
from src.extensions.supersplit.naming import (
    render_filename,
    resolve_conflict,
    sanitize_filename,
)


def _seg(title="标题", content="内容"):
    return SplitSegment(index=0, title=title, content=content, start_pos=0, end_pos=10)


# ---------------------------------------------------------------------------
# sanitize_filename
# ---------------------------------------------------------------------------


def test_sanitize_filename_removes_illegal_chars():
    name = 'a/b\\c:d*e?f"g<h>i|j'
    result = sanitize_filename(name)
    for ch in '/\\:*?"<>|':
        assert ch not in result


def test_sanitize_filename_replaces_whitespace():
    assert sanitize_filename("a b c") == "a_b_c"


def test_sanitize_filename_length_cap():
    result = sanitize_filename("x" * 150)
    assert len(result) == 100


# ---------------------------------------------------------------------------
# render_filename
# ---------------------------------------------------------------------------


def test_render_filename_index():
    seg = _seg()
    result = render_filename("{index}", seg, Path("source.txt"), 0)
    assert result == "1"


def test_render_filename_index_padded():
    seg = _seg()
    result = render_filename("{index:03d}", seg, Path("source.txt"), 0)
    assert result == "001"


def test_render_filename_title():
    seg = _seg(title="我的标题")
    result = render_filename("{title}", seg, Path("source.txt"), 0)
    assert result == "我的标题"


def test_render_filename_source():
    seg = _seg()
    result = render_filename("{source}", seg, Path("myfile.txt"), 0)
    assert result == "myfile"


def test_render_filename_date_time():
    seg = _seg()
    result = render_filename("{date}_{time}", seg, Path("source.txt"), 0)
    # date is YYYYMMDD (8 digits), time is HHMMSS (6 digits)
    parts = result.split("_")
    assert len(parts[0]) == 8
    assert len(parts[1]) == 6


def test_render_filename_unknown_var():
    seg = _seg()
    result = render_filename("{unknown}", seg, Path("source.txt"), 0)
    # handled gracefully, returns a non-empty filename
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# resolve_conflict
# ---------------------------------------------------------------------------


def test_resolve_conflict_overwrite(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("existing", encoding="utf-8")
    result = resolve_conflict(f, ConflictStrategy.OVERWRITE)
    assert result == f


def test_resolve_conflict_skip(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("existing", encoding="utf-8")
    result = resolve_conflict(f, ConflictStrategy.SKIP)
    assert result is None
    # non-existing returns path
    result2 = resolve_conflict(tmp_path / "new.txt", ConflictStrategy.SKIP)
    assert result2 == tmp_path / "new.txt"


def test_resolve_conflict_rename(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("existing", encoding="utf-8")
    result = resolve_conflict(f, ConflictStrategy.RENAME)
    assert result is not None
    assert result.name == "file_1.txt"


def test_resolve_conflict_max_counter(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("existing", encoding="utf-8")
    # create all counter variants 1..9999 so the counter exhausts
    for i in range(1, 10000):
        (tmp_path / f"file_{i}.txt").write_text("x", encoding="utf-8")
    result = resolve_conflict(f, ConflictStrategy.RENAME)
    assert result is None
