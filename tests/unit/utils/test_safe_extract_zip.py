"""Unit tests for :func:`src.utils.files.safe_extract_zip` zip-slip guard.

The traversal check must be boundary-based (``Path.relative_to``), not a
prefix string match, so sibling-directory escapes are refused.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from src.utils.files import safe_extract_zip


def _zip_bytes(entries: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_sibling_directory_traversal_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "cache"
    target.mkdir()
    # Resolves to tmp_path/"cache_evil/x", whose string prefix matches
    # tmp_path/"cache" — a prefix match would (incorrectly) accept it.
    sibling_escape = "../cache_evil/traversal.txt"
    zf = zipfile.ZipFile(io.BytesIO(_zip_bytes({sibling_escape: "boom"})))
    with pytest.raises(ValueError, match="path traversal"):
        safe_extract_zip(zf, target)


def test_normal_entries_extract_within_target(tmp_path: Path) -> None:
    target = tmp_path / "cache"
    target.mkdir()
    zf = zipfile.ZipFile(io.BytesIO(_zip_bytes({"sub/a.txt": "ok"})))
    safe_extract_zip(zf, target)
    assert (target / "sub" / "a.txt").read_text() == "ok"


def test_absolute_entry_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "cache"
    target.mkdir()
    zf = zipfile.ZipFile(io.BytesIO(_zip_bytes({"/tmp/evil.txt": "boom"})))
    with pytest.raises((OSError, ValueError)):
        # Path resolution of an absolute member escapes the target dir.
        safe_extract_zip(zf, target)

