"""Unit tests for the SuperSplit filters module."""

from __future__ import annotations

from src.extensions.supersplit.filters import (
    apply_exclude_patterns,
    apply_replace_rules,
    clean_whitespace,
)

# ---------------------------------------------------------------------------
# apply_exclude_patterns
# ---------------------------------------------------------------------------


def test_apply_exclude_patterns_basic():
    text = "line one\nTODO: fix this\nline three"
    result = apply_exclude_patterns(text, [r"TODO.*"])
    assert "TODO" not in result
    assert "line one" in result
    assert "line three" in result


def test_apply_exclude_patterns_empty():
    text = "some text"
    assert apply_exclude_patterns(text, []) == text


def test_apply_exclude_patterns_invalid_regex():
    text = "hello [world"
    result = apply_exclude_patterns(text, ["["])
    # invalid regex falls back to literal removal
    assert "[" not in result


# ---------------------------------------------------------------------------
# apply_replace_rules
# ---------------------------------------------------------------------------


def test_apply_replace_rules_basic():
    result = apply_replace_rules("hello world", [("hello", "hi")])
    assert result == "hi world"


def test_apply_replace_rules_multiple():
    result = apply_replace_rules("hello world", [("hello", "hi"), ("world", "earth")])
    assert result == "hi earth"


def test_apply_replace_rules_empty():
    assert apply_replace_rules("text", []) == "text"


def test_apply_replace_rules_invalid_regex():
    result = apply_replace_rules("hello [world", [("[", "")])
    # invalid regex falls back to literal replacement
    assert "[" not in result


# ---------------------------------------------------------------------------
# clean_whitespace
# ---------------------------------------------------------------------------


def test_clean_whitespace_normalize_newlines():
    assert clean_whitespace("a\r\nb\rc") == "a\nb\nc"


def test_clean_whitespace_collapse_spaces():
    assert clean_whitespace("a   b\t\tc") == "a b c"


def test_clean_whitespace_compress_newlines():
    assert clean_whitespace("a\n\n\n\nb") == "a\n\nb"


def test_clean_whitespace_strip_lines():
    # right-strips trailing whitespace from each line (leading whitespace kept)
    assert clean_whitespace("hello   \nworld   ") == "hello\nworld"


def test_clean_whitespace_compress_toggle():
    text = "a\n\n\n\nb"
    # with compress_newlines=False, 3+ newlines are kept
    result = clean_whitespace(text, compress_newlines=False)
    assert "\n\n\n" in result
