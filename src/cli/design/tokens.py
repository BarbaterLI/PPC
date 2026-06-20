"""Centralized design tokens for PPC10 CLI output.

This module defines a single source of truth for colors, spacing, icons,
typography and Rich style descriptions used across the CLI. All visual
values live in the token dataclasses below; helper functions only read from
these tokens and respect the global ``no_color`` / ``no_emoji`` flags.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.style import Style

# ---------------------------------------------------------------------------
# Global CLI flags (set by typer_app after argument parsing)
# ---------------------------------------------------------------------------

_NO_COLOR: bool = False
_NO_EMOJI: bool = False


def set_no_color(value: bool) -> None:
    """Set the global ``--no-color`` flag.

    When enabled, :func:`get_style` returns a plain, unstyled ``Style``.
    """
    global _NO_COLOR
    _NO_COLOR = bool(value)


def set_no_emoji(value: bool) -> None:
    """Set the global ``--no-emoji`` flag.

    When enabled, :func:`get_icon` returns the ASCII fallback glyph.
    """
    global _NO_EMOJI
    _NO_EMOJI = bool(value)


# ---------------------------------------------------------------------------
# Color tokens (semantic hex palette)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColorTokens:
    """Semantic color tokens.

    Values are hex colors suitable for Rich markup or ``Style`` construction.
    """

    success: str = "#27AE60"
    error: str = "#E74C3C"
    warning: str = "#F1C40F"
    info: str = "#3498DB"
    accent: str = "#F39C12"
    primary: str = "#4A90D9"
    secondary: str = "#2ECC71"
    muted: str = "#7F8C8D"
    border: str = "#ECF0F1"


COLORS = ColorTokens()


# ---------------------------------------------------------------------------
# Spacing tokens
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpacingTokens:
    """Spacing tokens measured in terminal cells / characters."""

    xs: int = 1
    sm: int = 2
    md: int = 4
    lg: int = 6
    xl: int = 8


SPACING = SpacingTokens()


# ---------------------------------------------------------------------------
# Icon tokens (ASCII default + optional emoji fallback)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IconTokens:
    """Icon glyphs.

    Each field is a tuple ``(ascii, emoji)``. The ASCII glyph is the default;
    the emoji glyph is returned when ``no_emoji`` is False.
    """

    success: tuple[str, str] = ("+", "\u2713")
    error: tuple[str, str] = ("-", "\u2717")
    warning: tuple[str, str] = ("!", "\u26a0")
    info: tuple[str, str] = ("i", "\u2139")
    pending: tuple[str, str] = ("o", "\u25cb")
    running: tuple[str, str] = ("*", "\u25d0")
    completed: tuple[str, str] = ("+", "\u2713")
    skipped: tuple[str, str] = (">", "\u2192")
    gear: tuple[str, str] = ("G", "\u2699")
    rocket: tuple[str, str] = ("R", "\U0001f680")
    book: tuple[str, str] = ("B", "\U0001f4d6")
    chart: tuple[str, str] = ("C", "\U0001f4ca")
    clock: tuple[str, str] = ("T", "\u23f1")
    file: tuple[str, str] = ("F", "\U0001f4c4")
    folder: tuple[str, str] = ("D", "\U0001f4c1")
    sound: tuple[str, str] = ("S", "\U0001f50a")
    microphone: tuple[str, str] = ("M", "\U0001f3a4")
    star: tuple[str, str] = ("*", "\u2b50")
    link: tuple[str, str] = ("L", "\U0001f517")


ICONS = IconTokens()


# ---------------------------------------------------------------------------
# Typography tokens (Rich-parseable style descriptions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypographyTokens:
    """Typography style descriptions parseable by ``rich.style.Style.parse``."""

    bold: str = "bold"
    dim: str = "dim"
    header: str = "bold white"
    header_success: str = "bold green"
    header_error: str = "bold red"
    header_warning: str = "bold yellow"
    header_info: str = "bold cyan"
    body: str = "none"
    muted: str = "dim"


TYPOGRAPHY = TypographyTokens()


# ---------------------------------------------------------------------------
# Style lookup table
# ---------------------------------------------------------------------------

STYLE_TOKENS: dict[str, str] = {
    # Semantic status styles
    "success": "bold green",
    "error": "bold red",
    "warning": "bold yellow",
    "info": "cyan",
    "debug": "dim cyan",
    # Brand / accent styles
    "accent": f"bold {COLORS.accent}",
    "primary": COLORS.primary,
    "secondary": COLORS.secondary,
    "muted": TYPOGRAPHY.dim,
    "border": COLORS.border,
    # Typography styles
    "bold": TYPOGRAPHY.bold,
    "dim": TYPOGRAPHY.dim,
    "header": TYPOGRAPHY.header,
    "header_success": TYPOGRAPHY.header_success,
    "header_error": TYPOGRAPHY.header_error,
    "header_warning": TYPOGRAPHY.header_warning,
    "header_info": TYPOGRAPHY.header_info,
    "body": TYPOGRAPHY.body,
}


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def get_icon(name: str) -> str:
    """Return the glyph for ``name`` respecting the ``no_emoji`` flag.

    Args:
        name: One of the field names on :data:`ICONS`.

    Returns:
        ASCII glyph when ``no_emoji`` is set, otherwise the emoji glyph.

    Raises:
        KeyError: If ``name`` is not a registered icon token.
    """
    if not hasattr(ICONS, name):
        raise KeyError(f"Unknown icon name: {name!r}. Valid: {sorted(ICONS.__dataclass_fields__)}")
    ascii_glyph, emoji_glyph = getattr(ICONS, name)
    ascii_glyph = str(ascii_glyph)
    emoji_glyph = str(emoji_glyph)
    if _NO_EMOJI:
        return ascii_glyph
    return emoji_glyph


def get_style(name: str) -> Style:
    """Return a ``Rich`` Style for ``name`` respecting the ``no_color`` flag.

    Args:
        name: One of the keys in :data:`STYLE_TOKENS`.

    Returns:
        A plain ``Style()`` when ``no_color`` is set, otherwise the parsed
        style from the token table.

    Raises:
        KeyError: If ``name`` is not a registered style token.
    """
    if name not in STYLE_TOKENS:
        raise KeyError(f"Unknown style name: {name!r}. Valid: {sorted(STYLE_TOKENS)}")
    if _NO_COLOR:
        return Style()
    return Style.parse(STYLE_TOKENS[name])
