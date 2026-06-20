from __future__ import annotations

import re

FILENAME_SANITIZE_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
DEFAULT_FILENAME_MAX_LENGTH = 100


def sanitize_filename(filename: str, max_length: int = DEFAULT_FILENAME_MAX_LENGTH) -> str:
    cleaned = FILENAME_SANITIZE_PATTERN.sub("_", filename)
    cleaned = cleaned.strip(". ")
    return cleaned[:max_length]


__all__ = [
    "sanitize_filename",
    "FILENAME_SANITIZE_PATTERN",
    "DEFAULT_FILENAME_MAX_LENGTH",
]
