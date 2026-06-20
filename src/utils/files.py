from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ENCODING_FALLBACK = ["utf-8", "gbk", "gb2312", "utf-16"]
DEFAULT_ENCODING_DETECT_BUFFER = 1024


def safe_extract_zip(zip_file: zipfile.ZipFile, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    for info in zip_file.infolist():
        if info.is_dir():
            continue
        extracted_path = (target_dir / info.filename).resolve()
        if not str(extracted_path).startswith(str(target_dir)):
            raise ValueError(f"ZIP path traversal detected: {info.filename}")
        target_dir_for_file = extracted_path.parent
        target_dir_for_file.mkdir(parents=True, exist_ok=True)
        with zip_file.open(info) as src, open(extracted_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def detect_encoding(
    file_path: Path,
    encodings: list[str] | None = None,
    detect_buffer: int = DEFAULT_ENCODING_DETECT_BUFFER,
) -> str | None:
    if encodings is None:
        encodings = DEFAULT_ENCODING_FALLBACK

    for encoding in encodings:
        try:
            with file_path.open("r", encoding=encoding) as f:
                f.read(detect_buffer)
            logger.debug(f"编码检测成功: {file_path.name} -> {encoding}")
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue

    logger.warning(f"编码检测失败: {file_path.name}")
    return None


__all__ = [
    "safe_extract_zip",
    "detect_encoding",
    "DEFAULT_ENCODING_FALLBACK",
    "DEFAULT_ENCODING_DETECT_BUFFER",
]
