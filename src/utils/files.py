from __future__ import annotations

import codecs
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
        try:
            # 边界匹配（非前缀字符串匹配）：拒绝任何逃逸到目标目录之外的条目
            extracted_path.relative_to(target_dir)
        except ValueError:
            raise ValueError(f"ZIP path traversal detected: {info.filename}") from None
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


def _head_decodes(head: bytes, encoding: str) -> bool:
    """增量解码器校验头部（容忍缓冲区末尾截断的多字节字符）。"""
    try:
        codecs.getincrementaldecoder(encoding)(errors="strict").decode(head, final=False)
        return True
    except (UnicodeDecodeError, UnicodeError, LookupError):
        return False


def quick_probe(
    file_path: Path, encodings: list[str] | None = None, detect_buffer: int = DEFAULT_ENCODING_DETECT_BUFFER
) -> tuple[str | None, int]:
    """单遍轻量探测：一次 ``open`` + 头部读 + ``stat``，返回 ``(编码, 字节大小)``。

    用于入队/恢复阶段的“是否存在/进度分母”估算——不做全文读取，也不调用
    ``detect_encoding``。第二个返回值是字节数（非字符数），CJK UTF-8 文本约
    高估 1/3，仅供启发式使用。
    """
    if encodings is None:
        encodings = DEFAULT_ENCODING_FALLBACK

    try:
        size = file_path.stat().st_size
        with file_path.open("rb") as fh:
            head = fh.read(max(4, detect_buffer))
    except OSError:
        return None, 0

    for encoding in encodings:
        if _head_decodes(head, encoding):
            return encoding, size

    logger.warning(f"编码检测失败: {file_path.name}")
    return None, size


def read_text_auto(
    file_path: Path, encodings: list[str] | None = None, detect_buffer: int = DEFAULT_ENCODING_DETECT_BUFFER
) -> tuple[str, str]:
    """一次轻量头部探测确定编码，再用该编码做一次全量 ``read_text``。

    返回 ``(文本, 编码)``。合并了旧 “detect_encoding(开一次) + read_text(再开
    一次并再解码)” 的两趟探测——编码判定只读头部，全量读与解码各一次。
    全部候选解码失败时按首候选编码严格解码，让 ``UnicodeDecodeError`` 上抛
    （与旧路径异常语义一致）。
    """
    if encodings is None:
        encodings = DEFAULT_ENCODING_FALLBACK

    encoding, _ = quick_probe(file_path, encodings=encodings, detect_buffer=detect_buffer)
    if encoding is None:
        encoding = encodings[0] if encodings else "utf-8"
    try:
        return file_path.read_text(encoding=encoding), encoding
    except (UnicodeDecodeError, UnicodeError):
        # 头部合法但正文出现非法字节（罕见）：按候选链重试一次
        for candidate in encodings:
            if candidate == encoding:
                continue
            try:
                return file_path.read_text(encoding=candidate), candidate
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise


__all__ = [
    "safe_extract_zip",
    "detect_encoding",
    "quick_probe",
    "read_text_auto",
    "DEFAULT_ENCODING_FALLBACK",
    "DEFAULT_ENCODING_DETECT_BUFFER",
]
