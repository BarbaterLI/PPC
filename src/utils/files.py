from __future__ import annotations

import codecs
import logging
import os
import re
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

#: 编码候选链。末尾的 ``utf-16-le`` / ``utf-16-be`` 是无 BOM 时唯一的正确解：
#: ``utf-16`` 解码器要求开头有 BOM，否则直接报错。
DEFAULT_ENCODING_FALLBACK = ["utf-8", "gb18030", "gbk", "gb2312", "utf-16", "utf-16-le", "utf-16-be"]
DEFAULT_ENCODING_DETECT_BUFFER = 1024

#: 解码后「明显不该出现在正文里」的字符：NUL、除 \t\n\r 外的 C0 控制符、
#: 替换符 U+FFFD、私用区。UTF-16 文本被单字节编码解码时几乎必然大量命中。
_SUSPICIOUS_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd\ue000-\uf8ff]")

#: 可疑字符占比阈值。UTF-16LE 的 ASCII 字符（数字/标点/空格）经 utf-8 或 gbk
#: 解码后每 2 字节里就有 1 个 NUL，占比 ~50%；正常文本为 0。
_SUSPICION_LIMIT = 0.02


def _decode_suspicion(text: str) -> float:
    """返回**可疑字符占比**（0..1）。用于在候选链里排除「能解码但内容明显坏掉」。

    触发条件：无 BOM 的 UTF-16LE 章节（Windows 记事本「Unicode」另存会产生）。
    旧行为：``_head_decodes`` 只问「能不能解码」——而 gbk 双字节编码几乎能解
    任意字节对，UTF-16 文本会被判成 gbk 并**静默解出乱码**（不抛异常），
    整本书被当正文送去合成。
    """
    if not text:
        return 0.0
    return len(_SUSPICIOUS_CHARS.findall(text)) / len(text)



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
    """增量解码器校验头部（容忍缓冲区末尾截断的多字节字符）。

    除了「能否解码」，还要看「解出来的东西像不像正文」：UTF-16 文本被 gbk /
    utf-8 解码通常能成功，但会产出大量 NUL 与控制符。以前只看前者，于是
    无 BOM 的 UTF-16LE 会被判成 gbk 并静默乱码。
    """
    try:
        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        text = decoder.decode(head, final=False)
    except (UnicodeDecodeError, UnicodeError, LookupError):
        return False
    return _decode_suspicion(text) <= _SUSPICION_LIMIT


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


BOM_CHAR = "\ufeff"


def strip_bom(text: str) -> str:
    """去除**文本开头**的 BOM（U+FEFF）。

    ``DEFAULT_ENCODING_FALLBACK`` 以 ``utf-8`` 打头，而 ``utf-8`` 解码**不会**
    剥离 BOM，于是 ``\\ufeff`` 会留在正文首位。这会让按行首匹配的逻辑失配
    （典型：章节正则 ``^第…章`` 不匹配首行 → 首章被并入下一章），并可能被
    当作正文字符送入 TTS 朗读。

    只剥离首个字符；正文中间的 U+FEFF 保留（可能是零宽不换行空格，语义不同）。

    参数:
        text: 已解码的文本。

    返回:
        去掉开头 BOM 的文本。
    """
    return text[1:] if text.startswith(BOM_CHAR) else text


def read_text_auto(
    file_path: Path, encodings: list[str] | None = None, detect_buffer: int = DEFAULT_ENCODING_DETECT_BUFFER
) -> tuple[str, str]:
    """一次轻量头部探测确定编码，再用该编码做一次全量 ``read_text``。

    返回 ``(文本, 编码)``。合并了旧 “detect_encoding(开一次) + read_text(再开
    一次并再解码)” 的两趟探测——编码判定只读头部，全量读与解码各一次。
    全部候选解码失败时按首候选编码严格解码，让 ``UnicodeDecodeError`` 上抛
    （与旧路径异常语义一致）。

    返回值中的文本已由 :func:`strip_bom` 去除开头的 BOM。
    """
    if encodings is None:
        encodings = DEFAULT_ENCODING_FALLBACK

    encoding, _ = quick_probe(file_path, encodings=encodings, detect_buffer=detect_buffer)
    if encoding is None:
        encoding = encodings[0] if encodings else "utf-8"
    try:
        return strip_bom(file_path.read_text(encoding=encoding)), encoding
    except (UnicodeDecodeError, UnicodeError):
        # 头部合法但正文出现非法字节（罕见）：按候选链重试一次
        for candidate in encodings:
            if candidate == encoding:
                continue
            try:
                return strip_bom(file_path.read_text(encoding=candidate)), candidate
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise


def _part_path(path: Path) -> Path:
    """同目录 ``.part`` 临时文件路径（``a.mp3`` → ``a.mp3.part``）。"""
    return path.with_name(path.name + ".part")


def _commit_part(tmp_path: Path, path: Path) -> None:
    """``fsync`` 后 ``os.replace`` 原子替换目标；失败时清掉临时文件。"""
    try:
        with open(tmp_path, "rb+") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp_path), str(path))
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Path | str, data: bytes) -> Path:
    """原子地把 *data* 写到 *path*（同目录 ``.part`` + ``fsync`` + ``os.replace``）。

    触发条件：任何「先落盘、失败即视为脏数据」的产物写出（例如 TTS 插件
    后端直接返回音频字节）。
    后果（旧实现直接 ``write_bytes``）：进程在写一半时被中断/崩溃，目标路径
    上会留下截断的半成品，而调用方/缓存把它当作完整产物。
    改为先写 ``<name>.part``、``fsync`` 后再原子替换；任何异常都会删除临时
    文件并向上抛，目标路径要么是旧内容、要么不存在，绝不会是半截文件。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _part_path(target)
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    _commit_part(tmp, target)
    return target


def atomic_copy_file(src: Path | str, dst: Path | str) -> Path:
    """流式复制 *src* → *dst*，以 ``.part`` + ``fsync`` + ``os.replace`` 落盘。

    与 :func:`atomic_write_bytes` 同为原子落盘工具，供「插件已生成文件、需要
    搬移到目标路径」的路径复用（旧实现用 ``shutil.copy2`` 直接写目标路径，
    中途失败会留下半成品）。不保留源文件的 mtime/权限——音频产物的可播放性
    只取决于字节完整性。
    """
    source = Path(src)
    target = Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _part_path(target)
    try:
        with open(source, "rb") as fsrc, open(tmp, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
            fdst.flush()
            os.fsync(fdst.fileno())
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    _commit_part(tmp, target)
    return target


__all__ = [
    "BOM_CHAR",
    "atomic_copy_file",
    "atomic_write_bytes",
    "safe_extract_zip",
    "detect_encoding",
    "quick_probe",
    "read_text_auto",
    "strip_bom",
    "DEFAULT_ENCODING_FALLBACK",
    "DEFAULT_ENCODING_DETECT_BUFFER",
]
