"""EPUB 引擎章节提取测试（manifest href 解码/规范化）。"""

from __future__ import annotations

import zipfile
from pathlib import Path

from src.config import get_preset
from src.engines.epub_engine import EPUBEngine

_CONTAINER = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""

_OPF = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>T</dc:title></metadata>
<manifest><item id="ch1" href="{href}" media-type="application/xhtml+xml"/></manifest>
<spine><itemref idref="ch1"/></spine>
</package>"""

_XHTML = "<html><body><h1>Title</h1><p>" + "这是一段足够长的章节正文用于超过最小长度限制。" * 3 + "</p></body></html>"


def _build_epub(path: Path, href: str, filename: str) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", _CONTAINER)
        zf.writestr("OEBPS/content.opf", _OPF.format(href=href))
        zf.writestr(f"OEBPS/{filename}", _XHTML)


def _engine() -> EPUBEngine:
    return EPUBEngine(get_preset("balanced"))


def test_url_encoded_href_chapter_extracted(tmp_path: Path) -> None:
    """manifest ``href`` 百分号编码时章节仍应被提取（不得静默丢失）。

    Bug: ``_parse_chapter_from_href`` 只用原始 href 直读 zip。EPUB 的 href 是
    URI 引用（``chap%201.xhtml``），而 zip 内是解码名（``chap 1.xhtml``），
    ``zf.read`` 抛 KeyError 被 ``except`` 吞掉 → **章节静默丢失**
    （实测 URL 编码 chapters=0，未编码 chapters=1）。
    修复: 依次尝试 原始 → 百分号解码 → 规范化 候选路径。
    """
    p = tmp_path / "enc.epub"
    _build_epub(p, href="chap%201.xhtml", filename="chap 1.xhtml")

    _md, chapters = _engine()._parse_epub(p)

    assert len(chapters) == 1, f"URL 编码 href 的章节被静默丢失: chapters={len(chapters)}"


def test_plain_href_still_extracted(tmp_path: Path) -> None:
    """未编码 href 行为不变（无回归）。"""
    p = tmp_path / "plain.epub"
    _build_epub(p, href="ch1.xhtml", filename="ch1.xhtml")

    _md, chapters = _engine()._parse_epub(p)

    assert len(chapters) == 1


def test_relative_parent_href_extracted(tmp_path: Path) -> None:
    """含 ``../`` 的相对 href 规范化后仍应命中 zip 内条目。"""
    # OPF 在 OEBPS/ 下，章节实际在 zip 根的 text/ch1.xhtml（href 用 ../ 指回）
    path = tmp_path / "rel.epub"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", _CONTAINER)
        zf.writestr("OEBPS/content.opf", _OPF.format(href="../text/ch1.xhtml"))
        zf.writestr("text/ch1.xhtml", _XHTML)

    _md, chapters = _engine()._parse_epub(path)

    assert len(chapters) == 1, f"../ 相对 href 未被规范化，章节丢失: {len(chapters)}"
