"""EPUB 引擎

封装 EPUB 文件解析、元数据提取和章节分割逻辑。
"""

import logging
import posixpath
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from bs4 import BeautifulSoup

from src.config import PPCConfig
from src.core import BaseEngine
from src.core.exceptions import ErrorCodes
from src.engines.chapter_engine import ChapterEngine
from src.reliability import ExecutionMetrics, ExecutionResult

logger = logging.getLogger(__name__)

CONTAINER_XML_PATH = "META-INF/container.xml"
MIMETYPE = "application/epub+zip"
DEFAULT_LANGUAGE = "zh-CN"
MIN_CONTENT_LENGTH = 10
HTML_PARSER = "html.parser"
XML_PARSER = "xml"
ENCODING_UTF8 = "utf-8"


@dataclass
class EPUBMetadata:
    """EPUB 元数据"""

    title: str
    author: str
    language: str
    publisher: str
    published_date: str
    description: str


@dataclass
class EPUBChapter:
    """EPUB 章节"""

    id: str
    title: str
    content: str
    order: int


class EPUBEngine(BaseEngine[Path, dict[str, Any]]):
    """EPUB 引擎

    负责 EPUB 文件的解析、元数据提取、章节分割。
    """

    def __init__(self, config: PPCConfig) -> None:
        super().__init__()
        self.config = config
        self.chapter_engine = ChapterEngine(config)

    async def initialize(self) -> None:
        """初始化引擎"""
        await super().initialize()
        await self.chapter_engine.initialize()
        logger.info("EPUB引擎初始化完成")

    async def cleanup(self) -> None:
        """清理引擎"""
        await super().cleanup()
        await self.chapter_engine.cleanup()
        logger.info("EPUB引擎已清理")

    async def process(self, input_data: Path, /, **kwargs: Any) -> dict[str, Any]:
        """处理输入数据并返回结果（统一接口）"""
        output_dir = kwargs.get("output_dir")
        if not output_dir:
            raise ValueError("output_dir is required")

        result = await self.extract(input_data, output_dir)
        if not result.success:
            raise RuntimeError(result.error or "Extraction failed")
        if result.data is None:
            raise RuntimeError("Extraction returned no data")
        return result.data

    async def extract(self, epub_path: Path, output_dir: Path) -> ExecutionResult[dict[str, Any]]:
        """提取 EPUB 内容"""
        start_time = time.perf_counter()

        try:
            if not epub_path.exists():
                return ExecutionResult.fail(
                    error=f"EPUB文件不存在: {epub_path}", error_code=ErrorCodes.FILE_NOT_FOUND.value
                )

            if not self._is_valid_epub(epub_path):
                return ExecutionResult.fail(error="无效的EPUB文件", error_code=ErrorCodes.INVALID_EPUB.value)

            output_dir.mkdir(parents=True, exist_ok=True)

            metadata, chapters = self._parse_epub(epub_path)

            if not chapters:
                return ExecutionResult.fail(error="未提取到章节内容", error_code=ErrorCodes.NO_CHAPTERS.value)

            content = self._chapters_to_text(chapters)
            result = await self.chapter_engine.split(content, output_dir, metadata.title)

            chapter_files = result.data or []
            if result.success:
                result_data = {
                    "metadata": metadata,
                    "chapters_count": len(chapter_files),
                    "output_files": chapter_files,
                }
                metrics = ExecutionMetrics(
                    duration=time.perf_counter() - start_time,
                    bytes_processed=len(chapter_files),
                )
                return ExecutionResult.ok(result_data, metrics)

            return ExecutionResult[dict[str, Any]].fail(
                error=result.error or "Chapter split failed",
                error_code=result.error_code,
                metrics=result.metrics,
            )

        except Exception as e:
            logger.error(f"EPUB提取失败: {e}")
            return ExecutionResult.fail(error=str(e), error_code=ErrorCodes.EPUB_EXTRACTION_FAILED.value)

    @staticmethod
    def _is_valid_epub(file_path: Path) -> bool:
        """验证 EPUB 文件格式"""
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                return MIMETYPE in zf.namelist() and zf.read(MIMETYPE).decode(ENCODING_UTF8).strip() == MIMETYPE
        except Exception:
            return False

    def _parse_epub(self, epub_path: Path) -> tuple[EPUBMetadata, list[EPUBChapter]]:
        """单次 zip 遍历提取元数据与章节，避免重复打开文件/重复解析 OPF。"""
        metadata = EPUBMetadata(
            title="Unknown",
            author="Unknown",
            language=DEFAULT_LANGUAGE,
            publisher="",
            published_date="",
            description="",
        )
        chapters: list[EPUBChapter] = []
        try:
            with zipfile.ZipFile(epub_path, "r") as zf:
                opf_path = self._get_opf_path(zf)
                if not opf_path:
                    return metadata, chapters
                self._parse_opf_metadata(zf, opf_path, metadata)
                opf_dir = str(Path(opf_path).parent) + "/"
                self._parse_manifest_and_spine(zf, opf_path, opf_dir, chapters)
        except Exception as e:
            logger.warning(f"解析EPUB失败: {e}")
        return metadata, chapters

    @staticmethod
    def _get_opf_path(zf: zipfile.ZipFile) -> str | None:
        """从 container.xml 获取 OPF 文件路径"""
        if CONTAINER_XML_PATH not in zf.namelist():
            return None

        container_content = zf.read(CONTAINER_XML_PATH).decode(ENCODING_UTF8)
        soup = BeautifulSoup(container_content, XML_PARSER)
        rootfile = soup.find("rootfile")

        return str(rootfile.get("full-path", "")) if rootfile else None

    def _parse_opf_metadata(self, zf: zipfile.ZipFile, opf_path: str, metadata: EPUBMetadata) -> None:
        """解析 OPF 文件中的元数据"""
        try:
            opf_content = zf.read(opf_path).decode(ENCODING_UTF8)
            opf_soup = BeautifulSoup(opf_content, XML_PARSER)

            tag_mapping = {
                "dc:title": "title",
                "dc:creator": "author",
                "dc:language": "language",
                "dc:publisher": "publisher",
                "dc:date": "published_date",
                "dc:description": "description",
            }

            for tag_name, attr_name in tag_mapping.items():
                tag = opf_soup.find(tag_name)
                if tag:
                    setattr(metadata, attr_name, tag.get_text())

        except Exception as e:
            logger.warning(f"解析OPF文件失败: {e}")

    def _parse_manifest_and_spine(
        self,
        zf: zipfile.ZipFile,
        opf_path: str,
        opf_dir: str,
        chapters: list[EPUBChapter],
    ) -> None:
        """解析 manifest 和 spine 以提取章节"""
        try:
            opf_content = zf.read(opf_path).decode(ENCODING_UTF8)
            opf_soup = BeautifulSoup(opf_content, XML_PARSER)

            manifest = opf_soup.find("manifest")
            spine = opf_soup.find("spine")

            if not manifest or not spine:
                return

            manifest_items = {
                item.get("id"): item.get("href")
                for item in manifest.find_all("item")
                if item.get("id") and item.get("href")
            }

            for i, itemref in enumerate(spine.find_all("itemref"), start=1):
                item_id = str(itemref.get("idref")) if itemref.get("idref") else None
                if item_id and item_id in manifest_items:
                    href = opf_dir + str(manifest_items[item_id])
                    self._parse_chapter_from_href(zf, href, item_id, i, chapters)

        except Exception as e:
            logger.warning(f"解析OPF文件失败: {e}")

    @staticmethod
    def _href_candidates(href: str) -> list[str]:
        """生成 manifest href 对应的候选 zip 路径（去重、保序）。

        EPUB 的 ``href`` 是 **URI 引用**：可能百分号编码（``chap%201.xhtml``）、
        含 ``../`` 相对路径；而 zip 内是解码后的规范名。原实现只用原始 href
        直读，编码/相对路径的章节 ``zf.read`` 抛 KeyError 被吞 → **章节静默丢失**。
        候选顺序：原始 → 百分号解码 → 规范化（``../`` 归一），原始优先保证已能
        读取的 EPUB 行为不变（无回归）。
        """
        candidates: list[str] = []
        decoded = unquote(href)
        normalized = posixpath.normpath(decoded)
        for cand in (href, decoded, normalized):
            if cand and cand not in candidates:
                candidates.append(cand)
        return candidates

    def _parse_chapter_from_href(
        self,
        zf: zipfile.ZipFile,
        href: str,
        item_id: str,
        order: int,
        chapters: list[EPUBChapter],
    ) -> None:
        """从指定路径解析章节内容（依次尝试各候选 zip 路径）。"""
        for candidate in self._href_candidates(href):
            try:
                html_content = zf.read(candidate).decode(ENCODING_UTF8)
            except KeyError:
                continue  # 该候选不在 zip 内，尝试下一个（解码/规范化路径）
            except Exception as e:  # noqa: BLE001
                logger.warning(f"解析章节失败: {candidate}, {e}")
                return
            chapter = self._parse_html_chapter(item_id, html_content, order)
            if chapter:
                chapters.append(chapter)
            return
        logger.warning(f"章节未能从任何候选路径读取: {href}")

    def _parse_html_chapter(self, item_id: str, html_content: str, order: int) -> EPUBChapter | None:
        """解析 HTML 章节内容"""
        soup = BeautifulSoup(html_content, HTML_PARSER)

        title = self._extract_title(soup, order)
        self._remove_non_content_tags(soup)
        content = soup.get_text(separator="\n", strip=True)

        if len(content) > MIN_CONTENT_LENGTH:
            return EPUBChapter(
                id=item_id,
                title=title,
                content=content,
                order=order,
            )
        return None

    @staticmethod
    def _extract_title(soup: BeautifulSoup, order: int) -> str:
        """从 HTML 中提取标题"""
        title_tag = soup.find(["h1", "h2", "h3", "title"])
        if title_tag:
            return title_tag.get_text().strip()
        return f"Chapter {order}"

    @staticmethod
    def _remove_non_content_tags(soup: BeautifulSoup) -> None:
        """移除脚本、样式和导航等非内容标签"""
        for tag in soup.find_all(["script", "style", "nav"]):
            tag.decompose()

    def _chapters_to_text(self, chapters: list[EPUBChapter]) -> str:
        """将章节列表转换为纯文本"""
        lines = []
        for chapter in sorted(chapters, key=lambda c: c.order):
            header = f"第{chapter.order}章 {chapter.title}"
            lines.append(header)
            lines.append("=" * len(header))
            lines.append(chapter.content)
            lines.append("")

        return "\n".join(lines)

    async def extract_text_only(self, epub_path: Path, output_file: Path) -> ExecutionResult[Path]:
        """仅提取文本内容，不分割章节"""
        start_time = time.perf_counter()

        try:
            chapters = self._parse_epub(epub_path)[1]
            if not chapters:
                return ExecutionResult.fail(error="未提取到章节内容", error_code=ErrorCodes.NO_CHAPTERS.value)

            content = self._chapters_to_text(chapters)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content, encoding=ENCODING_UTF8)

            metrics = ExecutionMetrics(
                duration=time.perf_counter() - start_time,
                bytes_processed=len(content.encode(ENCODING_UTF8)),
            )
            return ExecutionResult.ok(output_file, metrics)

        except Exception as e:
            logger.error(f"EPUB文本提取失败: {e}")
            return ExecutionResult.fail(error=str(e), error_code=ErrorCodes.EPUB_EXTRACTION_FAILED.value)

    def get_stats(self) -> dict[str, Any]:
        """获取引擎统计信息"""
        return self.chapter_engine.get_stats()
