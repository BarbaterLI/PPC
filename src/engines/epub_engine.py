"""EPUB引擎
封装EPUB文件处理逻辑

修复记录 (2026-04-08):
- 将 PPC7Config 改为 PPC8Config，统一配置类型
"""

import asyncio
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from datetime import datetime, timezone

from ..config import PPC8Config
from ..core import BaseEngine
from ..reliability import ExecutionResult, ExecutionMetrics
from .chapter_engine import ChapterEngine

logger = logging.getLogger(__name__)


@dataclass
class EPUBMetadata:
    """EPUB元数据"""
    title: str
    author: str
    language: str
    publisher: str
    published_date: str
    description: str


@dataclass
class EPUBChapter:
    """EPUB章节"""
    id: str
    title: str
    content: str
    order: int


class EPUBEngine(BaseEngine[Path, Dict[str, Any]]):
    """EPUB引擎"""

    def __init__(self, config: PPC8Config):
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

    async def process(
        self,
        input_data: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """处理输入数据并返回结果（统一接口）"""
        output_dir = kwargs.get('output_dir')
        if not output_dir:
            raise ValueError('output_dir is required')
        result = await self.extract(input_data, output_dir)
        if not result.success:
            raise RuntimeError(result.error or 'Extraction failed')
        return result.data

    async def extract(
        self,
        epub_path: Path,
        output_dir: Path
    ) -> ExecutionResult[Dict[str, Any]]:
        """提取EPUB内容"""
        import time
        start_time = time.time()

        try:
            if not epub_path.exists():
                return ExecutionResult.failure(
                    error=f"EPUB文件不存在: {epub_path}",
                    error_code="FILE_NOT_FOUND"
                )

            if not self._is_valid_epub(epub_path):
                return ExecutionResult.failure(
                    error="无效的EPUB文件",
                    error_code="INVALID_EPUB"
                )

            output_dir.mkdir(parents=True, exist_ok=True)

            metadata = self._extract_metadata(epub_path)
            chapters = self._extract_chapters(epub_path)

            if not chapters:
                return ExecutionResult.failure(
                    error="未提取到章节内容",
                    error_code="NO_CHAPTERS"
                )

            content = self._chapters_to_text(chapters)
            result = await self.chapter_engine.split(content, output_dir, metadata.title)

            if result.success:
                result_data = {
                    "metadata": metadata,
                    "chapters_count": len(result.data),
                    "output_files": result.data
                }

                metrics = ExecutionMetrics(
                    duration_seconds=time.time() - start_time,
                    items_processed=len(result.data)
                )

                return ExecutionResult.success(result_data, metrics)

            return result

        except Exception as e:
            logger.error(f"EPUB提取失败: {e}")
            return ExecutionResult.error(
                error=str(e),
                error_code="EXTRACTION_FAILED"
            )

    def _is_valid_epub(self, file_path: Path) -> bool:
        """验证EPUB文件"""
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                return 'mimetype' in zf.namelist() and \
                       zf.read('mimetype').decode('utf-8').strip() == 'application/epub+zip'
        except Exception:
            return False

    def _extract_metadata(self, epub_path: Path) -> EPUBMetadata:
        """提取元数据"""
        metadata = EPUBMetadata(
            title="Unknown",
            author="Unknown",
            language="zh-CN",
            publisher="",
            published_date="",
            description=""
        )

        try:
            with zipfile.ZipFile(epub_path, 'r') as zf:
                container_path = "META-INF/container.xml"
                if container_path not in zf.namelist():
                    return metadata

                container_content = zf.read(container_path).decode('utf-8')
                soup = BeautifulSoup(container_content, 'xml')

                rootfile = soup.find('rootfile')
                if rootfile:
                    opf_path = rootfile.get('full-path', '')

                    try:
                        opf_content = zf.read(opf_path).decode('utf-8')
                        opf_soup = BeautifulSoup(opf_content, 'xml')

                        if opf_soup.find('dc:title'):
                            metadata.title = opf_soup.find('dc:title').get_text()

                        if opf_soup.find('dc:creator'):
                            metadata.author = opf_soup.find('dc:creator').get_text()

                        if opf_soup.find('dc:language'):
                            metadata.language = opf_soup.find('dc:language').get_text()

                        if opf_soup.find('dc:publisher'):
                            metadata.publisher = opf_soup.find('dc:publisher').get_text()

                        if opf_soup.find('dc:date'):
                            metadata.published_date = opf_soup.find('dc:date').get_text()

                        if opf_soup.find('dc:description'):
                            metadata.description = opf_soup.find('dc:description').get_text()

                    except Exception as e:
                        logger.warning(f"解析OPF文件失败: {e}")

        except Exception as e:
            logger.warning(f"提取EPUB元数据失败: {e}")

        return metadata

    def _extract_chapters(self, epub_path: Path) -> List[EPUBChapter]:
        """提取章节"""
        chapters = []

        try:
            with zipfile.ZipFile(epub_path, 'r') as zf:
                container_path = "META-INF/container.xml"
                if container_path not in zf.namelist():
                    return chapters

                container_content = zf.read(container_path).decode('utf-8')
                soup = BeautifulSoup(container_content, 'xml')

                rootfile = soup.find('rootfile')
                if rootfile:
                    opf_path = rootfile.get('full-path', '')
                    opf_dir = str(Path(opf_path).parent) + "/"

                    try:
                        opf_content = zf.read(opf_path).decode('utf-8')
                        opf_soup = BeautifulSoup(opf_content, 'xml')

                        manifest = opf_soup.find('manifest')
                        spine = opf_soup.find('spine')

                        if manifest and spine:
                            itemrefs = spine.find_all('itemref')
                            manifest_items = {
                                item.get('id'): item.get('href')
                                for item in manifest.find_all('item')
                                if item.get('id') and item.get('href')
                            }

                            for i, itemref in enumerate(itemrefs):
                                item_id = itemref.get('itemref')
                                if item_id and item_id in manifest_items:
                                    href = opf_dir + manifest_items[item_id]
                                    try:
                                        html_content = zf.read(href).decode('utf-8')
                                        chapter = self._parse_html_chapter(
                                            item_id,
                                            html_content,
                                            i + 1
                                        )
                                        if chapter:
                                            chapters.append(chapter)
                                    except Exception as e:
                                        logger.warning(f"解析章节失败: {href}, {e}")

                    except Exception as e:
                        logger.warning(f"解析OPF文件失败: {e}")

        except Exception as e:
            logger.warning(f"提取章节失败: {e}")

        return chapters

    def _parse_html_chapter(
        self,
        item_id: str,
        html_content: str,
        order: int
    ) -> Optional[EPUBChapter]:
        """解析HTML章节"""
        soup = BeautifulSoup(html_content, 'html.parser')

        title = ""
        title_tag = soup.find(['h1', 'h2', 'h3', 'title'])
        if title_tag:
            title = title_tag.get_text().strip()

        for tag in soup.find_all(['script', 'style', 'nav']):
            tag.decompose()

        content = soup.get_text(separator='\n', strip=True)

        if len(content) > 10:
            return EPUBChapter(
                id=item_id,
                title=title or f"Chapter {order}",
                content=content,
                order=order
            )

        return None

    def _chapters_to_text(self, chapters: List[EPUBChapter]) -> str:
        """将章节转换为文本"""
        lines = []
        for chapter in sorted(chapters, key=lambda c: c.order):
            lines.append(f"第{chapter.order}章 {chapter.title}")
            lines.append("=" * len(f"第{chapter.order}章 {chapter.title}"))
            lines.append(chapter.content)
            lines.append("")

        return '\n'.join(lines)

    async def extract_text_only(
        self,
        epub_path: Path,
        output_file: Path
    ) -> ExecutionResult[Path]:
        """仅提取文本"""
        import time
        start_time = time.time()

        try:
            metadata = self._extract_metadata(epub_path)
            chapters = self._extract_chapters(epub_path)

            if not chapters:
                return ExecutionResult.failure(
                    error="未提取到章节内容",
                    error_code="NO_CHAPTERS"
                )

            content = self._chapters_to_text(chapters)

            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content, encoding='utf-8')

            metrics = ExecutionMetrics(
                duration_seconds=time.time() - start_time,
                bytes_processed=len(content.encode('utf-8'))
            )

            return ExecutionResult.success(output_file, metrics)

        except Exception as e:
            logger.error(f"EPUB文本提取失败: {e}")
            return ExecutionResult.error(
                error=str(e),
                error_code="EXTRACTION_FAILED"
            )

    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计"""
        return self.chapter_engine.get_stats()
