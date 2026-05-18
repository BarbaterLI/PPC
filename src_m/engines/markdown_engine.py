"""Markdown 文本提取引擎

支持从 Markdown 文件提取文本内容，保留标题层级结构。
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from html import unescape

from src_m.core.base import BaseEngine

logger = logging.getLogger(__name__)

MARKDOWN_AVAILABLE = False
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    pass


@dataclass
class MarkdownSection:
    """Markdown 章节"""
    level: int
    title: str
    content: str
    line_number: int


@dataclass
class MarkdownExtractResult:
    """Markdown 提取结果"""
    sections: List[MarkdownSection]
    full_text: str
    metadata: dict


class MarkdownTextExtractor:
    """Markdown 文本提取器"""

    def __init__(self):
        self._use_markdown_lib = MARKDOWN_AVAILABLE

    def extract(
        self,
        md_path: Path,
        strip_code_blocks: bool = True,
        strip_links: bool = False,
        preserve_structure: bool = True,
    ) -> MarkdownExtractResult:
        """提取 Markdown 文本

        Args:
            md_path: Markdown 文件路径
            strip_code_blocks: 是否移除代码块
            strip_links: 是否移除链接（保留文本）
            preserve_structure: 是否保留结构信息

        Returns:
            MarkdownExtractResult 对象
        """
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown 文件不存在: {md_path}")

        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        sections = []
        full_text_parts = []

        if preserve_structure:
            sections = self._extract_sections(content)
            full_text_parts = [f"{s.title}\n{s.content}" for s in sections]
        else:
            text = self._extract_plain_text(content, strip_code_blocks, strip_links)
            sections = [MarkdownSection(level=0, title="", content=text, line_number=1)]
            full_text_parts = [text]

        metadata = self._extract_metadata(content)

        return MarkdownExtractResult(
            sections=sections,
            full_text="\n\n".join(full_text_parts),
            metadata=metadata,
        )

    def _extract_sections(self, content: str) -> List[MarkdownSection]:
        """提取章节结构"""
        sections = []
        lines = content.split('\n')

        current_section = None
        current_content_lines = []

        for i, line in enumerate(lines, 1):
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                if current_section is not None:
                    current_section.content = '\n'.join(current_content_lines).strip()
                    sections.append(current_section)
                    current_content_lines = []

                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = MarkdownSection(
                    level=level,
                    title=title,
                    content="",
                    line_number=i,
                )
            else:
                if current_section is not None:
                    current_content_lines.append(line)
                else:
                    if line.strip():
                        current_content_lines.append(line)

        if current_section is not None:
            current_section.content = '\n'.join(current_content_lines).strip()
            sections.append(current_section)
        elif current_content_lines:
            sections.append(MarkdownSection(
                level=0,
                title="",
                content='\n'.join(current_content_lines).strip(),
                line_number=1,
            ))

        return sections

    def _extract_plain_text(
        self,
        content: str,
        strip_code_blocks: bool,
        strip_links: bool,
    ) -> str:
        """提取纯文本"""
        text = content

        if strip_code_blocks:
            text = re.sub(r'```[\s\S]*?```', '', text)
            text = re.sub(r'`[^`]+`', '', text)

        if strip_links:
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
            text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)

        text = re.sub(r'[#*_~\[\]()>|+-]+', '', text)

        text = re.sub(r'\n{3,}', '\n\n', text)

        if not MARKDOWN_AVAILABLE:
            return text.strip()

        try:
            html = markdown.markdown(text, extensions=['markdown.extensions.fenced_code'])
            text = self._strip_html(html)
        except Exception:
            pass

        return text.strip()

    def _strip_html(self, html: str) -> str:
        """移除 HTML 标签"""
        text = re.sub(r'<[^>]+>', '', html)
        text = unescape(text)
        return text.strip()

    def _extract_metadata(self, content: str) -> dict:
        """提取元数据（YAML front matter）"""
        metadata = {}

        front_matter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)

        if front_matter_match:
            front_matter = front_matter_match.group(1)
            for line in front_matter.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()

        return metadata

    def get_hierarchy(self, sections: List[MarkdownSection]) -> str:
        """获取章节层级树"""
        tree_lines = []
        for section in sections:
            indent = "  " * (section.level - 1)
            prefix = "├─ " if section.level > 0 else ""
            tree_lines.append(f"{indent}{prefix}[{'#' * section.level}] {section.title}")

        return '\n'.join(tree_lines)

    def is_available(self) -> bool:
        """检查是否可用"""
        return True


class MarkdownEngine(BaseEngine[str, List[str]]):
    """Markdown 处理引擎

    继承 BaseEngine，将 Markdown 文件转换为文本内容。
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self._extractor = MarkdownTextExtractor()

    async def initialize(self) -> None:
        """初始化引擎"""
        await super().initialize()
        if not self._extractor.is_available():
            logger.warning("Markdown 处理库初始化失败")
        logger.info("Markdown 引擎初始化完成")

    async def cleanup(self) -> None:
        """清理引擎资源"""
        await super().cleanup()
        logger.info("Markdown 引擎已清理")

    async def process(self, input_path: Path, **kwargs) -> List[str]:
        """处理 Markdown 文件

        Args:
            input_path: Markdown 文件路径
            **kwargs: 可选参数

        Returns:
            章节内容列表
        """
        strip_code_blocks = kwargs.get("strip_code_blocks", True)
        preserve_structure = kwargs.get("preserve_structure", True)

        result = self._extractor.extract(
            input_path,
            strip_code_blocks=strip_code_blocks,
            preserve_structure=preserve_structure,
        )

        return [section.content for section in result.sections if section.content.strip()]

    def get_sections(self, md_path: Path, **kwargs) -> List[MarkdownSection]:
        """获取章节列表"""
        result = self._extractor.extract(md_path, preserve_structure=True)
        return result.sections

    def get_full_text(self, md_path: Path, **kwargs) -> str:
        """获取完整文本"""
        result = self._extractor.extract(md_path, preserve_structure=False)
        return result.full_text

    def get_hierarchy(self, md_path: Path) -> str:
        """获取章节层级树"""
        sections = self.get_sections(md_path)
        return self._extractor.get_hierarchy(sections)
