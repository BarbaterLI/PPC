"""HTML 文本提取引擎

支持从 HTML 文件/网页提取文本内容，保留语义结构。
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.base import BaseEngine

logger = logging.getLogger(__name__)

BS4_AVAILABLE = False
try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    pass


@dataclass
class HTMLElement:
    """HTML 元素"""

    tag: str
    text: str
    level: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class HTMLExtractResult:
    """HTML 提取结果"""

    elements: list[HTMLElement]
    full_text: str
    title: str
    metadata: dict


class HTMLTextExtractor:
    """HTML 文本提取器"""

    def __init__(self) -> None:
        self._use_bs4: bool = BS4_AVAILABLE

    def extract(
        self,
        html_source: str | Path,
        strip_scripts: bool = True,
        strip_styles: bool = True,
        preserve_structure: bool = True,
    ) -> HTMLExtractResult:
        """提取 HTML 文本

        Args:
            html_source: HTML 文件路径或 HTML 字符串
            strip_scripts: 是否移除脚本标签
            strip_styles: 是否移除样式标签
            preserve_structure: 是否保留结构信息

        Returns:
            HTMLExtractResult 对象
        """
        if isinstance(html_source, Path):
            if not html_source.exists():
                raise FileNotFoundError(f"HTML 文件不存在: {html_source}")
            with open(html_source, encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
        else:
            html_content = html_source

        if not self._use_bs4:
            return self._extract_without_bs4(html_content, preserve_structure)

        soup = BeautifulSoup(html_content, "lxml")

        if strip_scripts:
            for script in soup.find_all("script"):
                script.decompose()
            for script in soup.find_all("noscript"):
                script.decompose()

        if strip_styles:
            for style in soup.find_all("style"):
                style.decompose()

        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text().strip()

        metadata = self._extract_metadata(soup)

        elements = []
        full_text_parts = []

        if preserve_structure:
            elements = self._extract_structured_text(soup)
            full_text_parts = [elem.text.strip() for elem in elements if elem.text.strip()]
        else:
            text = self._get_plain_text(soup)
            elements = [HTMLElement(tag="body", text=text, level=0)]
            full_text_parts = [text]

        return HTMLExtractResult(
            elements=elements,
            full_text="\n\n".join(full_text_parts),
            title=title,
            metadata=metadata,
        )

    def _extract_without_bs4(self, html_content: str, preserve_structure: bool) -> HTMLExtractResult:
        """不使用 BeautifulSoup 提取"""
        text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html_content)
        text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        title_match = re.search(r"<title[^>]*>([^<]+)</title>", html_content, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""

        element = HTMLElement(tag="body", text=text, level=0)

        return HTMLExtractResult(
            elements=[element],
            full_text=text,
            title=title,
            metadata={},
        )

    def _extract_structured_text(self, soup: BeautifulSoup) -> list[HTMLElement]:
        """提取结构化文本"""
        elements = []

        heading_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]
        p_tags = ["p", "div", "article", "section"]

        heading_levels = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

        for tag in heading_tags:
            for heading in soup.find_all(tag):
                text = self._clean_text(heading)
                if text:
                    elements.append(
                        HTMLElement(
                            tag=tag,
                            text=text,
                            level=heading_levels.get(tag, 0),
                        )
                    )

        for tag in p_tags:
            for para in soup.find_all(tag):
                if para.name in heading_tags:
                    continue

                text = self._clean_text(para)

                if text and len(text) > 20:
                    is_heading = False
                    for child in para.find_all(recursive=False):
                        if child.name in heading_tags:
                            is_heading = True
                            break

                    if not is_heading:
                        elements.append(
                            HTMLElement(
                                tag=tag,
                                text=text,
                                level=0,
                            )
                        )

        list_items = soup.find_all(["li", "dd"])
        for item in list_items:
            text = self._clean_text(item)
            if text and len(text) > 5:
                elements.append(
                    HTMLElement(
                        tag=item.name,
                        text=text,
                        level=0,
                    )
                )

        return elements

    def _get_plain_text(self, soup: BeautifulSoup) -> str:
        """获取纯文本"""
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _clean_text(self, element) -> str:
        """清理元素文本"""
        text = element.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_metadata(self, soup: BeautifulSoup) -> dict:
        """提取元数据"""
        metadata = {}

        meta_tags = {
            "description": ["description"],
            "keywords": ["keywords"],
            "author": ["author"],
        }

        for key, names in meta_tags.items():
            for name in names:
                tag = soup.find("meta", attrs={"name": name})
                if tag and tag.get("content"):
                    metadata[key] = tag.get("content")
                    break

        return metadata

    def is_available(self) -> bool:
        """检查是否可用"""
        return True


class HTMLEngine(BaseEngine[Path, list[str]]):
    """HTML 处理引擎

    继承 BaseEngine，将 HTML 文件转换为文本内容。
    """

    def __init__(self, config: Any = None) -> None:
        super().__init__()
        self.config = config
        self._extractor = HTMLTextExtractor()

    async def initialize(self) -> None:
        """初始化引擎"""
        await super().initialize()
        if not self._extractor.is_available():
            logger.warning("HTML 处理库初始化失败")
        logger.info("HTML 引擎初始化完成")

    async def cleanup(self) -> None:
        """清理引擎资源"""
        await super().cleanup()
        logger.info("HTML 引擎已清理")

    async def process(self, input_path: Path, /, **kwargs: Any) -> list[str]:
        """处理 HTML 文件

        Args:
            input_path: HTML 文件路径
            **kwargs: 可选参数

        Returns:
            元素文本列表
        """
        preserve_structure = kwargs.get("preserve_structure", True)

        result = self._extractor.extract(
            input_path,
            preserve_structure=preserve_structure,
        )

        return [elem.text for elem in result.elements if elem.text.strip()]

    def get_elements(self, html_path: Path, **kwargs) -> list[HTMLElement]:
        """获取元素列表"""
        result = self._extractor.extract(html_path, preserve_structure=True)
        return result.elements

    def get_full_text(self, html_path: Path, **kwargs) -> str:
        """获取完整文本"""
        result = self._extractor.extract(html_path, preserve_structure=False)
        return result.full_text

    def get_title(self, html_path: Path) -> str:
        """获取页面标题"""
        result = self._extractor.extract(html_path)
        return result.title
