"""PDF 文本提取引擎

支持从 PDF 文件提取文本内容，支持加密 PDF 和按页面分割。
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.base import BaseEngine

logger = logging.getLogger(__name__)

PDFPLUMBER_AVAILABLE = False
PYPDF2_AVAILABLE = False

try:
    import pdfplumber  # noqa: F401

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    pass

try:
    from PyPDF2 import PdfReader  # noqa: F401

    PYPDF2_AVAILABLE = True
except ImportError:
    pass


@dataclass
class PDFPage:
    """PDF 页面"""

    page_number: int
    text: str
    is_encrypted: bool = False


@dataclass
class PDFExtractResult:
    """PDF 提取结果"""

    pages: list[PDFPage]
    total_pages: int
    is_encrypted: bool
    metadata: dict


class PDFTextExtractor:
    """PDF 文本提取器"""

    def __init__(self) -> None:
        self._use_pdfplumber: bool = PDFPLUMBER_AVAILABLE
        self._use_pypdf2: bool = PYPDF2_AVAILABLE

    def extract(
        self,
        pdf_path: Path,
        password: str | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> PDFExtractResult:
        """提取 PDF 文本

        Args:
            pdf_path: PDF 文件路径
            password: PDF 密码（如果需要）
            start_page: 起始页（从 1 开始）
            end_page: 结束页（包含）

        Returns:
            PDFExtractResult 对象
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        if self._use_pdfplumber:
            return self._extract_with_pdfplumber(pdf_path, password, start_page, end_page)
        elif self._use_pypdf2:
            return self._extract_with_pypdf2(pdf_path, password, start_page, end_page)
        else:
            raise ImportError(
                "无法导入 PDF 处理库。请安装 pdfplumber 或 PyPDF2：\npip install pdfplumber\n或\npip install PyPDF2"
            )

    def _extract_with_pdfplumber(
        self,
        pdf_path: Path,
        password: str | None,
        start_page: int | None,
        end_page: int | None,
    ) -> PDFExtractResult:
        """使用 pdfplumber 提取"""
        import pdfplumber

        pages = []
        metadata = {}
        is_encrypted = False

        with pdfplumber.open(pdf_path, password=password) as pdf:
            if start_page is None:
                start_page = 1
            if end_page is None:
                end_page = len(pdf.pages)

            for i, page in enumerate(pdf.pages[start_page - 1 : end_page], start=start_page):
                text = page.extract_text() or ""
                pages.append(PDFPage(page_number=i, text=text))

            if pdf.metadata:
                metadata = pdf.metadata

        return PDFExtractResult(
            pages=pages,
            total_pages=len(pages),
            is_encrypted=is_encrypted,
            metadata=metadata,
        )

    def _extract_with_pypdf2(
        self,
        pdf_path: Path,
        password: str | None,
        start_page: int | None,
        end_page: int | None,
    ) -> PDFExtractResult:
        """使用 PyPDF2 提取"""
        from PyPDF2 import PdfReader

        reader = PdfReader(str(pdf_path))
        is_encrypted = reader.is_encrypted

        if is_encrypted:
            if password:
                reader.decrypt(password)
            else:
                raise ValueError("PDF 文件已加密，需要提供密码")

        pages = []
        total = len(reader.pages)

        start = (start_page - 1) if start_page else 0
        end = end_page if end_page else total

        for i in range(start, min(end, total)):
            page = reader.pages[i]
            text = page.extract_text() or ""
            pages.append(PDFPage(page_number=i + 1, text=text))

        metadata = {
            "/Title": reader.metadata.get("/Title") if reader.metadata else None,
            "/Author": reader.metadata.get("/Author") if reader.metadata else None,
        }

        return PDFExtractResult(
            pages=pages,
            total_pages=len(pages),
            is_encrypted=is_encrypted,
            metadata=metadata,
        )

    def is_available(self) -> bool:
        """检查是否可用"""
        return self._use_pdfplumber or self._use_pypdf2


class PDFEngine(BaseEngine[Path, list[str]]):
    """PDF 处理引擎

    继承 BaseEngine，将 PDF 文件转换为文本内容。
    """

    def __init__(self, config: Any = None) -> None:
        super().__init__()
        self.config = config
        self._extractor = PDFTextExtractor()

    async def initialize(self) -> None:
        """初始化引擎"""
        await super().initialize()
        if not self._extractor.is_available():
            logger.warning("PDF 处理库未安装，PDF 功能将不可用。\n安装命令: pip install pdfplumber")
        logger.info("PDF 引擎初始化完成")

    async def cleanup(self) -> None:
        """清理引擎资源"""
        await super().cleanup()
        logger.info("PDF 引擎已清理")

    async def process(self, input_path: Path, /, **kwargs: Any) -> list[str]:
        """处理 PDF 文件

        Args:
            input_path: PDF 文件路径
            **kwargs: 可选参数
                - password: PDF 密码
                - start_page: 起始页
                - end_page: 结束页

        Returns:
            页面文本列表
        """
        password = kwargs.get("password")
        start_page = kwargs.get("start_page")
        end_page = kwargs.get("end_page")

        result = self._extractor.extract(
            input_path,
            password=password,
            start_page=start_page,
            end_page=end_page,
        )

        texts = [page.text for page in result.pages]
        return texts

    def get_text_by_pages(self, pdf_path: Path, **kwargs) -> list[tuple[int, str]]:
        """获取按页面分割的文本

        Returns:
            [(页号, 文本), ...] 列表
        """
        password = kwargs.get("password")
        start_page = kwargs.get("start_page")
        end_page = kwargs.get("end_page")

        result = self._extractor.extract(
            pdf_path,
            password=password,
            start_page=start_page,
            end_page=end_page,
        )

        return [(page.page_number, page.text) for page in result.pages]

    def get_full_text(self, pdf_path: Path, **kwargs) -> str:
        """获取完整文本

        Returns:
            合并后的完整文本
        """
        password = kwargs.get("password")
        start_page = kwargs.get("start_page")
        end_page = kwargs.get("end_page")

        result = self._extractor.extract(
            pdf_path,
            password=password,
            start_page=start_page,
            end_page=end_page,
        )

        return "\n\n".join(page.text for page in result.pages)
