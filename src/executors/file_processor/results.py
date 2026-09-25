"""文件处理结果数据类"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EPUBSplitResult:
    """EPUB 分割结果"""

    success: bool = False
    epub_title: str = ""
    author: str = ""
    chapters_count: int = 0
    total_chars: int = 0
    total_words: int = 0
    duration: float = 0.0
    errors: list[str] = field(default_factory=list)
    output_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "epub_title": self.epub_title,
            "author": self.author,
            "chapters_count": self.chapters_count,
            "total_chars": self.total_chars,
            "total_words": self.total_words,
            "duration": self.duration,
            "errors": self.errors,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }


@dataclass
class FileProcessingResult:
    """文件处理结果"""

    success: bool
    files_processed: int = 0
    files_failed: int = 0
    total_size: int = 0
    duration: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "total_size": self.total_size,
            "duration": self.duration,
            "errors": self.errors,
        }


__all__ = [
    "EPUBSplitResult",
    "FileProcessingResult",
]
