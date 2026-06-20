"""分章引擎

封装章节检测和分割逻辑，支持多种语言模式。

Phase 1 升级:
* 支持 YAML 规则文件加载
* 标题标准化 (去 BOM/全角空格/零宽字符)
* 多语言标题正则 (中文/英文/日文常见模式)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from re import Pattern
from typing import Any

from src.config import PPC10Config
from src.core import BaseEngine
from src.core.exceptions import ErrorCodes
from src.reliability import ExecutionMetrics, ExecutionResult
from src.utils.core import detect_encoding, sanitize_filename

logger = logging.getLogger(__name__)

TITLE_SEPARATOR_MULTIPLIER = 2
DEFAULT_PREFIX = "chapter"

# 需要在标题标准化过程中过滤/替换的不可见字符
_INVISIBLE_CHARS = {
    "\ufeff",  # BOM
    "\u200b",  # 零宽空格
    "\u200c",  # ZWNJ
    "\u200d",  # ZWJ
    "\u2060",  # 字符连接器
    "\u180e",  # 蒙古文元音分隔符
    "\u00a0",  # 不间断空格
    "\u3000",  # 中文全角空格
    "\u2002",  # en 空格
    "\u2003",  # em 空格
    "\u2009",  # thin 空格
}

# 内置的日文/韩文章节模式
_BUILTIN_MULTILINGUAL_PATTERNS: dict[str, list[str]] = {
    "chinese_novel": [
        r"^(引子|序章|序|前言|引子|后记|附录)(：|:)?(.*)$",
        r"^第[一二两三四五六七八九十百千万亿\d零〇]+[章篇章节回集部卷]\s*(.*)$",
        r"^\d+[\.、\s][^\d]",
    ],
    "english_novel": [
        r"^Chapter\s+\d+\s*[:：\-]?\s*(.*)$",
        r"^Part\s+\d+\s*[:：\-]?\s*(.*)$",
        r"^(Prologue|Epilogue|Interlude|Epilogue|Appendix)\s*[:：\-]?\s*(.*)$",
        r"^\d+[\.\s]+[A-Z][^\d]*$",
    ],
    "japanese_novel": [
        r"^第[一二三四五六七八九十百千万\d]+話\s*[:：\-]?\s*(.*)$",
        r"^第[一二三四五六七八九十百千万\d]+章\s*[:：\-]?\s*(.*)$",
        r"^(プロローグ|エピローグ|序章|終章|前編|後編)",
        r"^\d+[．\.\s]\S+",
    ],
    "korean_novel": [
        r"^\d+화\s*[:：\-]?\s*(.*)$",
        r"^\d+장\s*[:：\-]?\s*(.*)$",
        r"^(프롤로그|에필로그|서장|종장|1부|2부)",
    ],
}


@dataclass
class ChapterInfo:
    """章节信息"""

    index: int
    title: str
    start_line: int
    end_line: int
    content: str


@dataclass
class ChapterRuleSet:
    """从 YAML 文件加载的章节规则集合。"""

    name: str
    patterns: list[str] = field(default_factory=list)
    description: str = ""
    options: dict[str, Any] = field(default_factory=dict)

    def to_pattern_list(self) -> list[str]:
        return list(self.patterns)


def _strip_invisible(text: str) -> str:
    """去除不可见字符: BOM/全角空格/零宽字符等。"""
    if not text:
        return text
    return "".join(ch for ch in text if ch not in _INVISIBLE_CHARS)


def normalize_chapter_title(
    title: str,
    *,
    strip_invisible: bool = True,
    collapse_whitespace: bool = True,
    strip: bool = True,
) -> str:
    """标准化章节标题。

    步骤:
    1. 去除 BOM/全角空格/零宽字符
    2. 折叠连续空白
    3. 去除首尾空白
    """
    if not title:
        return title
    if strip_invisible:
        title = _strip_invisible(title)
    if collapse_whitespace:
        title = re.sub(r"[ \t\f\v\u2002-\u200a\u2028\u2029]+", " ", title)
    if strip:
        title = title.strip()
    return title


class ChapterEngine(BaseEngine[str, list[Path]]):
    """分章引擎

    根据预设的正则表达式模式检测并分割章节。
    支持中文、英文、日文、韩文和自定义模式。
    支持 YAML 规则文件加载与标题标准化。
    """

    CHAPTER_PATTERNS: dict[str, list[str]] = {
        "chinese_novel": [
            r"^(引子|序章|前言|后记|附录)(：|:)?(.*)$",
            r"^第[一二两三四五六七八九十百千万亿\d零〇]+[章篇章节回集部卷]\s*(.*)$",
        ],
        "english_novel": [
            r"^Chapter\s+\d+(.*)$",
            r"^Part\s+\d+(.*)$",
            r"^(Prologue|Epilogue)\s*$",
        ],
        "default": [
            r"^第[一二两三四五六七八九十百千万亿\d零〇]+[章篇章节回集部卷]\s*(.*)$",
            r"^\d+[\.\s]+(.*)$",
        ],
    }

    def __init__(self, config: PPC10Config) -> None:
        super().__init__()
        self.config = config
        self.split_config = config.split
        self._patterns: dict[str, list[Pattern[str]]] = self._compile_patterns()
        self._extra_rule_sets: dict[str, ChapterRuleSet] = {}

    # ------------------------------------------------------------------
    # 规则加载
    # ------------------------------------------------------------------

    def _compile_patterns(self) -> dict[str, list[Pattern[str]]]:
        """预编译章节正则表达式。

        合并内置模式 + 多语言内置模式, 以便按 preset 选择时也能匹配其他语言。
        """
        compiled: dict[str, list[Pattern[str]]] = {}
        all_patterns: dict[str, list[str]] = {}
        for k, v in self.CHAPTER_PATTERNS.items():
            all_patterns.setdefault(k, []).extend(v)
        for k, v in _BUILTIN_MULTILINGUAL_PATTERNS.items():
            all_patterns.setdefault(k, []).extend(v)
        for k, v in all_patterns.items():
            compiled[k] = [re.compile(p, re.IGNORECASE) for p in v]
        return compiled

    def load_rules_from_file(self, file_path: str | Path) -> ChapterRuleSet:
        """从 YAML 文件加载规则集。"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"规则文件不存在: {path}")
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError("加载 YAML 规则需要安装 pyyaml") from e
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"YAML 规则文件格式错误, 期望字典: {path}")
        name = str(raw.get("name") or path.stem)
        patterns = raw.get("patterns") or []
        if isinstance(patterns, str):
            patterns = [patterns]
        rule_set = ChapterRuleSet(
            name=name,
            patterns=[str(p) for p in patterns],
            description=str(raw.get("description", "")),
            options=raw.get("options") or {},
        )
        self._extra_rule_sets[name] = rule_set
        compiled = [re.compile(p, re.IGNORECASE) for p in rule_set.patterns]
        self._patterns[name] = compiled
        logger.info(f"已加载章节规则集: {name} ({len(rule_set.patterns)} 条)")
        return rule_set

    def load_rules_from_dict(self, name: str, patterns: list[str]) -> ChapterRuleSet:
        """从字典中加载规则集 (用于测试/动态配置)。"""
        rule_set = ChapterRuleSet(name=name, patterns=list(patterns))
        self._extra_rule_sets[name] = rule_set
        self._patterns[name] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return rule_set

    def list_rule_sets(self) -> list[str]:
        """列出所有已加载的规则集名称。"""
        names = list(self._patterns.keys())
        return names

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """初始化引擎"""
        await super().initialize()
        logger.info("分章引擎初始化完成")

    async def cleanup(self) -> None:
        """清理引擎"""
        await super().cleanup()
        logger.info("分章引擎已清理")

    async def process(
        self,
        input_data: str,
        /,
        **kwargs: Any,
    ) -> list[Path]:
        """处理输入数据并返回结果（统一接口）"""
        output_dir = kwargs.get("output_dir")
        if not output_dir:
            raise ValueError("output_dir is required")

        filename_prefix = kwargs.get("filename_prefix", DEFAULT_PREFIX)
        result = await self.split(input_data, output_dir, filename_prefix)

        if not result.success:
            raise RuntimeError(result.error or "Split failed")
        return result.data or []

    async def split(
        self,
        content: str,
        output_dir: Path,
        filename_prefix: str = DEFAULT_PREFIX,
    ) -> ExecutionResult[list[Path]]:
        """分割内容为章节文件"""
        start_time = time.perf_counter()

        try:
            chapters = self._detect_chapters(content)
            if not chapters:
                return ExecutionResult.fail(error="未检测到章节", error_code=ErrorCodes.NO_CHAPTERS.value)

            output_dir.mkdir(parents=True, exist_ok=True)
            output_files = [
                self._write_chapter_to_file(
                    self._generate_filename(output_dir, chapter.index, chapter.title, filename_prefix),
                    chapter,
                )
                for chapter in chapters
            ]

            metrics = ExecutionMetrics(
                duration=time.perf_counter() - start_time,
                bytes_processed=len(output_files),
            )
            return ExecutionResult.ok(output_files, metrics)

        except Exception as e:
            logger.error(f"分章失败: {e}")
            return ExecutionResult.fail(error=str(e), error_code=ErrorCodes.CHAPTER_SPLIT_FAILED.value)

    # ------------------------------------------------------------------
    # 章节检测
    # ------------------------------------------------------------------

    def _detect_chapters(self, content: str) -> list[ChapterInfo]:
        """检测并提取章节"""
        lines = content.splitlines(keepends=True)
        preset = self.split_config.preset
        patterns = self._patterns.get(preset, self._patterns.get("default", []))

        chapters: list[ChapterInfo] = []
        current_chapter: ChapterInfo | None = None
        chapter_index = 0

        for i, line in enumerate(lines):
            raw_line = line
            normalized = normalize_chapter_title(raw_line)
            if not normalized:
                continue

            if self._match_any_pattern(normalized, patterns):
                if current_chapter:
                    current_chapter.end_line = i
                    self._finalize_chapter(current_chapter, lines)
                    if self._is_valid_chapter(current_chapter):
                        chapters.append(current_chapter)

                chapter_index += 1
                current_chapter = ChapterInfo(
                    index=chapter_index,
                    title=normalized,
                    start_line=i,
                    end_line=len(lines),
                    content="",
                )

        # Finalize the last chapter
        if current_chapter:
            self._finalize_chapter(current_chapter, lines)
            if self._is_valid_chapter(current_chapter):
                chapters.append(current_chapter)

        if not chapters and lines:
            normalized_title = normalize_chapter_title("全文")
            chapters.append(
                ChapterInfo(
                    index=1,
                    title=normalized_title,
                    start_line=0,
                    end_line=len(lines),
                    content="".join(lines).strip(),
                )
            )

        return chapters

    @staticmethod
    def _match_any_pattern(text: str, patterns: list[Pattern[str]]) -> bool:
        """检查文本是否匹配任意模式"""
        return any(p.match(text) for p in patterns)

    def _finalize_chapter(self, chapter: ChapterInfo, lines: list[str]) -> None:
        """完成章节内容提取"""
        chapter.end_line = chapter.end_line if chapter.end_line < len(lines) else len(lines)
        chapter.content = "".join(lines[chapter.start_line : chapter.end_line]).strip()

    def _is_valid_chapter(self, chapter: ChapterInfo) -> bool:
        """验证章节是否满足最小长度要求"""
        return len(chapter.content) >= self.split_config.min_chapter_length

    def _generate_filename(
        self,
        output_dir: Path,
        index: int,
        title: str,
        prefix: str,
    ) -> Path:
        """生成输出文件名"""
        safe_title = self._sanitize_filename(title) or f"{prefix}_{index:03d}"
        return output_dir / f"{index:03d}_{safe_title}.txt"

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名中的非法字符"""
        return sanitize_filename(filename, max_length=self.split_config.max_filename_length)

    def _write_chapter_to_file(self, output_file: Path, chapter: ChapterInfo) -> Path:
        """写入章节到文件"""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            f.write(f"{chapter.title}\n")
            if self.split_config.add_title_separator:
                f.write("=" * len(chapter.title) * TITLE_SEPARATOR_MULTIPLIER + "\n\n")
            else:
                f.write("\n")
            f.write(chapter.content)
            f.write("\n")

        return output_file

    async def split_file(
        self,
        input_file: Path,
        output_dir: Path,
    ) -> ExecutionResult[list[Path]]:
        """分割文件"""
        if not input_file.exists():
            return ExecutionResult.fail(error=f"文件不存在: {input_file}", error_code=ErrorCodes.FILE_NOT_FOUND.value)

        encoding = self._detect_encoding(input_file)
        content = input_file.read_text(encoding=encoding)
        return await self.split(content, output_dir, input_file.stem)

    def _detect_encoding(self, file_path: Path) -> str:
        """检测文件编码"""
        encoding = detect_encoding(file_path, encodings=self.split_config.encoding_fallback, detect_buffer=1024)
        return encoding if encoding else "utf-8"

    def get_stats(self) -> dict[str, Any]:
        """获取引擎统计信息"""
        return {
            "preset": self.split_config.preset,
            "min_chapter_length": self.split_config.min_chapter_length,
            "encoding_fallback": self.split_config.encoding_fallback,
            "rule_sets": self.list_rule_sets(),
            "extra_rule_sets": list(self._extra_rule_sets.keys()),
        }


__all__ = [
    "ChapterEngine",
    "ChapterInfo",
    "ChapterRuleSet",
    "normalize_chapter_title",
]
