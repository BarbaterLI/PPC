"""分章引擎

封装章节检测和分割逻辑，支持多种语言模式。

切分算法统一委托给 :class:`src.executors.chapter_splitter.ChapterSplitter`
（与 ``SplitterExecutor`` 共用同一实现，engine profile 提供含多语言扩展的
模式集与标题标准化钩子）；本模块保留 YAML 规则集加载、标题标准化与
``BaseEngine`` 生命周期接口。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import PPCConfig
from src.core import BaseEngine
from src.core.exceptions import ErrorCodes
from src.executors.chapter_splitter import ChapterSplitter
from src.executors.chapter_types import ChapterInfo
from src.executors.splitter_strategies import ENGINE_PROFILE, ChapterWriter, PatternRegistry
from src.reliability import ExecutionMetrics, ExecutionResult
from src.utils.files import read_text_auto

logger = logging.getLogger(__name__)

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

    def __init__(self, config: PPCConfig) -> None:
        super().__init__()
        self.config = config
        self.split_config = config.split
        self._registry = PatternRegistry(ENGINE_PROFILE)
        self._splitter = ChapterSplitter(
            config, registry=self._registry, title_normalizer=normalize_chapter_title
        )
        self._writer = ChapterWriter(config)
        self._extra_rule_sets: dict[str, ChapterRuleSet] = {}

    # ------------------------------------------------------------------
    # 规则加载
    # ------------------------------------------------------------------

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
        self._registry.register(name, rule_set.patterns)
        logger.info(f"已加载章节规则集: {name} ({len(rule_set.patterns)} 条)")
        return rule_set

    def load_rules_from_dict(self, name: str, patterns: list[str]) -> ChapterRuleSet:
        """从字典中加载规则集 (用于测试/动态配置)。"""
        rule_set = ChapterRuleSet(name=name, patterns=list(patterns))
        self._extra_rule_sets[name] = rule_set
        self._registry.register(name, rule_set.patterns)
        return rule_set

    def list_rule_sets(self) -> list[str]:
        """列出所有已加载的规则集名称。"""
        return self._registry.list_names()

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
            chapters = self._splitter.split_content(content, hierarchical=False)
            if not chapters:
                return ExecutionResult.fail(error="未检测到章节", error_code=ErrorCodes.NO_CHAPTERS.value)

            output_dir.mkdir(parents=True, exist_ok=True)
            output_files = [
                self._writer.write_chapter(
                    self._writer.generate_output_name(output_dir, chapter.index, chapter.title, filename_prefix),
                    chapter,
                )
                for chapter in chapters
            ]

            metrics = ExecutionMetrics(
                duration=time.perf_counter() - start_time,
                bytes_processed=sum(
                    p.stat().st_size if p.exists() else 0 for p in output_files
                ),
            )
            return ExecutionResult.ok(output_files, metrics)

        except Exception as e:
            logger.error(f"分章失败: {e}")
            return ExecutionResult.fail(error=str(e), error_code=ErrorCodes.CHAPTER_SPLIT_FAILED.value)

    async def split_file(
        self,
        input_file: Path,
        output_dir: Path,
    ) -> ExecutionResult[list[Path]]:
        """分割文件"""
        if not input_file.exists():
            return ExecutionResult.fail(error=f"文件不存在: {input_file}", error_code=ErrorCodes.FILE_NOT_FOUND.value)

        content, _enc = await asyncio.to_thread(
            read_text_auto, input_file, self.split_config.encoding_fallback
        )
        return await self.split(content, output_dir, input_file.stem)

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
