"""文本分段器模块
负责将长文本分割成适合TTS处理的段落
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class TextSegmenter:
    """文本分段器"""

    DEFAULT_PUNCTUATIONS = ['。', '！', '？', '；', '，', '、', '……', '——', '.', '!', '?', ';', ',', '\n']
    DEFAULT_MIN_LENGTH = 100

    def __init__(
        self,
        punctuations: Optional[List[str]] = None,
        min_segment_length: int = DEFAULT_MIN_LENGTH
    ):
        self._punctuations = punctuations or self.DEFAULT_PUNCTUATIONS
        self._min_length = min_segment_length

    def split(self, text: str, max_length: int) -> List[str]:
        """分割文本"""
        if not text or not text.strip():
            return []

        if len(text) <= max_length:
            return [text.strip()]

        segments = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + max_length, text_len)

            if end < text_len:
                split_point = self._find_split_point(text, start, end)
                if split_point > start:
                    end = split_point

            segment = text[start:end].strip()
            if segment:
                segments.append(segment)

            start = end

        return self._merge_short_segments(segments, max_length)

    def _find_split_point(self, text: str, start: int, end: int) -> int:
        """找到最佳分割点"""
        search_range = max(end - self._min_length, start + self._min_length)

        for i in range(end - 1, search_range - 1, -1):
            if text[i] in self._punctuations:
                if i > start + self._min_length:
                    return i + 1

        return end

    def _merge_short_segments(self, segments: List[str], max_length: int) -> List[str]:
        """合并短片段"""
        if len(segments) <= 1:
            return segments

        merged = []
        current = ""

        for segment in segments:
            if len(current) + len(segment) <= max_length and len(segment) < self._min_length:
                if current:
                    current += "\n"
                current += segment
            else:
                if current:
                    merged.append(current)
                current = segment

        if current:
            merged.append(current)

        return merged

    def set_punctuations(self, punctuations: List[str]) -> None:
        """设置分段标点符号"""
        self._punctuations = punctuations

    def set_min_length(self, min_length: int) -> None:
        """设置最小段落长度"""
        self._min_length = min_length
        
    @classmethod
    def from_config(cls, config) -> 'TextSegmenter':
        """从配置对象创建分段器
        
        Args:
            config: TTSConfig 对象
            
        Returns:
            TextSegmenter 实例
        """
        return cls(
            punctuations=getattr(config, 'punctuations', cls.DEFAULT_PUNCTUATIONS),
            min_segment_length=getattr(config, 'min_segment_length', cls.DEFAULT_MIN_LENGTH)
        )
