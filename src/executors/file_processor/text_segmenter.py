"""文本分段器"""

from typing import Any


class TextSegmenter:
    """文本分段器"""

    DEFAULT_PUNCTUATION = ["。", "！", "？", "；", "：", "，", "、", "……", "——", ".", "!", "?", ";", ",", "\n"]

    def __init__(self, max_length: int = 2500, min_length: int = 100, punctuations: list[str] | None = None):
        self.max_length = max_length
        self.min_length = min_length
        self.punctuations = punctuations or self.DEFAULT_PUNCTUATION
        self._punctuation_set = set(self.punctuations)

    def split(self, text: str) -> list[str]:
        if not text or len(text) <= self.max_length:
            return [text] if text else []

        segments = []
        start = 0

        while start < len(text):
            end = min(start + self.max_length, len(text))

            if end < len(text):
                split_point = self._find_best_split_point(text, start, end)
                if split_point > start:
                    end = split_point

            segment = text[start:end].strip()
            if segment:
                segments.append(segment)

            start = end

        segments = self._merge_short_segments(segments)

        return segments

    def _find_best_split_point(self, text: str, start: int, end: int) -> int:
        search_range = max(end - self.min_length, start + self.min_length)

        best_split = end

        for i in range(end - 1, search_range - 1, -1):
            if text[i] in self._punctuation_set and i > start + self.min_length:
                j = i + 1
                while j < end and text[j] in " \t\n":
                    j += 1
                best_split = j if j > i + 1 else i + 1
                break

        return best_split

    def _merge_short_segments(self, segments: list[str]) -> list[str]:
        if len(segments) <= 1:
            return segments

        merged = []
        current = ""

        for segment in segments:
            if len(current) + len(segment) <= self.max_length:
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

    def split_with_markers(self, text: str) -> list[tuple[int, str]]:
        segments = self.split(text)
        return list(enumerate(segments, 1))

    def get_segment_count(self, text: str) -> int:
        if not text or len(text) <= self.max_length:
            return 1
        return (len(text) + self.max_length - 1) // self.max_length

    def get_stats(self) -> dict[str, Any]:
        return {
            "max_length": self.max_length,
            "min_length": self.min_length,
            "punctuation_marks": len(self.punctuations),
        }


__all__ = [
    "TextSegmenter",
]
