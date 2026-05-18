"""Text segmentation module.

Splits long text into smaller segments suitable for TTS processing,
with intelligent boundary detection at punctuation marks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from src_m.config import TTSConfig

logger = logging.getLogger(__name__)


class TextSegmenter:
    """Segments text into chunks respecting length limits and natural boundaries.

    Uses punctuation-based heuristics to find semantically appropriate
    split points rather than cutting at arbitrary positions.
    """

    DEFAULT_PUNCTUATIONS: List[str] = [
        '。', '！', '？', '；', '，', '、', '……', '——',
        '.', '!', '?', ';', ',', '\n',
    ]
    DEFAULT_MIN_LENGTH = 100
    DEFAULT_SEPARATOR = '\n'

    def __init__(
        self,
        punctuations: Optional[List[str]] = None,
        min_segment_length: int = DEFAULT_MIN_LENGTH,
        separator: str = DEFAULT_SEPARATOR,
    ) -> None:
        self._punctuations = punctuations or self.DEFAULT_PUNCTUATIONS
        self._min_length = min_segment_length
        self._separator = separator

    def split(self, text: str, max_length: int) -> List[str]:
        """Split text into segments not exceeding max_length.

        Args:
            text: Input text to segment.
            max_length: Maximum character count per segment.

        Returns:
            List of text segments, with short adjacent segments merged.
        """
        stripped = text.strip() if text else ''
        if not stripped:
            return []

        if len(stripped) <= max_length:
            return [stripped]

        return self._merge_short_segments(
            self._chunk_text(stripped, max_length),
            max_length,
        )

    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """Internal chunking logic without merging."""
        segments: List[str] = []
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

        return segments

    def _find_split_point(self, text: str, start: int, end: int) -> int:
        """Locate the best punctuation-based split position within range.

        Searches backward from end toward start to find the last punctuation
        mark that yields a segment at least min_length characters long.

        Args:
            text: The full input text.
            start: Start index of the current chunk.
            end: Proposed end index of the current chunk.

        Returns:
            Index after the punctuation mark, or end if no good split found.
        """
        if end - start < 2 * self._min_length:
            for i in range(end - 1, start, -1):
                if text[i] in self._punctuations:
                    return i + 1
            return end

        search_start = max(end - self._min_length, start + self._min_length)

        for i in range(end - 1, search_start - 1, -1):
            if text[i] in self._punctuations:
                return i + 1

        return end

    def _merge_short_segments(self, segments: List[str], max_length: int) -> List[str]:
        """Merge adjacent short segments to avoid fragmented output.

        Segments shorter than min_length are combined with neighbors
        as long as the merged result stays within max_length.

        Args:
            segments: List of initially split segments.
            max_length: Maximum allowed length for merged segments.

        Returns:
            Segments with short ones merged where possible.
        """
        if len(segments) <= 1:
            return segments

        merged: List[str] = []
        current: Optional[str] = None

        for segment in segments:
            should_merge = (
                current is not None
                and len(current) + len(segment) + 1 <= max_length
                and len(segment) < self._min_length
            )

            if should_merge:
                current += self._separator + segment
            else:
                if current is not None:
                    merged.append(current)
                current = segment

        if current is not None:
            merged.append(current)

        return merged

    # Configuration setters

    def set_punctuations(self, punctuations: List[str]) -> None:
        """Replace the set of punctuation marks used for split detection."""
        self._punctuations = punctuations

    def set_min_length(self, min_length: int) -> None:
        """Set the minimum segment length threshold."""
        self._min_length = min_length

    @classmethod
    def from_config(cls, config: TTSConfig) -> TextSegmenter:
        """Construct a TextSegmenter from a TTSConfig object.

        Args:
            config: TTSConfig instance with segmenter-related settings.

        Returns:
            Configured TextSegmenter instance.
        """
        return cls(
            punctuations=getattr(config, 'punctuations', cls.DEFAULT_PUNCTUATIONS),
            min_segment_length=getattr(config, 'min_segment_length', cls.DEFAULT_MIN_LENGTH),
        )
