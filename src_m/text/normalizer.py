"""Text normalization module.

Provides text formatting and standardization utilities
for preprocessing and cleaning input text.
"""

import re
from typing import Optional

# Precompiled regex patterns for performance
_SSML_XML_PATTERN = re.compile(
    r'<(?:speak|voice|prosody|break|emphasis|say-as|phonetic|audio|p|s|sub|mark|bookmark|lang|xml:[^>]*)[^>]*>',
    re.IGNORECASE,
)

_SELF_CLOSING_TAG_PATTERN = re.compile(r'<[^>]+/>')
_GENERIC_TAG_PATTERN = re.compile(r'<[^>]*>')
_WHITESPACE_PATTERN = re.compile(r'[ \t]+')


class TextNormalizer:
    """Normalizes text through configurable transformation pipeline.

    Supports whitespace normalization, linebreak standardization,
    punctuation conversion, SSML/XML tag removal, and empty line handling.
    """

    def __init__(
        self,
        enable_whitespace_normalization: bool = True,
        enable_linebreak_normalization: bool = True,
        enable_punctuation_normalization: bool = True,
        enable_trim_whitespace: bool = True,
        enable_empty_line_normalization: bool = True,
        enable_ssml_xml_cleaning: bool = False,
        max_consecutive_empty_lines: int = 2,
    ) -> None:
        self.enable_whitespace_normalization = enable_whitespace_normalization
        self.enable_linebreak_normalization = enable_linebreak_normalization
        self.enable_punctuation_normalization = enable_punctuation_normalization
        self.enable_trim_whitespace = enable_trim_whitespace
        self.enable_empty_line_normalization = enable_empty_line_normalization
        self.enable_ssml_xml_cleaning = enable_ssml_xml_cleaning
        self.max_consecutive_empty_lines = max_consecutive_empty_lines

    _FULL_WIDTH_TO_HALF_WIDTH = str.maketrans({
        '\uff0c': ',', '\u3002': '.', '\uff01': '!', '\uff1f': '?',
        '\uff1b': ';', '\uff1a': ':', '\uff08': '(', '\uff09': ')',
        '\u3010': '[', '\u3011': ']', '\u300c': '\u201c', '\u300d': '\u201d',
        '\u300e': '\u2018', '\u300f': '\u2019', '\u3001': ',', '\u2026': '...',
        '\u2014': '\u2014', '\u3008': '<', '\u3009': '>',
        '\u3000': ' ',
    })

    def normalize(self, text: str) -> str:
        """Apply all enabled normalization rules sequentially.

        Args:
            text: Input text to normalize.

        Returns:
            Normalized text with all enabled transformations applied.
        """
        if not text:
            return text

        result = text

        if self.enable_ssml_xml_cleaning:
            result = self.clean_ssml_xml(result)

        if self.enable_linebreak_normalization:
            result = self.normalize_linebreaks(result)

        if self.enable_punctuation_normalization:
            result = self.normalize_punctuations(result)

        if self.enable_whitespace_normalization:
            result = self.normalize_whitespace(result)

        if self.enable_trim_whitespace:
            result = self.trim_whitespace(result)

        if self.enable_empty_line_normalization:
            result = self.normalize_empty_lines(result)

        return result

    @staticmethod
    def normalize_linebreaks(text: str) -> str:
        """Convert all linebreak variants to Unix-style LF.

        Args:
            text: Text with mixed linebreak styles.

        Returns:
            Text with all linebreaks converted to '\n'.
        """
        return text.replace('\r\n', '\n').replace('\r', '\n')

    def normalize_punctuations(self, text: str) -> str:
        """Convert full-width punctuation marks to half-width equivalents.

        Args:
            text: Text containing full-width punctuation.

        Returns:
            Text with full-width punctuation converted to half-width.
        """
        return text.translate(self._FULL_WIDTH_TO_HALF_WIDTH)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse multiple consecutive spaces/tabs into single spaces.

        Args:
            text: Text with potentially excessive whitespace.

        Returns:
            Text with normalized whitespace.
        """
        return _WHITESPACE_PATTERN.sub(' ', text)

    @staticmethod
    def trim_whitespace(text: str) -> str:
        """Strip leading and trailing whitespace from each line.

        Args:
            text: Multi-line text to trim.

        Returns:
            Text with each line stripped of surrounding whitespace.
        """
        return '\n'.join(line.strip() for line in text.split('\n'))

    def normalize_empty_lines(self, text: str) -> str:
        """Reduce consecutive empty lines to the configured maximum.

        Args:
            text: Text potentially containing excessive empty lines.

        Returns:
            Text with consecutive empty lines limited to max_consecutive_empty_lines.
        """
        lines = text.split('\n')
        result = []
        empty_count = 0

        for line in lines:
            if line:
                empty_count = 0
                result.append(line)
            elif empty_count < self.max_consecutive_empty_lines:
                empty_count += 1
                result.append(line)

        return '\n'.join(result)

    @staticmethod
    def clean_ssml_xml(text: str) -> str:
        """Remove SSML/XML tags from text, preserving plain content.

        Supported tags include: speak, voice, prosody, break, emphasis,
        say-as, phonetic, audio, p, s, sub, mark, bookmark, lang, and
        xml:* namespace tags.

        Args:
            text: Text containing SSML/XML markup.

        Returns:
            Plain text with all SSML/XML tags removed.
        """
        cleaned = _SSML_XML_PATTERN.sub('', text)
        cleaned = _SELF_CLOSING_TAG_PATTERN.sub('', cleaned)
        return _GENERIC_TAG_PATTERN.sub('', cleaned)

    # Configuration setters for runtime toggling

    def set_whitespace_normalization(self, enabled: bool) -> None:
        """Toggle whitespace normalization."""
        self.enable_whitespace_normalization = enabled

    def set_linebreak_normalization(self, enabled: bool) -> None:
        """Toggle linebreak normalization."""
        self.enable_linebreak_normalization = enabled

    def set_punctuation_normalization(self, enabled: bool) -> None:
        """Toggle punctuation normalization."""
        self.enable_punctuation_normalization = enabled

    def set_trim_whitespace(self, enabled: bool) -> None:
        """Toggle per-line whitespace trimming."""
        self.enable_trim_whitespace = enabled

    def set_empty_line_normalization(
        self, enabled: bool, max_lines: Optional[int] = None
    ) -> None:
        """Toggle empty line normalization and optionally set the max limit."""
        self.enable_empty_line_normalization = enabled
        if max_lines is not None:
            self.max_consecutive_empty_lines = max_lines

    def set_ssml_xml_cleaning(self, enabled: bool) -> None:
        """Toggle SSML/XML tag removal."""
        self.enable_ssml_xml_cleaning = enabled
