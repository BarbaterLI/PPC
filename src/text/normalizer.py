"""文本正则化模块
负责对文本进行格式化和标准化处理
"""

import re
from typing import Optional


class TextNormalizer:
    """文本正则化器"""

    def __init__(
        self,
        enable_whitespace_normalization: bool = True,
        enable_linebreak_normalization: bool = True,
        enable_punctuation_normalization: bool = True,
        enable_trim_whitespace: bool = True,
        enable_empty_line_normalization: bool = True,
        enable_ssml_xml_cleaning: bool = False,
        max_consecutive_empty_lines: int = 2
    ):
        self._enable_whitespace_normalization = enable_whitespace_normalization
        self._enable_linebreak_normalization = enable_linebreak_normalization
        self._enable_punctuation_normalization = enable_punctuation_normalization
        self._enable_trim_whitespace = enable_trim_whitespace
        self._enable_empty_line_normalization = enable_empty_line_normalization
        self._enable_ssml_xml_cleaning = enable_ssml_xml_cleaning
        self._max_consecutive_empty_lines = max_consecutive_empty_lines
        
        # SSML/XML 标签匹配正则表达式
        # 匹配常见的 SSML 标签和 XML 控制字符
        self._ssml_xml_pattern = re.compile(
            r'<(?:speak|voice|prosody|break|emphasis|say-as|phonetic|audio|p|s|sub|mark|bookmark|lang|xml:[^>]*)[^>]*>',
            re.IGNORECASE
        )
        
        self._full_width_to_half_width = str.maketrans({
            '，': ',', '。': '.', '！': '!', '？': '?', '；': ';', '：': ':',
            '（': '(', '）': ')', '【': '[', '】': ']', '「': '"', '」': '"',
            '『': "'", '』': "'", '、': ',', '…': '...', '—': '-', '〈': '<', '〉': '>'
        })

    def normalize(self, text: str) -> str:
        """一次性应用所有正则化规则"""
        if not text:
            return text

        result = text

        # 首先清洗 SSML/XML 控制字符（如果启用）
        if self._enable_ssml_xml_cleaning:
            result = self.clean_ssml_xml(result)

        if self._enable_linebreak_normalization:
            result = self.normalize_linebreaks(result)

        if self._enable_punctuation_normalization:
            result = self.normalize_punctuations(result)

        if self._enable_whitespace_normalization:
            result = self.normalize_whitespace(result)

        if self._enable_trim_whitespace:
            result = self.trim_whitespace(result)

        if self._enable_empty_line_normalization:
            result = self.normalize_empty_lines(result)

        return result

    def normalize_linebreaks(self, text: str) -> str:
        """统一换行符"""
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        return text

    def normalize_punctuations(self, text: str) -> str:
        """标准化标点符号"""
        return text.translate(self._full_width_to_half_width)

    def normalize_whitespace(self, text: str) -> str:
        """去除多余空白字符"""
        text = re.sub(r'[ \t]+', ' ', text)
        return text

    def trim_whitespace(self, text: str) -> str:
        """去除行首尾空白"""
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        return '\n'.join(lines)

    def normalize_empty_lines(self, text: str) -> str:
        """去除空行或过多连续空行"""
        lines = text.split('\n')
        result = []
        empty_line_count = 0

        for line in lines:
            if not line:
                empty_line_count += 1
                if empty_line_count <= self._max_consecutive_empty_lines:
                    result.append(line)
            else:
                empty_line_count = 0
                result.append(line)

        return '\n'.join(result)

    def clean_ssml_xml(self, text: str) -> str:
        """清洗 SSML/XML 控制字符
        
        移除常见的 SSML 标签和 XML 控制字符，保留纯文本内容。
        这对于处理包含语音合成标记的文本非常有用。
        
        支持的标签包括：
        - <speak>, </speak>
        - <voice>, </voice>
        - <prosody>, </prosody>
        - <break/>
        - <emphasis>, </emphasis>
        - <say-as>, </say-as>
        - <phonetic>, </phonetic>
        - <audio>, </audio>
        - <p>, </p>, <s>, </s>
        - <sub>, </sub>
        - <mark/>, <bookmark/>
        - <lang>, </lang>
        - 其他 xml:* 命名空间的标签
        
        Args:
            text: 包含 SSML/XML 标签的文本
            
        Returns:
            清洗后的纯文本
        """
        # 移除所有匹配的 SSML/XML 标签
        cleaned = self._ssml_xml_pattern.sub('', text)
        
        # 移除可能的自闭合标签（如 <break time="500ms"/>）
        cleaned = re.sub(r'<[^>]+/>', '', cleaned)
        
        # 移除残留的 HTML/XML 标签（通用匹配）
        cleaned = re.sub(r'<[^>]*>', '', cleaned)
        
        return cleaned

    def set_whitespace_normalization(self, enabled: bool) -> None:
        """设置是否启用空白字符规范化"""
        self._enable_whitespace_normalization = enabled

    def set_linebreak_normalization(self, enabled: bool) -> None:
        """设置是否启用换行符规范化"""
        self._enable_linebreak_normalization = enabled

    def set_punctuation_normalization(self, enabled: bool) -> None:
        """设置是否启用标点符号规范化"""
        self._enable_punctuation_normalization = enabled

    def set_trim_whitespace(self, enabled: bool) -> None:
        """设置是否启用行首尾空白去除"""
        self._enable_trim_whitespace = enabled

    def set_empty_line_normalization(self, enabled: bool, max_lines: Optional[int] = None) -> None:
        """设置是否启用空行规范化"""
        self._enable_empty_line_normalization = enabled
        if max_lines is not None:
            self._max_consecutive_empty_lines = max_lines

    def set_ssml_xml_cleaning(self, enabled: bool) -> None:
        """设置是否启用 SSML/XML 控制字符清洗"""
        self._enable_ssml_xml_cleaning = enabled
