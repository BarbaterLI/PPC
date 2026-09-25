"""文本处理策略接口（分割/预处理/后处理/元数据）。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SplitSegment:
    """文本分割段数据类。

    表示文本分割后的一个片段，包含位置、内容和元数据信息。
    """

    index: int
    title: str
    content: str
    start_pos: int
    end_pos: int
    metadata: dict[str, Any] = field(default_factory=dict)


class SplitStrategy(ABC):
    """文本分割策略接口

    用户可实现此接口来创建自定义文本分割策略。
    """

    @abstractmethod
    def split(self, content: str, options: dict[str, Any]) -> list[SplitSegment]:
        """将内容分割为多个片段

        参数:
            content: 要分割的原始文本内容
            options: 分割选项配置

        返回:
            分割后的片段列表
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """返回分割策略名称。"""
        pass

    def get_description(self) -> str:
        """返回分割策略描述（默认空字符串）。"""
        return ""

    def get_default_options(self) -> dict[str, Any]:
        """返回默认分割选项配置。"""
        return {}


class TextPreprocessor(ABC):
    """文本预处理器接口

    用户可实现此接口来创建自定义文本预处理器，
    在文本分割前对原始内容进行预处理。
    """

    @abstractmethod
    def process(self, text: str, context: dict[str, Any]) -> str:
        """处理文本

        参数:
            text: 原始文本
            context: 处理上下文信息

        返回:
            预处理后的文本
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """返回预处理器名称。"""
        pass


class TextPostprocessor(ABC):
    """文本后处理器接口

    用户可实现此接口来创建自定义文本后处理器，
    在文本分割后对分割片段进行后处理。
    """

    @abstractmethod
    def process(self, segments: list[SplitSegment], context: dict[str, Any]) -> list[SplitSegment]:
        """处理分割后的片段列表

        参数:
            segments: 分割后的片段列表
            context: 处理上下文信息

        返回:
            后处理后的片段列表
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """返回后处理器名称。"""
        pass


class MetadataProvider(ABC):
    """元数据提供者接口

    用户可实现此接口来创建自定义元数据提供者，
    为文本内容提供额外的元数据信息。
    """

    @abstractmethod
    def get_metadata(self, content: str) -> dict[str, Any]:
        """获取内容的元数据

        参数:
            content: 文本内容

        返回:
            元数据字典
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """返回元数据提供者名称。"""
        pass
