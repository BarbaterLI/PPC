"""章节切分共享类型

``ChapterInfo`` 为全库唯一定义（字段取原 splitter / engine 两处定义的并集），
``SplitterExecutor``（CLI ``ppc split`` / web split 端点）与
``ChapterEngine``（EPUB 分章）共用。
"""

from dataclasses import dataclass


@dataclass
class ChapterInfo:
    """章节信息"""

    index: int
    title: str
    start_line: int
    end_line: int
    content: str
    _volume_index: int | None = None
    _volume_title: str | None = None


@dataclass
class VolumeInfo:
    """卷信息"""

    index: int
    title: str
    start_line: int
    end_line: int
    chapters: list["ChapterInfo"] | None = None


__all__ = [
    "ChapterInfo",
    "VolumeInfo",
]
