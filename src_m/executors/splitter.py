"""分割执行器 - 向后兼容性模块

包含从拆分模块重新导出的内容，保持 API 不变。
"""

from .splitter_core import (
    SplitterExecutor,
    ChapterInfo,
    VolumeInfo,
    split_file
)
from .splitter_strategies import (
    split_directory,
    _split_content,
    _split_hierarchical,
    _split_flat
)

# 为 SplitterExecutor 添加 split_directory 方法
def _split_directory_method(
    self,
    input_dir,
    output_dir,
    pattern="*.txt"
):
    return split_directory(self, input_dir, output_dir, pattern)

SplitterExecutor.split_directory = _split_directory_method

# 重新导出
__all__ = [
    "SplitterExecutor",
    "ChapterInfo",
    "VolumeInfo"
]
