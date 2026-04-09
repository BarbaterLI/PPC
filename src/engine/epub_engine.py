"""EPUB引擎兼容性模块
为了保持向后兼容性，此模块将所有内容重定向到 src.engines.epub_engine 模块
"""

import warnings

warnings.warn(
    "src.engine.epub_engine 已弃用，请使用 src.engines.epub_engine",
    DeprecationWarning,
    stacklevel=2
)

from src.engines.epub_engine import *
