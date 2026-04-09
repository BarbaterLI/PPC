"""PPC8 - 冰璃岩项目开发组 (BLY Team) - 终极文本转语音工具
支持 CLI 双模式、智能分章、高性能 TTS 处理
"""

__version__ = "8.0.0"
__author__ = "BLY Team"

from .config.manager import ConfigManager, get_default_config_dir
from .cli.typer_app import app as TyperApp
from .infrastructure.config.schema import PPC8Config

__all__ = [
    "__version__",
    "ConfigManager",
    "get_default_config_dir",
    "TyperApp",
    "PPC8Config",
]
