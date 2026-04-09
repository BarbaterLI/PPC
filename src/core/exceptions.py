"""核心异常定义
定义项目统一的异常类型

修复记录 (2026-04-08):
- 添加完整的类型注解
- 使用 Optional 标注可为 None 的参数
"""

from typing import Optional, Dict, Any


class PPC8Error(Exception):
    """PPC8 项目基础异常"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class EngineError(PPC8Error):
    """引擎相关错误"""
    pass


class ExecutorError(PPC8Error):
    """执行器相关错误"""
    pass


class InitializationError(PPC8Error):
    """初始化错误"""
    pass


class CleanupError(PPC8Error):
    """清理错误"""
    pass


class ValidationError(PPC8Error):
    """验证错误"""
    pass


class ConfigurationError(PPC8Error):
    """配置错误"""
    pass


class TimeoutError(PPC8Error):
    """超时错误"""
    pass


class NetworkError(PPC8Error):
    """网络错误"""
    pass


class FileIOError(PPC8Error):
    """文件IO错误"""
    pass


class TTSError(PPC8Error):
    """TTS相关错误"""
    pass


class ChapterError(PPC8Error):
    """章节处理相关错误"""
    pass
