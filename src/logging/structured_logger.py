"""结构化日志系统
提供JSON格式、文本格式和Rich格式的日志输出
支持日志上下文管理和动态日志级别调整
"""

import json
import logging
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class LogFormat(str, Enum):
    """日志输出格式"""
    JSON = "json"
    TEXT = "text"
    RICH = "rich"


class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


LOG_LEVEL_MAP = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
    LogLevel.CRITICAL: logging.CRITICAL,
}


@dataclass
class LogConfig:
    """日志配置"""
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.TEXT
    output: str = "stdout"
    include_timestamp: bool = True
    include_level: bool = True
    include_module: bool = True
    include_extra: bool = True
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    json_indent: Optional[int] = None
    module_levels: Dict[str, LogLevel] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "format": self.format.value,
            "output": self.output,
            "include_timestamp": self.include_timestamp,
            "include_level": self.include_level,
            "include_module": self.include_module,
            "include_extra": self.include_extra,
            "timestamp_format": self.timestamp_format,
            "json_indent": self.json_indent,
            "module_levels": {k: v.value for k, v in self.module_levels.items()},
        }


class JsonFormatter(logging.Formatter):
    """JSON格式日志格式化器"""

    def __init__(self, config: LogConfig):
        self.config = config
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {}

        if self.config.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().strftime(self.config.timestamp_format)

        if self.config.include_level:
            log_data["level"] = record.levelname.lower()

        log_data["message"] = record.getMessage()

        if self.config.include_module:
            log_data["logger"] = record.name
            log_data["module"] = record.module
            log_data["function"] = record.funcName
            log_data["line"] = record.lineno

        if hasattr(record, "context") and record.context:
            log_data["context"] = record.context

        if self.config.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    "name", "msg", "args", "created", "filename", "funcName",
                    "levelname", "levelno", "lineno", "module", "msecs",
                    "pathname", "process", "processName", "relativeCreated",
                    "stack_info", "exc_info", "exc_text", "thread", "threadName",
                    "message", "context", "asctime"
                }:
                    try:
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)
            if extra_fields:
                log_data["extra"] = extra_fields

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            log_data["stack_trace"] = record.stack_info

        return json.dumps(log_data, indent=self.config.json_indent, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """文本格式日志格式化器"""

    def __init__(self, config: LogConfig):
        self.config = config
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        parts: List[str] = []

        if self.config.include_timestamp:
            timestamp = datetime.utcnow().strftime(self.config.timestamp_format)
            parts.append(f"[{timestamp}]")

        if self.config.include_level:
            level_name = record.levelname.ljust(8)
            parts.append(f"{level_name}")

        if self.config.include_module:
            parts.append(f"[{record.name}]")

        parts.append(record.getMessage())

        if hasattr(record, "context") and record.context:
            context_str = " ".join(f"{k}={v}" for k, v in record.context.items())
            parts.append(f"| {context_str}")

        message = " ".join(parts)

        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        if record.stack_info:
            message += f"\n{record.stack_info}"

        return message


class RichFormatter(logging.Formatter):
    """Rich格式日志格式化器（支持彩色输出）"""

    LEVEL_COLORS = {
        "DEBUG": "dim",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red bold",
    }

    def __init__(self, config: LogConfig):
        self.config = config
        self._rich_available = self._check_rich()
        super().__init__()

    def _check_rich(self) -> bool:
        try:
            from rich.console import Console
            from rich.text import Text
            return True
        except ImportError:
            return False

    def format(self, record: logging.LogRecord) -> str:
        if self._rich_available:
            return self._format_rich(record)
        return self._format_fallback(record)

    def _format_rich(self, record: logging.LogRecord) -> str:
        from rich.console import Console
        from rich.text import Text

        console = Console(force_terminal=True)
        text = Text()

        if self.config.include_timestamp:
            timestamp = datetime.utcnow().strftime(self.config.timestamp_format)
            text.append(f"[{timestamp}] ", style="dim")

        if self.config.include_level:
            level_style = self.LEVEL_COLORS.get(record.levelname, "")
            text.append(f"{record.levelname:8}", style=level_style)

        if self.config.include_module:
            text.append(f"[{record.name}] ", style="cyan")

        text.append(record.getMessage())

        if hasattr(record, "context") and record.context:
            context_str = " ".join(f"{k}={v}" for k, v in record.context.items())
            text.append(f" | {context_str}", style="dim")

        if record.exc_info:
            text.append(f"\n{self.formatException(record.exc_info)}", style="red")

        return str(text)

    def _format_fallback(self, record: logging.LogRecord) -> str:
        parts: List[str] = []

        if self.config.include_timestamp:
            timestamp = datetime.utcnow().strftime(self.config.timestamp_format)
            parts.append(f"[{timestamp}]")

        if self.config.include_level:
            parts.append(f"{record.levelname:8}")

        if self.config.include_module:
            parts.append(f"[{record.name}]")

        parts.append(record.getMessage())

        if hasattr(record, "context") and record.context:
            context_str = " ".join(f"{k}={v}" for k, v in record.context.items())
            parts.append(f"| {context_str}")

        message = " ".join(parts)

        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        return message


class LogContext:
    """日志上下文管理器"""

    _local = threading.local()

    def __init__(self, **kwargs: Any):
        self._new_context = kwargs
        self._previous_context: Optional[Dict[str, Any]] = None

    def __enter__(self) -> "LogContext":
        self._previous_context = self.get_context().copy()
        self._merge_context(self._new_context)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._set_context(self._previous_context or {})

    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        if not hasattr(cls._local, "context"):
            cls._local.context = {}
        return cls._local.context

    @classmethod
    def _set_context(cls, context: Dict[str, Any]) -> None:
        cls._local.context = context

    @classmethod
    def _merge_context(cls, context: Dict[str, Any]) -> None:
        current = cls.get_context()
        current.update(context)
        cls._local.context = current

    @classmethod
    def bind(cls, **kwargs: Any) -> "LogContext":
        return cls(**kwargs)

    @classmethod
    def unbind(cls, *keys: str) -> None:
        current = cls.get_context()
        for key in keys:
            current.pop(key, None)


@contextmanager
def log_context(**kwargs: Any):
    """日志上下文上下文管理器便捷函数"""
    ctx = LogContext(**kwargs)
    with ctx:
        yield ctx


class StructuredLogger:
    """结构化日志记录器"""

    def __init__(self, name: str, config: Optional[LogConfig] = None):
        self.name = name
        self.config = config or LogConfig()
        self._logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}
        self._setup_logger()

    def _setup_logger(self) -> None:
        level = LOG_LEVEL_MAP.get(self.config.level, logging.INFO)
        self._logger.setLevel(level)

        if self._logger.handlers:
            self._logger.handlers.clear()

        handler = self._create_handler()
        formatter = self._create_formatter()
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

        self._logger.propagate = False

    def _create_handler(self) -> logging.Handler:
        if self.config.output == "stdout":
            return logging.StreamHandler(sys.stdout)
        elif self.config.output == "stderr":
            return logging.StreamHandler(sys.stderr)
        else:
            return logging.FileHandler(self.config.output, encoding="utf-8")

    def _create_formatter(self) -> logging.Formatter:
        if self.config.format == LogFormat.JSON:
            return JsonFormatter(self.config)
        elif self.config.format == LogFormat.RICH:
            return RichFormatter(self.config)
        else:
            return TextFormatter(self.config)

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        extra = kwargs.pop("extra", {})
        exc_info = kwargs.pop("exc_info", None)
        stack_info = kwargs.pop("stack_info", False)

        context = {}
        context.update(LogContext.get_context())
        context.update(self._context)
        if kwargs:
            context.update(kwargs)

        if context:
            extra["context"] = context

        self._logger.log(level, message, extra=extra, exc_info=exc_info, stack_info=stack_info)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        kwargs["exc_info"] = True
        self._log(logging.ERROR, message, **kwargs)

    def bind(self, **kwargs: Any) -> "StructuredLogger":
        self._context.update(kwargs)
        return self

    def unbind(self, *keys: str) -> "StructuredLogger":
        for key in keys:
            self._context.pop(key, None)
        return self

    def with_context(self, **kwargs: Any) -> "StructuredLogger":
        new_logger = StructuredLogger(self.name, self.config)
        new_logger._context = {**self._context, **kwargs}
        return new_logger

    def set_level(self, level: Union[LogLevel, str]) -> None:
        if isinstance(level, str):
            level = LogLevel(level.lower())
        self.config.level = level
        self._logger.setLevel(LOG_LEVEL_MAP[level])

    def get_level(self) -> LogLevel:
        return self.config.level


class LoggerRegistry:
    """日志记录器注册表"""

    _instance: Optional["LoggerRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "LoggerRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._loggers: Dict[str, StructuredLogger] = {}
                    cls._instance._config: LogConfig = LogConfig()
        return cls._instance

    def get_logger(self, name: str) -> StructuredLogger:
        if name not in self._loggers:
            config = self._get_module_config(name)
            self._loggers[name] = StructuredLogger(name, config)
        return self._loggers[name]

    def _get_module_config(self, name: str) -> LogConfig:
        config = LogConfig(
            level=self._config.level,
            format=self._config.format,
            output=self._config.output,
            include_timestamp=self._config.include_timestamp,
            include_level=self._config.include_level,
            include_module=self._config.include_module,
            include_extra=self._config.include_extra,
            timestamp_format=self._config.timestamp_format,
            json_indent=self._config.json_indent,
        )

        for module_prefix, level in self._config.module_levels.items():
            if name.startswith(module_prefix):
                config.level = level
                break

        return config

    def set_global_config(self, config: LogConfig) -> None:
        self._config = config
        for name, logger in self._loggers.items():
            module_config = self._get_module_config(name)
            logger.config = module_config
            logger._setup_logger()

    def set_level(self, level: Union[LogLevel, str], module: Optional[str] = None) -> None:
        if isinstance(level, str):
            level = LogLevel(level.lower())

        if module:
            self._config.module_levels[module] = level
        else:
            self._config.level = level

        for name, logger in self._loggers.items():
            if module is None or name.startswith(module):
                logger.set_level(level)

    def get_config(self) -> LogConfig:
        return self._config

    def clear(self) -> None:
        self._loggers.clear()


def get_logger(name: str) -> StructuredLogger:
    """获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        StructuredLogger实例
    """
    registry = LoggerRegistry()
    return registry.get_logger(name)


def setup_logging(config: Optional[Union[LogConfig, Dict[str, Any]]] = None) -> None:
    """配置日志系统

    Args:
        config: 日志配置，可以是LogConfig实例或字典
    """
    if config is None:
        config = LogConfig()
    elif isinstance(config, dict):
        if "level" in config and isinstance(config["level"], str):
            config["level"] = LogLevel(config["level"].lower())
        if "format" in config and isinstance(config["format"], str):
            config["format"] = LogFormat(config["format"].lower())
        if "module_levels" in config:
            config["module_levels"] = {
                k: LogLevel(v.lower()) if isinstance(v, str) else v
                for k, v in config["module_levels"].items()
            }
        config = LogConfig(**config)

    registry = LoggerRegistry()
    registry.set_global_config(config)


def set_log_level(level: Union[LogLevel, str], module: Optional[str] = None) -> None:
    """设置日志级别

    Args:
        level: 日志级别
        module: 模块名称前缀，为None时设置全局级别
    """
    registry = LoggerRegistry()
    registry.set_level(level, module)


def bind_context(**kwargs: Any) -> LogContext:
    """绑定日志上下文

    Args:
        **kwargs: 上下文字段

    Returns:
        LogContext实例
    """
    return LogContext.bind(**kwargs)


def unbind_context(*keys: str) -> None:
    """解绑日志上下文字段

    Args:
        *keys: 要解绑的字段名
    """
    LogContext.unbind(*keys)


def get_current_context() -> Dict[str, Any]:
    """获取当前日志上下文

    Returns:
        当前上下文字典
    """
    return LogContext.get_context().copy()
