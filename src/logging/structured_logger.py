"""Structured logging implementation

Provides JSON, text, and Rich format structured logging output.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log level enumeration"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """Log format enumeration"""

    JSON = "json"
    TEXT = "text"
    RICH = "rich"


@dataclass
class LogConfig:
    """Logging configuration"""

    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.TEXT
    output: str | None = None
    max_file_size_mb: int = 100
    backup_count: int = 5
    console_output: bool = True
    include_timestamp: bool = True
    include_thread: bool = False
    include_module: bool = True
    rich_theme: str = "default"
    enable_performance: bool = False
    performance_threshold_ms: float = 100.0


@dataclass
class LogContext:
    """Log context information"""

    request_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {}
        if self.request_id:
            data["request_id"] = self.request_id
        if self.user_id:
            data["user_id"] = self.user_id
        if self.session_id:
            data["session_id"] = self.session_id
        if self.trace_id:
            data["trace_id"] = self.trace_id
        if self.span_id:
            data["span_id"] = self.span_id
        data.update(self.extra)
        return data


class JsonFormatter(logging.Formatter):
    """JSON format log formatter"""

    def __init__(self, config: LogConfig | None = None):
        super().__init__()
        self.config = config or LogConfig()

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module if self.config.include_module else None,
            "function": record.funcName,
            "line": record.lineno,
        }

        if self.config.include_thread:
            log_data["thread"] = record.threadName

        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)

        context = get_current_context()
        if context:
            ctx_dict = context.to_dict()
            if ctx_dict:
                log_data["context"] = ctx_dict

        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        return json.dumps(log_data, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Text format log formatter"""

    def __init__(self, config: LogConfig | None = None):
        self.config = config or LogConfig()
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        if config and config.include_module:
            fmt = "%(asctime)s [%(levelname)s] %(name)s (%(module)s): %(message)s"
        if config and config.include_thread:
            fmt = "%(asctime)s [%(levelname)s] %(threadName)s %(name)s: %(message)s"
        super().__init__(fmt)


class RichFormatter(logging.Formatter):
    """Rich format log formatter"""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def __init__(self, config: LogConfig | None = None):
        super().__init__()
        self.config = config or LogConfig()

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        message = f"{color}[{timestamp}] [{record.levelname}] {record.name}: {self.RESET}{record.getMessage()}"

        if record.exc_info and record.exc_info[0] is not None:
            message += f"\n{self.COLORS['ERROR']}{self.formatException(record.exc_info)}{self.RESET}"

        return message


class StructuredLogger:
    """Structured logger

    Provides structured logging, context support, and multiple output formats.
    """

    def __init__(self, name: str, config: LogConfig | None = None):
        self.config = config or LogConfig()
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, self.config.level.value))
        self._logger.handlers.clear()
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup handlers"""
        if self.config.console_output:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(getattr(logging, self.config.level.value))
            console.setFormatter(self._create_formatter())
            self._logger.addHandler(console)

        if self.config.output:
            os.makedirs(
                os.path.dirname(self.config.output) if os.path.dirname(self.config.output) else ".", exist_ok=True
            )
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                self.config.output,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(getattr(logging, self.config.level.value))
            file_handler.setFormatter(self._create_formatter())
            self._logger.addHandler(file_handler)

    def _create_formatter(self) -> logging.Formatter:
        """Create formatter based on config"""
        if self.config.format == LogFormat.JSON:
            return JsonFormatter(self.config)
        elif self.config.format == LogFormat.RICH:
            return RichFormatter(self.config)
        return TextFormatter(self.config)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message"""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message"""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message"""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message"""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message"""
        self._log(logging.CRITICAL, message, **kwargs)

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Internal log method"""
        extra = {"extra": kwargs} if kwargs else {}
        self._logger.log(level, message, extra=extra)

    def set_level(self, level: LogLevel) -> None:
        """Set log level"""
        self._logger.setLevel(getattr(logging, level.value))
        for handler in self._logger.handlers:
            handler.setLevel(getattr(logging, level.value))


class LoggerRegistry:
    """Logger registry for managing multiple logger instances"""

    _instance: LoggerRegistry | None = None
    _lock = threading.Lock()
    _loggers: dict[str, StructuredLogger] = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._loggers: dict[str, StructuredLogger] = {}
        return cls._instance

    def get_logger(self, name: str, config: LogConfig | None = None) -> StructuredLogger:
        """Get or create logger instance"""
        if name not in self._loggers:
            with self._lock:
                if name not in self._loggers:
                    self._loggers[name] = StructuredLogger(name, config)
        return self._loggers[name]

    def set_level_all(self, level: LogLevel) -> None:
        """Set log level for all loggers"""
        for log in self._loggers.values():
            log.set_level(level)


_context_local = threading.local()


def _get_context_stack() -> list[LogContext]:
    stack: list[LogContext] = getattr(_context_local, "stack", [])
    if not isinstance(stack, list):
        stack = []
        _context_local.stack = stack
    return stack


@contextmanager
def log_context(**kwargs: Any):
    """Context manager for log context

    Usage:
        with log_context(request_id="123", user_id="abc"):
            logger.info("message")
    """
    stack = _get_context_stack()

    current = get_current_context() or LogContext()

    for key, value in kwargs.items():
        if hasattr(current, key) and key not in ("extra",):
            setattr(current, key, value)
        else:
            current.extra[key] = value

    stack.append(current)
    try:
        yield
    finally:
        stack.pop()


def get_current_context() -> LogContext | None:
    """Get current log context"""
    stack = _get_context_stack()
    if stack:
        return stack[-1]
    return None


def bind_context(**kwargs: Any) -> None:
    """Bind values to current context"""
    ctx = get_current_context() or LogContext()
    for key, value in kwargs.items():
        if hasattr(ctx, key) and key not in ("extra",):
            setattr(ctx, key, value)
        else:
            ctx.extra[key] = value
    stack = _get_context_stack()
    if not stack:
        stack.append(ctx)


def unbind_context(*keys: str) -> None:
    """Unbind values from current context"""
    ctx = get_current_context()
    if ctx:
        for key in keys:
            if hasattr(ctx, key) and key not in ("extra",):
                setattr(ctx, key, None)
            elif key in ctx.extra:
                del ctx.extra[key]


_registry = LoggerRegistry()


def get_logger(name: str, config: LogConfig | None = None) -> StructuredLogger:
    """Get structured logger"""
    return _registry.get_logger(name, config)


def setup_logging(config: LogConfig | None = None) -> StructuredLogger:
    """Setup logging system"""
    return get_logger("ppc10", config)


def set_log_level(level: LogLevel) -> None:
    """Set global log level"""
    _registry.set_level_all(level)
