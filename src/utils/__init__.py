"""Public API for src.utils.

12.0.0 起子模块惰性导出（PEP 562）：``import src.utils.format`` 之类的轻量
引用不再连带拉起 asyncio/ssl/subprocess 等重依赖（CLI 冷启动实测省 ~90ms）；
首次访问符号时才加载对应子模块。
"""

from __future__ import annotations

from typing import Any

# 符号 → 子模块名（点号前）。新增符号请保持单一来源（子模块本体不改动）。
_SYMBOL_MAP: dict[str, str] = {
    # .core
    "DEFAULT_ENCODING_DETECT_BUFFER": "core",
    "DEFAULT_ENCODING_FALLBACK": "core",
    "DEFAULT_FILENAME_MAX_LENGTH": "core",
    "FILENAME_SANITIZE_PATTERN": "core",
    "AsyncObjectPool": "core",
    "CircularBuffer": "core",
    "MemoryMonitor": "core",
    "ObjectPool": "core",
    "PerformanceStats": "core",
    "detect_encoding": "core",
    "memory_efficient": "core",
    "sanitize_filename": "core",
    # .files
    "quick_probe": "files",
    "read_text_auto": "files",
    "safe_extract_zip": "files",
    # .elevation
    "ERROR_ACCESS_DENIED": "elevation",
    "ERROR_NO_SUCH_PRIVILEGE": "elevation",
    "handle_permission_error_or_elevate": "elevation",
    "is_admin": "elevation",
    "is_permission_error": "elevation",
    "is_windows": "elevation",
    "set_console_factory": "elevation",
    # .format
    "format_duration": "format",
    # .keep_awake
    "ScreenKeepAwake": "keep_awake",
    "keep_awake_context": "keep_awake",
}

__all__ = list(_SYMBOL_MAP)


def __getattr__(name: str) -> Any:
    target = _SYMBOL_MAP.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{target}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # 缓存，后续访问零开销
    return value


def __dir__() -> list[str]:
    import builtins

    return sorted(set(builtins.dir()) | set(_SYMBOL_MAP))
