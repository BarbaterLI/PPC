"""CLI module - Command Line Interface

Typer + Rich 双入口。12.0.0 起本包 ``__init__`` 不再 eager 导入 ``typer_app``
（连带 typer/click/pygments 全链），改为 PEP 562 惰性转发：``from src.cli
import app`` 语义不变，但 ``import src.cli.output`` 不再触发 ~120ms 的
typer 导入链。
"""

from __future__ import annotations

from typing import Any

_SYMBOL_MAP: dict[str, str] = {
    "app": "typer_app",
    "run": "typer_app",
    "get_output": "output",
    "OutputFormatter": "output",
    "setup_logging": "output",
    "setup_console": "output",
    "console": "output",
}

__all__ = list(_SYMBOL_MAP)


def __getattr__(name: str) -> Any:
    target = _SYMBOL_MAP.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{target}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    import builtins

    return sorted(set(builtins.dir()) | set(_SYMBOL_MAP))
