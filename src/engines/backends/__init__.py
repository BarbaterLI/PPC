"""内置合成后端（实现 ``ppc.tts.v1`` 协议，经 :mod:`src.engines.tts_protocol` 注册）。

导入本子包即完成注册；子模块本身应保持惰性可导入（无重量级依赖）。
"""

from __future__ import annotations

from . import openai_http as _openai_http  # noqa: F401  (import = registers "openai_http")

__all__ = ["openai_http"]
