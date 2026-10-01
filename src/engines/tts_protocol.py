"""``ppc.tts.v1`` —— TTS 合成后端协议（版本化能力协商 + 注册表）。

12.x 起，TTS 后端不再只是「一个碰巧有 ``synthesize`` 方法的对象」，而是
一份显式协议：

* **版本协商** —— 后端可声明 ``protocol_version()``（如 ``"ppc.tts.v1"``），
  宿主用 :func:`negotiate` 校验主版本兼容后才允许路由合成。未声明版本的
  对象按 v1 基线处理（向后兼容早期扩展）。
* **能力声明** —— ``capabilities()`` 返回 :class:`BackendCapabilities`
  （是否可枚举语音、是否可探测健康度等），宿主据此决定是否调用可选方法，
  而不是 try/except 撞运气。
* **惰性注册表** —— :class:`BackendRegistry` 以「名称 → 工厂」注册，
  实例只在首次解析时构造（不拖慢启动，不引入无用 import）。内置后端
  （如 ``openai_http``）与加载器导入的扩展后端共用同一注册表。

依赖方向（单向）：engines 层拥有本协议与注册表，且不得 import
``src.extensions``——内置后端按 :func:`negotiate` 的鸭子类型实现即可，
无需继承扩展框架的基类。``src.extensions`` 访问注册表时延迟导入本模块
（extensions → engines 单向），不得出现反向 import。
"""

from __future__ import annotations

import inspect
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

PROTOCOL = "ppc.tts.v1"
PROTOCOL_MAJOR = 1


@dataclass(frozen=True)
class BackendCapabilities:
    """后端可选能力声明 / optional capability flags."""

    voices: bool = False      # 支持 list_voices()
    probe: bool = False       # 支持 async probe()（ppc doctor 使用）
    cancellation: bool = False
    extra: tuple[str, ...] = field(default_factory=tuple)


def parse_protocol_version(raw: Any) -> tuple[str, int] | None:
    """把 ``"ppc.tts.v1"`` / ``"1"`` / ``1`` 归一化为 ``(ns, major)``。"""
    if isinstance(raw, int):
        return (PROTOCOL.rsplit(".", 1)[0], raw)
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("ppc.tts."):
        text = text[len("ppc.tts.") :]
    if text.startswith("v") or text.startswith("V"):
        text = text[1:]
    if not text.isdigit():
        return None
    return (PROTOCOL.rsplit(".", 1)[0], int(text))


def negotiate(plugin: Any) -> str | None:
    """校验 *plugin* 是否满足 ``ppc.tts.v1``；满足返回其名称，否则 None。

    必选面：``get_name()``、``is_available()``、可等待的 ``synthesize()``。
    可选面（存在则校验合理性）：``protocol_version()``、``capabilities()``。
    """
    for attr in ("get_name", "is_available", "synthesize"):
        if not callable(getattr(plugin, attr, None)):
            return None
    if not inspect.iscoroutinefunction(plugin.synthesize):
        # 允许同步实现的极薄包装（少见）：协程检查仅在明显不匹配时拒绝
        fn = plugin.synthesize
        if not (inspect.iscoroutinefunction(getattr(fn, "__wrapped__", fn)) or inspect.isfunction(fn)):
            return None
    version_attr = getattr(plugin, "protocol_version", None)
    if callable(version_attr):
        try:
            parsed = parse_protocol_version(version_attr())
        except Exception:  # noqa: BLE001
            parsed = None
        if parsed is None or parsed[1] != PROTOCOL_MAJOR:
            logger.warning(
                "TTS 后端 %r 协议版本不兼容（需要 ppc.tts.v%d）/ incompatible protocol version",
                plugin, PROTOCOL_MAJOR,
            )
            return None
    return str(plugin.get_name())


def capabilities_of(plugin: Any) -> BackendCapabilities:
    """读取后端能力；未声明时按可用方法探测。"""
    raw = getattr(plugin, "capabilities", None)
    if callable(raw):
        try:
            value = raw()
            if isinstance(value, BackendCapabilities):
                return value
            if isinstance(value, dict):
                known = {f for f in BackendCapabilities.__dataclass_fields__ if f != "extra"}
                kwargs = {k: bool(value.get(k, False)) for k in known if k in value}
                extra = tuple(str(x) for x in value.get("extra", ()) or ())
                return BackendCapabilities(extra=extra, **kwargs)
        except Exception:  # noqa: BLE001
            logger.debug("TTS 后端 capabilities() 异常，按方法探测回退", exc_info=True)
    return BackendCapabilities(
        voices=callable(getattr(plugin, "list_voices", None)),
        probe=callable(getattr(plugin, "probe", None)),
    )


class BackendRegistry:
    """名称 → 后端工厂的惰性注册表（内置后端与扩展后端共用）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._factories: dict[str, Callable[[], Any]] = {}
        self._instances: dict[str, Any] = {}
        self._sources: dict[str, str] = {}

    def register(self, name: str, factory: Callable[[], Any] | Any, *, source: str = "builtin") -> None:
        """注册后端。*factory* 可为零参工厂或现成实例。"""
        name = str(name)
        if not name:
            raise ValueError("backend name must be non-empty")
        with self._lock:
            if callable(factory) and not hasattr(factory, "synthesize"):
                self._factories[name] = factory
                self._instances.pop(name, None)
            else:
                self._factories[name] = lambda f=factory: f
                self._instances[name] = factory
            self._sources[name] = source

    def unregister(self, name: str) -> None:
        with self._lock:
            self._factories.pop(name, None)
            self._instances.pop(name, None)
            self._sources.pop(name, None)

    def unregister_source(self, source: str) -> None:
        with self._lock:
            for name in [n for n, s in self._sources.items() if s == source]:
                self._factories.pop(name, None)
                self._instances.pop(name, None)
                self._sources.pop(name, None)

    def names(self, *, source: str | None = None) -> list[str]:
        with self._lock:
            items = [n for n, s in self._sources.items() if source is None or s == source]
        return sorted(items)

    def get(self, name: str | None) -> Any | None:
        """按名称解析实例（首次访问时构造）；失败返回 None。"""
        if not name:
            return None
        with self._lock:
            if name in self._instances:
                return self._instances[name]
            factory = self._factories.get(name)
            if factory is None:
                return None
            try:
                instance = factory()
            except Exception:  # noqa: BLE001
                logger.warning("TTS 后端 %s 构造失败 / factory failed", name, exc_info=True)
                return None
            self._instances[name] = instance
            return instance

    def resolve(self, name: str | None) -> Any | None:
        """解析并协商：仅当后端满足 ``ppc.tts.v1`` 且可用时返回。"""
        plugin = self.get(name)
        if plugin is None:
            return None
        if negotiate(plugin) is None:
            return None
        try:
            available = bool(plugin.is_available())
        except Exception:  # noqa: BLE001
            available = False
        return plugin if available else None


backend_registry = BackendRegistry()

_builtins_loaded = False


def ensure_builtin_backends() -> None:
    """惰性导入内置后端子包完成注册（幂等；不占启动路径）。"""
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    try:
        from src.engines import backends  # noqa: F401  (import side-effect: registers)
    except Exception:  # noqa: BLE001
        logger.warning("内置 TTS 后端加载失败 / builtin backends import failed", exc_info=True)


__all__ = [
    "PROTOCOL",
    "PROTOCOL_MAJOR",
    "BackendCapabilities",
    "BackendRegistry",
    "backend_registry",
    "capabilities_of",
    "ensure_builtin_backends",
    "negotiate",
    "parse_protocol_version",
]
