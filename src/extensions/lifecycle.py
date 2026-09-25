"""扩展生命周期管理：启用 / 禁用 / 卸载。

:class:`ExtensionLifecycle` 操作 :class:`ExtensionLoader` 持有的
扩展注册表（``_loaded_extensions`` / ``_extension_modules``），
门面通过委托保持公共 API 不变。
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from src.extensions.security import AuditLogger, AuditResult

if TYPE_CHECKING:
    from src.extensions.base import Extension
    from src.extensions.loader import ExtensionLoader

logger = logging.getLogger(__name__)


class ExtensionLifecycle:
    """扩展启用/禁用/卸载生命周期管理。"""

    def __init__(self, loader: ExtensionLoader):
        self._loader = loader

    @property
    def _loaded_extensions(self) -> dict[str, Extension]:
        return self._loader._loaded_extensions

    @property
    def _extension_modules(self) -> dict[str, str]:
        return self._loader._extension_modules

    @property
    def _audit_logger(self) -> AuditLogger:
        return self._loader._audit_logger

    async def enable(self, name: str) -> bool:
        """启用扩展（触发 before_enable / on_enable 钩子）。"""
        ext = self._loaded_extensions.get(name)
        if ext is None:
            return False
        if not hasattr(ext, "_enabled"):
            ext._enabled = True
        if ext._enabled:
            return True
        try:
            ctx = ext._context
            if ctx is None:
                logger.warning(f"Extension {name} has no context, cannot enable hooks")
                return False
            try:
                await ext.before_enable(ctx)
            except Exception as hook_e:
                logger.warning(f"Extension {name} before_enable hook error: {hook_e}")
            await ext.on_enable(ctx)
            ext._enabled = True
            logger.info(f"Extension enabled: {name}")
            self._audit_logger.log_event(
                name, "EXTENSION_ENABLE", name, AuditResult.SUCCESS
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to enable extension {name}: {e}")
            self._audit_logger.log_event(
                name, "EXTENSION_ENABLE", name, AuditResult.ERROR, {"error": str(e)}
            )
            return False

    async def disable(self, name: str) -> bool:
        """禁用扩展（触发 on_disable / after_disable 钩子）。"""
        ext = self._loaded_extensions.get(name)
        if ext is None:
            return False
        if not hasattr(ext, "_enabled") or not ext._enabled:
            return True
        try:
            ctx = ext._context
            if ctx is None:
                logger.warning(f"Extension {name} has no context, cannot disable hooks")
                return False
            await ext.on_disable(ctx)
            ext._enabled = False
            try:
                await ext.after_disable(ctx)
            except Exception as hook_e:
                logger.warning(f"Extension {name} after_disable hook error: {hook_e}")
            logger.info(f"Extension disabled: {name}")
            self._audit_logger.log_event(
                name, "EXTENSION_DISABLE", name, AuditResult.SUCCESS
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to disable extension {name}: {e}")
            self._audit_logger.log_event(
                name, "EXTENSION_DISABLE", name, AuditResult.ERROR, {"error": str(e)}
            )
            return False

    async def unload(self, name: str) -> bool:
        """卸载并移除指定扩展。

        清理扩展资源、触发禁用钩子，并从所有内部注册表中移除该扩展。

        参数:
            name: 扩展名称

        返回:
            True 表示卸载成功，False 表示扩展不存在
        """
        ext = self._loaded_extensions.get(name)
        if ext is None:
            return False
        try:
            # 清理扩展资源（异常不阻断卸载流程）
            try:
                await ext.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up extension {name}: {e}")

            # 若扩展仍处于启用状态，触发禁用钩子
            if getattr(ext, "_enabled", False):
                ctx = ext._context
                if ctx is not None:
                    try:
                        await ext.on_disable(ctx)
                    except Exception as hook_e:
                        logger.warning(f"Extension {name} on_disable hook error: {hook_e}")
                    ext._enabled = False
                    try:
                        await ext.after_disable(ctx)
                    except Exception as hook_e:
                        logger.warning(f"Extension {name} after_disable hook error: {hook_e}")

            # 从所有内部注册表中移除
            module_name = self._extension_modules.pop(name, None)
            if module_name is not None:
                sys.modules.pop(module_name, None)
            self._loaded_extensions.pop(name, None)
            backend_name = getattr(ext, "_ppc_backend_name", None)
            if backend_name:
                try:
                    from src.engines.tts_protocol import backend_registry

                    backend_registry.unregister(backend_name)
                except Exception:  # noqa: BLE001
                    pass

            logger.info(f"Extension unloaded: {name}")
            self._audit_logger.log_event(
                name, "EXTENSION_UNLOAD", name, AuditResult.SUCCESS
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to unload extension {name}: {e}")
            self._audit_logger.log_event(
                name, "EXTENSION_UNLOAD", name, AuditResult.ERROR, {"error": str(e)}
            )
            return False
