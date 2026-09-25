"""扩展生命周期钩子 Mixin。

将 :class:`Extension` 的默认空实现钩子按语义分为三组 Mixin：

* :class:`ExtensionLifecycleMixin` — 加载/执行/错误相关钩子
* :class:`ExtensionEnableMixin` — 启用/禁用相关钩子
* :class:`ExtensionConfigMixin` — 配置变更与 WebUI 面板相关钩子

所有钩子均为可选的默认空实现，扩展作者只需继承组合自己关心的 Mixin。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.extensions.base.core import ExtensionContext


class ExtensionLifecycleMixin:
    """加载/执行/错误相关钩子（默认空实现）。"""

    async def before_load(self, context: ExtensionContext) -> None:
        """扩展加载前调用（默认空实现）。

        参数:
            context: 扩展上下文，包含 loader、extension_dir、data_dir、sandbox 等信息
        """
        pass

    async def after_load(self, context: ExtensionContext) -> None:
        """扩展加载成功后调用（默认空实现）。

        参数:
            context: 扩展上下文，包含 loader、extension_dir、data_dir、sandbox 等信息
        """
        pass

    async def before_execute(self, context: ExtensionContext) -> None:
        """扩展执行前调用（默认空实现）。

        参数:
            context: 执行上下文
        """
        pass

    async def after_execute(self, context: ExtensionContext, result: Any) -> None:
        """扩展执行后调用（默认空实现）。

        参数:
            context: 执行上下文
            result: 执行结果
        """
        pass

    async def on_error(self, context: ExtensionContext, error: Exception) -> None:
        """扩展发生错误时调用（默认空实现）。

        参数:
            context: 错误上下文
            error: 异常对象
        """
        pass


class ExtensionEnableMixin:
    """启用/禁用相关钩子（默认空实现）。"""

    async def before_enable(self, context: ExtensionContext) -> None:
        """扩展被启用前调用（默认空实现）。"""
        pass

    async def after_disable(self, context: ExtensionContext) -> None:
        """扩展被禁用后调用（默认空实现）。"""
        pass

    async def on_enable(self, context: ExtensionContext) -> None:
        """扩展被启用时调用（默认空实现）。"""
        pass

    async def on_disable(self, context: ExtensionContext) -> None:
        """扩展被禁用时调用（默认空实现）。"""
        pass


class ExtensionConfigMixin:
    """配置变更与 WebUI 面板相关钩子（默认空实现）。"""

    async def on_config_change(self, context: ExtensionContext, config: dict[str, Any]) -> None:
        """配置变更通知（默认空实现）。

        参数:
            context: 扩展上下文
            config: 新的配置字典
        """
        pass

    def get_webui_config(self) -> dict[str, Any] | None:
        """返回扩展的 WebUI 面板配置（默认返回 None）。"""
        return None
