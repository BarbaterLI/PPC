"""工具集成与执行器接口。"""

from abc import ABC, abstractmethod
from typing import Any


class ToolIntegration(ABC):
    """工具集成接口

    用户可实现此接口来创建自定义工具集成扩展。
    """

    @abstractmethod
    def is_available(self) -> bool:
        """检查工具是否可用（已安装/可访问）"""
        pass

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """获取工具信息（名称、版本、路径等）"""
        pass

    def start(self, **kwargs) -> Any:
        """启动工具（默认实现）"""
        pass

    def stop(self) -> None:
        """停止工具（默认实现）"""
        pass


class ExecutorExtension(ABC):
    """执行器扩展接口

    用户可实现此接口来创建自定义执行器扩展。
    """

    @abstractmethod
    async def execute(self, task: Any) -> Any:
        """执行任务"""
        pass

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """获取执行器状态"""
        pass
