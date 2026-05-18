"""用户自定义扩展的扩展接口。

本模块定义了用户扩展必须实现的核心接口，以便与分布式 TTS 系统集成。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExtensionType(Enum):
    """扩展类型枚举。"""
    LOAD_BALANCE_STRATEGY = "load_balance_strategy"
    HEALTH_CHECK_STRATEGY = "health_check_strategy"
    TASK_SCHEDULING_STRATEGY = "task_scheduling_strategy"
    METRICS_EXPORTER = "metrics_exporter"
    EXECUTOR = "executor"
    TOOL_INTEGRATION = "tool_integration"


@dataclass
class ExtensionMetadata:
    """扩展元数据。"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    extension_type: Optional[ExtensionType] = None
    tags: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    config_schema: Optional[Any] = None


class Extension(ABC):
    """基础扩展类。"""

    def __init__(self, metadata: ExtensionMetadata):
        self.metadata = metadata
        self._initialized = False
        self._enabled = True

    @abstractmethod
    async def initialize(self) -> None:
        """初始化扩展。"""
        self._initialized = True

    @abstractmethod
    async def cleanup(self) -> None:
        """清理扩展资源。"""
        self._initialized = False

    def on_enable(self) -> None:
        """扩展被启用时调用（默认空实现）。"""
        pass

    def on_disable(self) -> None:
        """扩展被禁用时调用（默认空实现）。"""
        pass

    def on_config_change(self, config: Dict[str, Any]) -> None:
        """配置变更通知（默认空实现）。"""
        pass

    def get_webui_config(self) -> Optional[Dict[str, Any]]:
        """返回扩展的 WebUI 面板配置（默认返回 None）。"""
        return None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def enabled(self) -> bool:
        return self._enabled

    def publish_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """通过 EventBus 发布扩展事件。"""
        try:
            from src_m.events.event_bus import get_event_bus, Event
            bus = get_event_bus()
            event = Event(source=self.metadata.name, metadata=data or {})
            bus.publish(event)
        except Exception as e:
            logger.warning(f"Extension {self.metadata.name} failed to publish event: {e}")


class LoadBalanceStrategy(ABC):
    """负载均衡策略接口

    用户可实现此接口来创建自定义负载均衡策略。
    """

    @abstractmethod
    async def select_node(
        self,
        available_nodes: List[Any],
        task_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """从可用节点中选择最佳节点

        参数:
            available_nodes: 可用节点对象列表
            task_context: 可选任务上下文（文本长度、优先级等）

        返回:
            选中的节点对象，若无合适节点则返回 None
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """返回策略名称。"""
        pass


class HealthCheckStrategy(ABC):
    """健康检查策略接口

    用户可实现此接口来创建自定义健康检查策略。
    """

    @abstractmethod
    async def check_node_health(self, node: Any) -> bool:
        """检查节点是否健康

        参数:
            node: 要检查的节点对象

        返回:
            节点健康返回 True，否则返回 False
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the strategy name"""
        pass


class TaskSchedulingStrategy(ABC):
    """任务调度策略接口

    用户可实现此接口来创建自定义任务调度策略。
    """

    @abstractmethod
    async def schedule_task(
        self,
        task: Any,
        available_nodes: List[Any],
    ) -> Optional[Any]:
        """将任务调度到合适的节点

        参数:
            task: 要调度的任务对象
            available_nodes: 可用节点对象列表

        返回:
            选中的节点对象，若无合适节点则返回 None
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the strategy name"""
        pass


class MetricsExporter(ABC):
    """指标导出器接口

    用户可实现此接口来创建自定义指标导出器。
    """

    @abstractmethod
    async def export_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        """导出指标数据

        参数:
            metrics_data: 要导出的指标数据

        返回:
            导出成功返回 True，否则返回 False
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """返回导出器名称。"""
        pass


class ToolIntegration(ABC):
    """工具集成接口

    用户可实现此接口来创建自定义工具集成扩展。
    """

    @abstractmethod
    def is_available(self) -> bool:
        """检查工具是否可用（已安装/可访问）"""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
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
    def get_status(self) -> Dict[str, Any]:
        """获取执行器状态"""
        pass
