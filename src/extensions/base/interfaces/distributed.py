"""分布式调度策略接口（负载均衡/健康检查/任务调度/指标导出）。"""

from abc import ABC, abstractmethod
from typing import Any


class LoadBalanceStrategy(ABC):
    """负载均衡策略接口

    用户可实现此接口来创建自定义负载均衡策略。
    """

    @abstractmethod
    async def select_node(
        self,
        available_nodes: list[Any],
        task_context: dict[str, Any] | None = None,
    ) -> Any | None:
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
        available_nodes: list[Any],
    ) -> Any | None:
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
    async def export_metrics(self, metrics_data: dict[str, Any]) -> bool:
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
