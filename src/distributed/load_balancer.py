"""负载均衡组件：节点选择策略 ABC + 策略注册表 + 选择入口。

原 ``NodePool.get_best_node`` 内散落的 if/dict 策略统一为
:class:`NodeSelectionStrategy` 策略类，经 :class:`LoadBalancer`
注册表按名实例化（ABC 注入点）；外部自定义策略
（:class:`src.extensions.base.LoadBalanceStrategy`）作为最高优先级覆盖。
"""

import random
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from src.distributed.node_pool import NodeInfo
from src.extensions.base import LoadBalanceStrategy


class NodeSelectionStrategy(ABC):
    """节点选择策略基类（ABC 注入点）。"""

    name: ClassVar[str]

    @abstractmethod
    async def select(
        self,
        available_nodes: list[NodeInfo],
        task_context: dict[str, Any] | None = None,
    ) -> NodeInfo | None:
        """从可用节点中选择最佳节点；无可用节点时返回 None。"""
        pass

    def get_name(self) -> str:
        return self.name


class RoundRobinStrategy(NodeSelectionStrategy):
    """轮询策略。"""

    name = "round_robin"

    def __init__(self) -> None:
        self._index = 0

    async def select(
        self,
        available_nodes: list[NodeInfo],
        task_context: dict[str, Any] | None = None,
    ) -> NodeInfo | None:
        if not available_nodes:
            return None
        node = available_nodes[self._index % len(available_nodes)]
        self._index += 1
        return node


class LeastConnectionsStrategy(NodeSelectionStrategy):
    """最少连接策略。"""

    name = "least_connections"

    async def select(
        self,
        available_nodes: list[NodeInfo],
        task_context: dict[str, Any] | None = None,
    ) -> NodeInfo | None:
        if not available_nodes:
            return None
        return min(available_nodes, key=lambda n: n.current_concurrency)


class BestResponseTimeStrategy(NodeSelectionStrategy):
    """最快响应策略（平均响应时间最小；无统计时视为无穷大）。"""

    name = "best_response_time"

    async def select(
        self,
        available_nodes: list[NodeInfo],
        task_context: dict[str, Any] | None = None,
    ) -> NodeInfo | None:
        if not available_nodes:
            return None
        return min(
            available_nodes,
            key=lambda n: n.avg_response_time if n.avg_response_time > 0 else float("inf"),
        )


class RandomStrategy(NodeSelectionStrategy):
    """随机策略（未知策略名的兜底，与旧版行为一致）。"""

    name = "random"

    async def select(
        self,
        available_nodes: list[NodeInfo],
        task_context: dict[str, Any] | None = None,
    ) -> NodeInfo | None:
        if not available_nodes:
            return None
        return random.choice(available_nodes)


class LoadBalancer:
    """策略注册表 + 选择入口。

    内置策略经 :meth:`register` 注册；策略实例按名缓存复用
    （保留轮询等有状态策略的索引）。未知策略名回退 :class:`RandomStrategy`。
    """

    _registry: ClassVar[dict[str, type[NodeSelectionStrategy]]] = {
        "round_robin": RoundRobinStrategy,
        "least_connections": LeastConnectionsStrategy,
        "best_response_time": BestResponseTimeStrategy,
        "random": RandomStrategy,
    }

    def __init__(
        self,
        default_strategy: str = "round_robin",
        custom_strategy: LoadBalanceStrategy | None = None,
    ) -> None:
        self.default_strategy = default_strategy
        self.custom_strategy = custom_strategy
        self._instances: dict[str, NodeSelectionStrategy] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: type[NodeSelectionStrategy]) -> None:
        cls._registry[name] = strategy_cls

    @classmethod
    def registered_names(cls) -> list[str]:
        return list(cls._registry.keys())

    def set_custom_strategy(self, strategy: LoadBalanceStrategy | None) -> None:
        self.custom_strategy = strategy

    def get_strategy(self, name: str) -> NodeSelectionStrategy:
        if name not in self._instances:
            strategy_cls = self._registry.get(name) or RandomStrategy
            self._instances[name] = strategy_cls()
        return self._instances[name]

    async def select_node(
        self,
        available_nodes: list[NodeInfo],
        strategy: str | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> NodeInfo | None:
        if not available_nodes:
            return None
        if self.custom_strategy is not None:
            return await self.custom_strategy.select_node(available_nodes, task_context)
        return await self.get_strategy(strategy or self.default_strategy).select(
            available_nodes, task_context
        )
