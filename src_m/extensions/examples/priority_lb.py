"""Example custom load balance strategy extension.

This is an example extension that users can use as a template
for creating their own load balancing strategies.
"""

from src_m.extensions.base import Extension, ExtensionMetadata, ExtensionType, LoadBalanceStrategy
from typing import Any, Dict, List, Optional


class PriorityBasedStrategy(LoadBalanceStrategy):
    """A simple priority-based load balancing strategy.

    Selects the first available node in the list.
    Users can customize this to implement their own logic.
    """

    def __init__(self):
        metadata = ExtensionMetadata(
            name="priority_based_lb",
            version="1.0.0",
            description="Priority-based load balancing strategy",
            author="User Extension Example",
            extension_type=ExtensionType.LOAD_BALANCE_STRATEGY,
            tags=["load_balance", "priority", "example"],
        )
        super().__init__(metadata)

    async def initialize(self) -> None:
        """Initialize the strategy"""
        await super().initialize()

    async def cleanup(self) -> None:
        """Cleanup the strategy"""
        await super().cleanup()

    async def select_node(
        self,
        available_nodes: List[Any],
        task_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Select the first available node"""
        if not available_nodes:
            return None
        return available_nodes[0]

    def get_name(self) -> str:
        return self.metadata.name


"""Global instance of the priority-based strategy extension"""
extension = PriorityBasedStrategy()
