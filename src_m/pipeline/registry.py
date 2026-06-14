from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PipelineStepExecutor(ABC):
    """管道步骤执行器协议"""

    @abstractmethod
    async def execute(self, params: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行步骤

        Args:
            params: 步骤参数（来自管道定义）
            inputs: 上游步骤的输出（来自依赖步骤）

        Returns:
            步骤输出数据，传递给下游步骤
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """返回步骤名称"""
        pass

    @abstractmethod
    def get_input_type(self) -> str:
        """返回输入数据类型"""
        pass

    @abstractmethod
    def get_output_type(self) -> str:
        """返回输出数据类型"""
        pass


class StepRegistry:
    """管道步骤注册表"""

    def __init__(self):
        self._executors: Dict[str, PipelineStepExecutor] = {}

    def register(self, executor: PipelineStepExecutor) -> None:
        """注册步骤执行器"""
        name = executor.get_name()
        self._executors[name] = executor
        logger.info(f"Registered pipeline step: {name}")

    def unregister(self, name: str) -> bool:
        """注销步骤执行器"""
        if name in self._executors:
            del self._executors[name]
            return True
        return False

    def get_step(self, name: str) -> Optional[PipelineStepExecutor]:
        """获取步骤执行器"""
        return self._executors.get(name)

    def list_steps(self) -> List[Dict[str, str]]:
        """列出所有已注册步骤"""
        return [
            {"name": e.get_name(), "input_type": e.get_input_type(), "output_type": e.get_output_type()}
            for e in self._executors.values()
        ]

    def has_step(self, name: str) -> bool:
        """检查步骤是否已注册"""
        return name in self._executors

    def clear(self) -> None:
        """清空注册表"""
        self._executors.clear()
