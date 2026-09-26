"""分布式配置 Schema

12.x 精简：仅保留真实被读取的 ``nodes`` 与 ``node_max_concurrency``。
负载均衡策略 / 容错迁移 / 健康检查间隔等由 Web API 请求参数与运行时
构造参数控制（见 src/web/api/distributed.py 与 DistributedScheduler），
不再作为配置文件键存在。旧配置中的多余键被静默忽略。
"""

from pydantic import BaseModel, Field


class DistributedNodeConfig(BaseModel):
    """分布式节点配置"""

    host: str = Field(..., description="节点 IP 地址 (IPv4)")
    port: int = Field(..., ge=1, le=65535, description="节点端口")
    max_concurrency: int = Field(default=4, ge=1, le=64, description="节点最大并发数")
    enabled: bool = Field(default=True, description="是否启用该节点")


class DistributedConfig(BaseModel):
    """分布式配置"""

    node_max_concurrency: int = Field(default=4, ge=1, le=64, description="添加节点时的默认最大并发数")
    nodes: list[DistributedNodeConfig] = Field(default_factory=list, description="远程节点列表")
