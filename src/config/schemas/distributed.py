"""分布式配置 Schema（节点 / 负载均衡 / 容错迁移）"""

from pydantic import BaseModel, Field


class DistributedNodeConfig(BaseModel):
    """分布式节点配置"""

    host: str = Field(..., description="节点 IP 地址 (IPv4)")
    port: int = Field(..., ge=1, le=65535, description="节点端口")
    max_concurrency: int = Field(default=4, ge=1, le=64, description="节点最大并发数")
    enabled: bool = Field(default=True, description="是否启用该节点")


class AdaptiveLoadBalanceConfig(BaseModel):
    """自适应负载均衡配置"""

    enabled: bool = Field(default=False, description="启用自适应负载均衡")
    task_feature_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="任务特征权重")
    history_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="历史性能权重")
    decay_factor: float = Field(default=0.9, ge=0.0, le=1.0, description="历史数据衰减因子")
    history_window_size: int = Field(default=100, ge=10, le=1000, description="历史数据窗口大小")
    long_text_threshold: int = Field(default=2000, ge=500, le=5000, description="长文本阈值")


class FaultToleranceConfig(BaseModel):
    """节点容错和迁移配置"""

    enabled: bool = Field(default=True, description="启用节点容错和任务迁移")
    enable_degradation: bool = Field(default=True, description="启用节点降级策略")
    degradation_threshold: int = Field(default=3, ge=1, le=10, description="降级阈值（连续慢响应次数）")
    enable_task_migration: bool = Field(default=True, description="启用任务迁移")
    migration_delay: float = Field(default=5.0, ge=0.0, le=60.0, description="迁移延迟（秒）")
    recovery_check_interval: int = Field(default=60, ge=10, le=600, description="恢复检查间隔（秒）")


class DistributedConfig(BaseModel):
    """分布式配置"""

    enabled: bool = Field(default=False, description="启用分布式模式")
    mode: str = Field(default="master", description="运行模式: master | node")
    node_host: str = Field(default="0.0.0.0", description="节点监听地址")
    node_port: int = Field(default=8000, ge=1, le=65535, description="节点监听端口")
    node_max_concurrency: int = Field(default=4, ge=1, le=64, description="节点最大并发数")
    nodes: list[DistributedNodeConfig] = Field(default_factory=list, description="远程节点列表")
    master_max_concurrency: int = Field(default=8, ge=1, le=128, description="主控端最大并发数（分配到所有节点）")
    load_balance_strategy: str = Field(
        default="round_robin", description="负载均衡策略: round_robin | least_connections | best_response_time"
    )
    health_check_interval: int = Field(default=30, ge=5, le=300, description="健康检查间隔(秒)")
    health_check_timeout: int = Field(default=5, ge=1, le=30, description="健康检查超时(秒)")
    task_timeout: int = Field(default=300, ge=10, le=3600, description="任务超时(秒)")
    max_retries: int = Field(default=3, ge=0, le=10, description="任务最大重试次数")
    local_execution: bool = Field(default=True, description="主控端也执行任务（回退模式）")
    adaptive_load_balance: AdaptiveLoadBalanceConfig = Field(
        default_factory=AdaptiveLoadBalanceConfig, description="自适应负载均衡配置"
    )
    fault_tolerance: FaultToleranceConfig = Field(
        default_factory=FaultToleranceConfig, description="节点容错和迁移配置"
    )
