"""Distributed TTS cluster module.

Provides node pool, processing units, and adapters for building a
multi-node TTS cluster. The infrastructure/ adapter layer has been
merged into this package.
"""

from src.distributed.executor_adapter import (
    DistributedTTSExecutor,
    HealthCheckConfig,
    create_default_config,
)
from src.distributed.node_adapter import (
    NodeClient,
    NodeClientConfig,
    NodeProtocol,
    TTSNode,
)
from src.distributed.node_pool import (
    NodeInfo,
    NodePool,
    NodeStatus,
)
from src.distributed.processing_unit import (
    ConvertRequest,
    ConvertResult,
    MasterHttpServer,
    MasterUnit,
    ProcessingUnit,
    UnitRole,
    WorkerUnit,
    make_processing_unit,
)

__all__ = [
    "NodeInfo",
    "NodePool",
    "NodeStatus",
    "TTSNode",
    "NodeClient",
    "NodeClientConfig",
    "NodeProtocol",
    "DistributedTTSExecutor",
    "HealthCheckConfig",
    "create_default_config",
    "UnitRole",
    "ConvertRequest",
    "ConvertResult",
    "ProcessingUnit",
    "WorkerUnit",
    "MasterUnit",
    "MasterHttpServer",
    "make_processing_unit",
]
