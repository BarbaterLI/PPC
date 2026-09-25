"""用户自定义扩展的扩展接口。

本模块定义了用户扩展必须实现的核心接口，以便与分布式 TTS 系统集成。

该包由 ``core``、``enums``、``interfaces`` 三个子模块聚合而成，对外保持
``from src.extensions.base import X`` 的向后兼容导入。
"""

from src.extensions.base.core import (
    Extension,
    ExtensionContext,
    ExtensionMetadata,
)
from src.extensions.base.enums import (
    ExtensionPermission,
    ExtensionType,
    SecurityLevel,
)
from src.extensions.base.interfaces import (
    AudioConverter,
    ExecutorExtension,
    HealthCheckStrategy,
    LoadBalanceStrategy,
    MetadataProvider,
    MetricsExporter,
    SplitSegment,
    SplitStrategy,
    TaskSchedulingStrategy,
    TextPostprocessor,
    TextPreprocessor,
    ToolIntegration,
    TTSBackend,
)

__all__ = [
    "Extension",
    "ExtensionContext",
    "ExtensionMetadata",
    "ExtensionType",
    "ExtensionPermission",
    "SecurityLevel",
    "LoadBalanceStrategy",
    "HealthCheckStrategy",
    "TaskSchedulingStrategy",
    "MetricsExporter",
    "ToolIntegration",
    "ExecutorExtension",
    "SplitStrategy",
    "SplitSegment",
    "TextPreprocessor",
    "TextPostprocessor",
    "MetadataProvider",
    "AudioConverter",
    "TTSBackend",
]
