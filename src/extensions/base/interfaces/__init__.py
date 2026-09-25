"""扩展策略接口定义。

按语义拆分为四个分组模块：

* :mod:`src.extensions.base.interfaces.distributed` — 分布式调度策略
* :mod:`src.extensions.base.interfaces.text` — 文本处理策略
* :mod:`src.extensions.base.interfaces.audio` — 音频处理接口
* :mod:`src.extensions.base.interfaces.tools` — 工具集成与执行器接口

本包保持 ``from src.extensions.base.interfaces import X`` 的导入面不变。
"""

from src.extensions.base.interfaces.audio import AudioConverter, TTSBackend
from src.extensions.base.interfaces.distributed import (
    HealthCheckStrategy,
    LoadBalanceStrategy,
    MetricsExporter,
    TaskSchedulingStrategy,
)
from src.extensions.base.interfaces.text import (
    MetadataProvider,
    SplitSegment,
    SplitStrategy,
    TextPostprocessor,
    TextPreprocessor,
)
from src.extensions.base.interfaces.tools import ExecutorExtension, ToolIntegration

__all__ = [
    "LoadBalanceStrategy",
    "HealthCheckStrategy",
    "TaskSchedulingStrategy",
    "MetricsExporter",
    "ToolIntegration",
    "ExecutorExtension",
    "SplitSegment",
    "SplitStrategy",
    "TextPreprocessor",
    "TextPostprocessor",
    "MetadataProvider",
    "AudioConverter",
    "TTSBackend",
]
