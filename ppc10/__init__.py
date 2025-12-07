"""PPC-1.0 - 高性能文本转语音处理工具"""

from .config import OptimizedAppConfig, ConfigShareManager
from .core import (
    logger,
    OptimizedLogHandler,
    WorkerSignals,
    OptimizedFileProcessor,
    OptimizedTTSTask,
    OptimizedHistoryManager,
    OptimizedDynamicTimeout,
    OptimizedConnectivityMonitor,
    OptimizedTTSOrchestrator
)
from .chapter import (
    ChapterPattern,
    ChapterPreset,
    ChapterRule,
    SmartChapterDetector,
    AdvancedChapterProcessor
)
from .performance import ObjectPool, MemoryMonitor, memory_efficient, memory_monitor

__version__ = "1.0"
__author__ = "PPC-1.0 Team"

__all__ = [
    # Config
    "OptimizedAppConfig",
    "ConfigShareManager",
    
    # Core
    "logger",
    "OptimizedLogHandler",
    "WorkerSignals",
    "OptimizedFileProcessor",
    "OptimizedTTSTask",
    "OptimizedHistoryManager",
    "OptimizedDynamicTimeout",
    "OptimizedConnectivityMonitor",
    "OptimizedTTSOrchestrator",
    
    # Chapter
    "ChapterPattern",
    "ChapterPreset",
    "ChapterRule",
    "SmartChapterDetector",
    "AdvancedChapterProcessor",
    
    # Performance
    "ObjectPool",
    "MemoryMonitor",
    "memory_efficient",
    "memory_monitor"
]
