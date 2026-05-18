"""User-defined extension framework for PPC9 distributed TTS system.

This package provides extension points and loading mechanisms for users
to create custom extensions that integrate with the distributed system.
"""

from src_m.extensions.base import (
    Extension,
    ExtensionMetadata,
    ExtensionType,
    LoadBalanceStrategy,
    HealthCheckStrategy,
    TaskSchedulingStrategy,
    MetricsExporter,
    ToolIntegration,
    ExecutorExtension,
)
from src_m.extensions.loader import ExtensionLoader
from src_m.extensions.package import ExtensionManifest, ExtensionPackageManager

__all__ = [
    "Extension",
    "ExtensionMetadata",
    "ExtensionType",
    "LoadBalanceStrategy",
    "HealthCheckStrategy",
    "TaskSchedulingStrategy",
    "MetricsExporter",
    "ToolIntegration",
    "ExecutorExtension",
    "ExtensionLoader",
    "ExtensionManifest",
    "ExtensionPackageManager",
]
