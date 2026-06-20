"""User-defined extension framework for PPC10 distributed TTS system.

This package provides extension points and loading mechanisms for users
to create custom extensions that integrate with the distributed system.
"""

from src.extensions.base import (
    ExecutorExtension,
    Extension,
    ExtensionMetadata,
    ExtensionType,
    HealthCheckStrategy,
    LoadBalanceStrategy,
    MetricsExporter,
    TaskSchedulingStrategy,
    ToolIntegration,
)
from src.extensions.loader import ExtensionLoader
from src.extensions.package import ExtensionManifest, ExtensionPackageManager

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
