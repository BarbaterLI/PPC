"""User-defined extension framework for PPC_NEXT distributed TTS system.

This package provides extension points and loading mechanisms for users
to create custom extensions that integrate with the distributed system.
"""

from src.extensions.base import (
    ExecutorExtension,
    Extension,
    ExtensionMetadata,
    ExtensionPermission,
    ExtensionType,
    HealthCheckStrategy,
    LoadBalanceStrategy,
    MetadataProvider,
    MetricsExporter,
    SecurityLevel,
    SplitSegment,
    SplitStrategy,
    TaskSchedulingStrategy,
    TextPostprocessor,
    TextPreprocessor,
    ToolIntegration,
)
from src.extensions.loader import ExtensionLoader
from src.extensions.package import ExtensionManifest, ExtensionPackageManager
from src.extensions.sandbox import SandboxContext
from src.extensions.security import AuditAction, AuditEntry, AuditLogger, AuditResult

__all__ = [
    "Extension",
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
    "ExtensionLoader",
    "ExtensionManifest",
    "ExtensionPackageManager",
    "AuditLogger",
    "AuditAction",
    "AuditEntry",
    "AuditResult",
    "SandboxContext",
]
