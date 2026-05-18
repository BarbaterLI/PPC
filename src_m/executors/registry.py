"""Executor Registry - Plugin registration and discovery for executors.

This module provides a registry system for executors that allows:
- Dynamic registration and unregistration of executors
- Metadata management (name, version, description, capability tags)
- Factory method for creating executor instances by name
- Auto-discovery of executors from module scanning
"""

import importlib
import importlib.util
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from src_m.executors.base import BaseExecutor

logger = logging.getLogger(__name__)


@dataclass
class ExecutorMetadata:
    """Metadata for a registered executor"""
    name: str
    executor_class: Type[BaseExecutor]
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None


class ExecutorRegistry:
    """Registry for executor plugins.
    
    Provides centralized management of executor registration, discovery,
    and instantiation with metadata tracking.
    """

    def __init__(self):
        self._executors: Dict[str, ExecutorMetadata] = {}
        self._factories: Dict[str, Callable[..., BaseExecutor]] = {}
        self._discovery_paths: List[Path] = []
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        executor_class: Type[BaseExecutor],
        version: str = "1.0.0",
        description: str = "",
        author: str = "",
        tags: Optional[List[str]] = None,
        config_schema: Optional[Dict[str, Any]] = None,
        factory: Optional[Callable[..., BaseExecutor]] = None,
        force: bool = False,
    ) -> None:
        """Register an executor with the registry."""
        if not issubclass(executor_class, BaseExecutor):
            raise TypeError(
                f"Executor class must be a subclass of BaseExecutor, got {executor_class}"
            )

        with self._lock:
            if name in self._executors and not force:
                raise ValueError(f"Executor '{name}' already registered. Use force=True to overwrite.")

            self._executors[name] = ExecutorMetadata(
                name=name,
                executor_class=executor_class,
                version=version,
                description=description,
                author=author,
                tags=tags or [],
                config_schema=config_schema,
            )

            if factory is not None:
                self._factories[name] = factory

            logger.info(f"Executor registered: {name} (v{version})")

    def unregister(self, name: str) -> bool:
        """Unregister an executor from the registry."""
        with self._lock:
            if name in self._executors:
                del self._executors[name]
                self._factories.pop(name, None)
                logger.info(f"Executor unregistered: {name}")
                return True
            return False

    def get(self, name: str) -> Optional[ExecutorMetadata]:
        """Get executor metadata by name."""
        return self._executors.get(name)

    def get_all(self) -> Dict[str, ExecutorMetadata]:
        """Get all registered executors."""
        return self._executors.copy()

    def get_by_tag(self, tag: str) -> List[ExecutorMetadata]:
        """Get executors that have a specific capability tag."""
        return [
            meta for meta in self._executors.values()
            if tag in meta.tags
        ]

    def create(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[BaseExecutor]:
        """Create an executor instance by name."""
        with self._lock:
            metadata = self._executors.get(name)
            if metadata is None:
                logger.error(f"Executor not found: {name}")
                return None

            if name in self._factories:
                try:
                    executor = self._factories[name](*args, **kwargs)
                    logger.info(f"Executor created via factory: {name}")
                    return executor
                except Exception as e:
                    logger.error(f"Factory failed for executor '{name}': {e}")
                    return None

            try:
                executor = metadata.executor_class(*args, **kwargs)
                logger.info(f"Executor created: {name}")
                return executor
            except Exception as e:
                logger.error(f"Failed to create executor '{name}': {e}")
                return None

    def has(self, name: str) -> bool:
        """Check if an executor is registered."""
        return name in self._executors

    def add_discovery_path(self, path: Path) -> None:
        """Add a directory path for auto-discovery scanning."""
        if path.exists() and path.is_dir():
            self._discovery_paths.append(path)
            logger.info(f"Discovery path added: {path}")
        else:
            logger.warning(f"Discovery path does not exist: {path}")

    def discover(self) -> int:
        """Scan registered discovery paths for executor modules."""
        discovered_count = 0

        for path in self._discovery_paths:
            for py_file in path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                if self._try_discover_executor(py_file):
                    discovered_count += 1

        logger.info(f"Executor discovery complete: {discovered_count} found")
        return discovered_count

    def _try_discover_executor(self, file_path: Path) -> bool:
        """Try to discover and register an executor from a file."""
        try:
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return False

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            discovered = False
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseExecutor)
                    and attr is not BaseExecutor
                    and not attr.__name__.startswith("_")
                ):
                    name = getattr(attr, "executor_name", attr.__name__)
                    if not self.has(name):
                        self.register(
                            name=name,
                            executor_class=attr,
                            version=getattr(attr, "executor_version", "1.0.0"),
                            description=getattr(attr, "executor_description", ""),
                            author=getattr(attr, "executor_author", ""),
                            tags=getattr(attr, "executor_tags", []),
                        )
                        discovered = True

            return discovered

        except Exception as e:
            logger.debug(f"Failed to discover executor in {file_path}: {e}")

        return False

    def clear(self) -> None:
        """Clear all registered executors and discovery paths."""
        with self._lock:
            self._executors.clear()
            self._factories.clear()
            self._discovery_paths.clear()
            logger.info("Executor registry cleared")


"""Global registry instance for executors."""
registry = ExecutorRegistry()


def register_executor(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    author: str = "",
    tags: Optional[List[str]] = None,
    config_schema: Optional[Dict[str, Any]] = None,
    factory: Optional[Callable[..., BaseExecutor]] = None,
):
    """Decorator to register an executor class.
    
    Usage:
        @register_executor("my_executor", tags=["tts", "custom"])
        class MyExecutor(BaseExecutor):
            ...
    """
    def decorator(cls: Type[BaseExecutor]) -> Type[BaseExecutor]:
        registry.register(
            name=name,
            executor_class=cls,
            version=version,
            description=description,
            author=author,
            tags=tags,
            config_schema=config_schema,
            factory=factory,
        )
        return cls

    return decorator
