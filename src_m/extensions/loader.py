"""扩展加载器- 发现和加载用户自定义扩展"""

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

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
    PipelineStepExtension,
)

logger = logging.getLogger(__name__)

REQUIRED_INTERFACES = {
    ExtensionType.LOAD_BALANCE_STRATEGY: LoadBalanceStrategy,
    ExtensionType.HEALTH_CHECK_STRATEGY: HealthCheckStrategy,
    ExtensionType.TASK_SCHEDULING_STRATEGY: TaskSchedulingStrategy,
    ExtensionType.METRICS_EXPORTER: MetricsExporter,
    ExtensionType.EXECUTOR: ExecutorExtension,
    ExtensionType.TOOL_INTEGRATION: ToolIntegration,
    ExtensionType.PIPELINE_STEP: PipelineStepExtension,
}


class _PipelineStepAdapter:
    """将PipelineStepExtension 适配到PipelineStepExecutor 接口"""

    def __init__(self, step_ext: PipelineStepExtension):
        self._step_ext = step_ext

    async def execute(self, params, inputs):
        return await self._step_ext.execute(params, inputs)

    def get_name(self) -> str:
        return self._step_ext.get_step_name()

    def get_input_type(self) -> str:
        return self._step_ext.get_input_type()

    def get_output_type(self) -> str:
        return self._step_ext.get_output_type()


class ExtensionLoader:
    """加载并管理用户自定义扩展"""

    def __init__(self, extension_dirs: Optional[List[Path]] = None):
        self._extension_dirs = list(extension_dirs) if extension_dirs else []
        self._loaded_extensions: Dict[str, Extension] = {}
        self._failed_extensions: List[Dict[str, str]] = []
        self._extension_modules: Dict[str, str] = {}

        builtin_dir = Path(__file__).parent
        if builtin_dir.exists() and builtin_dir not in self._extension_dirs:
            self._extension_dirs.append(builtin_dir)

        default_ext_dir = Path(__file__).parent.parent.parent / "extensions"
        if default_ext_dir.exists() and default_ext_dir not in self._extension_dirs:
            self._extension_dirs.append(default_ext_dir)

    def add_extension_dir(self, directory: Path) -> None:
        if directory.exists() and directory.is_dir():
            self._extension_dirs.append(directory)
            logger.info(f"Extension directory added: {directory}")
        else:
            logger.warning(f"Extension directory does not exist: {directory}")

    async def load_all_extensions(self) -> Dict[str, Extension]:
        all_extensions = []
        for directory in self._extension_dirs:
            exts = self._discover_extensions(directory)
            all_extensions.extend(exts)

        sorted_exts = self._resolve_load_order(all_extensions)

        for ext_info in sorted_exts:
            if ext_info["type"] == "file":
                await self._load_extension_from_file(ext_info["path"])
            elif ext_info["type"] == "package":
                await self._load_extension_from_package(ext_info["path"])

        logger.info(
            f"Extensions loaded: {len(self._loaded_extensions)}, "
            f"failed: {len(self._failed_extensions)}"
        )
        return self._loaded_extensions.copy()

    def _discover_extensions(self, directory: Path) -> List[Dict]:
        discovered = []
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            discovered.append({"type": "file", "path": py_file})

        for subdir in directory.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("_"):
                if (subdir / "__init__.py").exists():
                    discovered.append({"type": "package", "path": subdir})

        return discovered

    def _resolve_load_order(self, extensions: List[Dict]) -> List[Dict]:
        return extensions

    async def _load_extensions_from_dir(self, directory: Path) -> None:
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            await self._load_extension_from_file(py_file)

        for subdir in directory.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("_"):
                if (subdir / "__init__.py").exists():
                    await self._load_extension_from_package(subdir)

    async def _load_extension_from_file(self, file_path: Path) -> None:
        parent_name = file_path.parent.name
        module_name = f"ppc10_ext_{parent_name}_{file_path.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load spec from {file_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            extension = self._find_extension_instance(module)
            if extension is not None:
                if self._validate_extension(extension):
                    await extension.initialize()
                    self._loaded_extensions[extension.metadata.name] = extension
                    self._extension_modules[extension.metadata.name] = module_name
                    self._register_pipeline_steps(extension)
                    logger.info(f"Extension loaded successfully: {extension.metadata.name}")
                else:
                    self._record_failure(file_path.name, "Interface validation failed")
                    sys.modules.pop(module_name, None)
            else:
                self._record_failure(file_path.name, "No extension class found")
                sys.modules.pop(module_name, None)

        except Exception as e:
            self._record_failure(file_path.name, str(e))
            sys.modules.pop(module_name, None)
            logger.warning(f"Failed to load extension {file_path.name}: {e}")

    async def _load_extension_from_package(self, package_path: Path) -> None:
        parent_name = package_path.parent.name
        module_name = f"ppc10_ext_{parent_name}_{package_path.name}"

        try:
            init_file = package_path / "__init__.py"
            spec = importlib.util.spec_from_file_location(module_name, init_file, submodule_search_locations=[str(package_path)])
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load spec from {package_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            extension = self._find_extension_instance(module)
            if extension is not None:
                if self._validate_extension(extension):
                    await extension.initialize()
                    self._loaded_extensions[extension.metadata.name] = extension
                    self._extension_modules[extension.metadata.name] = module_name
                    self._register_pipeline_steps(extension)
                    logger.info(f"Extension loaded successfully: {extension.metadata.name}")
                else:
                    self._record_failure(package_path.name, "Interface validation failed")
                    sys.modules.pop(module_name, None)
            else:
                self._record_failure(package_path.name, "No extension class found")
                sys.modules.pop(module_name, None)

        except Exception as e:
            self._record_failure(package_path.name, str(e))
            sys.modules.pop(module_name, None)
            logger.warning(f"Failed to load extension package {package_path.name}: {e}")

    def _find_extension_instance(self, module) -> Optional[Extension]:
        if hasattr(module, "extension") and isinstance(module.extension, Extension):
            return module.extension

        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            attr = getattr(module, attr_name)
            if isinstance(attr, Extension):
                return attr
        return None

    def _validate_extension(self, extension: Extension) -> bool:
        if extension.metadata.extension_type is None:
            return True

        required_interface = REQUIRED_INTERFACES.get(extension.metadata.extension_type)
        if required_interface is None:
            return True

        return isinstance(extension, required_interface)

    def _register_pipeline_steps(self, extension: Extension) -> None:
        """将扩展提供的管道步骤注册到StepRegistry"""
        try:
            from src_m.pipeline.registry import StepRegistry

            registry = StepRegistry()

            if isinstance(extension, PipelineStepExtension):
                adapter = _PipelineStepAdapter(extension)
                registry.register(adapter)
                logger.info(
                    f"Registered pipeline step '{adapter.get_name()}' from extension '{extension.metadata.name}'"
                )

            for step in extension.get_pipeline_steps():
                adapter = _PipelineStepAdapter(step)
                registry.register(adapter)
                logger.info(
                    f"Registered pipeline step '{adapter.get_name()}' from extension '{extension.metadata.name}'"
                )
        except Exception as e:
            logger.warning(
                f"Failed to register pipeline steps for extension '{extension.metadata.name}': {e}"
            )

    def _record_failure(self, filename: str, error: str) -> None:
        self._failed_extensions.append({
            "filename": filename,
            "error": error,
        })

    def get_loaded_extensions(self) -> Dict[str, Extension]:
        return self._loaded_extensions.copy()

    def get_failed_extensions(self) -> List[Dict[str, str]]:
        return self._failed_extensions.copy()

    def get_extensions_by_type(self, extension_type: ExtensionType) -> List[Extension]:
        return [
            ext for ext in self._loaded_extensions.values()
            if ext.metadata.extension_type == extension_type
        ]

    def get_extension(self, name: str) -> Optional[Extension]:
        return self._loaded_extensions.get(name)

    async def enable(self, name: str) -> bool:
        ext = self._loaded_extensions.get(name)
        if ext is None:
            return False
        if not hasattr(ext, '_enabled'):
            ext._enabled = True
        if ext._enabled:
            return True
        try:
            ext.on_enable()
            ext._enabled = True
            logger.info(f"Extension enabled: {name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to enable extension {name}: {e}")
            return False

    async def disable(self, name: str) -> bool:
        ext = self._loaded_extensions.get(name)
        if ext is None:
            return False
        if not hasattr(ext, '_enabled') or not ext._enabled:
            return True
        try:
            ext.on_disable()
            ext._enabled = False
            logger.info(f"Extension disabled: {name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to disable extension {name}: {e}")
            return False

    async def clear(self) -> None:
        for ext in self._loaded_extensions.values():
            try:
                await ext.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up extension {ext.metadata.name}: {e}")

        for module_name in self._extension_modules.values():
            sys.modules.pop(module_name, None)

        self._loaded_extensions.clear()
        self._failed_extensions.clear()
        self._extension_modules.clear()

    # -------------------------------------------------------------------------
    # CLI 子命令自动注册
    # -------------------------------------------------------------------------

    def has_cli(self, name: str) -> bool:
        """判断指定扩展是否实现了 register_cli 接口。"""
        ext = self._loaded_extensions.get(name)
        if ext is None:
            return False
        return callable(getattr(ext, "register_cli", None))

    def get_cli_extensions(self) -> List[Extension]:
        """返回所有实现了 register_cli 接口的扩展。"""
        return [
            ext for ext in self._loaded_extensions.values()
            if callable(getattr(ext, "register_cli", None))
        ]

    def register_cli_for(self, name: str, app) -> bool:
        """为指定扩展调用 register_cli(app) 注册 CLI 子命令。

        返回 True 表示成功注册，False 表示扩展不存在或未实现 register_cli。
        """
        ext = self._loaded_extensions.get(name)
        if ext is None:
            return False
        register_fn = getattr(ext, "register_cli", None)
        if not callable(register_fn):
            return False
        register_fn(app)
        return True
