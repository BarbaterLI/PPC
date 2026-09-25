"""扩展加载器- 发现和加载用户自定义扩展。

:class:`ExtensionLoader` 为门面，持有发现/安全扫描/IPC/进程管理/资源监控/
生命周期等职责组件（详见 :mod:`src.extensions.discovery` 等模块）。
"""

import importlib
import importlib.machinery
import importlib.util
import logging
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import yaml

from src.extensions._paths import (
    default_data_dir,
    default_user_extensions_dir,
    extensions_root,
)
from src.extensions.base import (
    AudioConverter,
    ExecutorExtension,
    Extension,
    ExtensionContext,
    ExtensionType,
    HealthCheckStrategy,
    LoadBalanceStrategy,
    MetadataProvider,
    MetricsExporter,
    SecurityLevel,
    SplitStrategy,
    TaskSchedulingStrategy,
    TextPostprocessor,
    TextPreprocessor,
    ToolIntegration,
    TTSBackend,
)
from src.extensions.discovery import (
    ExtensionDiscovery,
    discover_extensions,
)
from src.extensions.executable_adapter import ExecutableExtensionAdapter
from src.extensions.ipc import IPCClient, IPCServer
from src.extensions.ipc_bridge import IPCRequestHandler
from src.extensions.lifecycle import ExtensionLifecycle
from src.extensions.loader_support import (
    _check_requirements,
    _deep_merge,
    _resolve_entry_point,
)
from src.extensions.manifest import ExtensionManifest, read_manifest_from_dir
from src.extensions.process_manager import (
    ExecutableExtensionError,
    ExecutableProcessManager,
)
from src.extensions.resource_monitor import ResourceMonitor
from src.extensions.sandbox import SandboxConfig, SandboxContext
from src.extensions.scanner import (
    ExtensionSecurityScanner,
    SecurityScanResult,
)
from src.extensions.security import AuditLogger, AuditResult
from src.extensions.security_scan import (
    SecurityScanCoordinator,
    extract_declared_permissions,
)

logger = logging.getLogger(__name__)

REQUIRED_INTERFACES = {
    ExtensionType.LOAD_BALANCE_STRATEGY: LoadBalanceStrategy,
    ExtensionType.HEALTH_CHECK_STRATEGY: HealthCheckStrategy,
    ExtensionType.TASK_SCHEDULING_STRATEGY: TaskSchedulingStrategy,
    ExtensionType.METRICS_EXPORTER: MetricsExporter,
    ExtensionType.EXECUTOR: ExecutorExtension,
    ExtensionType.TOOL_INTEGRATION: ToolIntegration,
    ExtensionType.SPLIT_STRATEGY: SplitStrategy,
    ExtensionType.TEXT_PREPROCESSOR: TextPreprocessor,
    ExtensionType.TEXT_POSTPROCESSOR: TextPostprocessor,
    ExtensionType.METADATA_PROVIDER: MetadataProvider,
    ExtensionType.AUDIO_CONVERTER: AudioConverter,
    ExtensionType.TTS_BACKEND: TTSBackend,
}


class ExtensionLoader:
    """加载并管理用户自定义扩展"""

    def __init__(
        self,
        extension_dirs: list[Path] | None = None,
        security_scanner: ExtensionSecurityScanner | None = None,
        security_level: SecurityLevel = SecurityLevel.WARN,
        audit_logger: AuditLogger | None = None,
    ):
        self._extension_dirs = list(extension_dirs) if extension_dirs else []
        self._loaded_extensions: dict[str, Extension] = {}
        self._failed_extensions: list[dict[str, str]] = []
        self._extension_modules: dict[str, str] = {}
        self._security_scanner = security_scanner or ExtensionSecurityScanner()
        self._security_level = security_level
        self._audit_logger = audit_logger or AuditLogger()
        self._extensions_data_dir = default_data_dir()
        self._extensions_data_dir.mkdir(parents=True, exist_ok=True)

        self._discovery = ExtensionDiscovery()
        self._scan_coordinator = SecurityScanCoordinator(self._security_scanner)
        self._ipc_handler = IPCRequestHandler()
        self._process_manager = ExecutableProcessManager()
        self._resource_monitor = ResourceMonitor(self._process_manager)
        self._lifecycle = ExtensionLifecycle(self)

        builtin_dir = extensions_root()
        if builtin_dir.exists() and builtin_dir not in self._extension_dirs:
            self._extension_dirs.append(builtin_dir)

        default_ext_dir = default_user_extensions_dir()
        if default_ext_dir.exists() and default_ext_dir not in self._extension_dirs:
            self._extension_dirs.append(default_ext_dir)

    def get_extension_data_dir(self, name: str) -> Path:
        """返回扩展数据目录路径（首次访问时自动创建 cache/ 与 logs/ 子目录）。"""
        data_dir = self._extensions_data_dir / name
        data_dir.mkdir(parents=True, exist_ok=True)
        # 规范化子目录结构：cache/ 用于运行时缓存，logs/ 用于扩展日志
        (data_dir / "cache").mkdir(parents=True, exist_ok=True)
        (data_dir / "logs").mkdir(parents=True, exist_ok=True)
        return data_dir

    @staticmethod
    def _read_package_config_yml(package_path: Path) -> dict:
        """读取扩展包目录下的 ``config.yml``（扩展私有配置）。

        若 ``config.yml`` 不存在或解析失败（非字典类型、YAML 语法错误等），
        返回空字典并记录 warning。返回值始终为 dict。
        """
        config_file = package_path / "config.yml"
        if not config_file.is_file():
            return {}
        try:
            parsed = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError) as exc:
            logger.warning(
                f"Failed to parse config.yml in package {package_path.name}: {exc}"
            )
            return {}
        if not isinstance(parsed, dict):
            logger.warning(
                f"config.yml in package {package_path.name} is not a dict (got "
                f"{type(parsed).__name__}); ignoring"
            )
            return {}
        return parsed

    def add_extension_dir(self, directory: Path) -> None:
        if directory.exists() and directory.is_dir():
            self._extension_dirs.append(directory)
            logger.info(f"Extension directory added: {directory}")
        else:
            logger.warning(f"Extension directory does not exist: {directory}")

    async def load_all_extensions(self) -> dict[str, Extension]:
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

        logger.info(f"Extensions loaded: {len(self._loaded_extensions)}, failed: {len(self._failed_extensions)}")
        return self._loaded_extensions.copy()

    def _discover_extensions(self, directory: Path) -> list[dict]:
        return discover_extensions(directory)

    def _resolve_load_order(self, extensions: list[dict]) -> list[dict]:
        """解析扩展加载顺序（依赖拓扑排序，委托 :class:`ExtensionDiscovery`）。"""
        return self._discovery.resolve_load_order(extensions)

    def _perform_security_scan(self, file_path: Path) -> SecurityScanResult | None:
        """执行单文件安全扫描；应阻止加载时返回扫描结果，否则返回 None。"""
        return self._scan_coordinator.scan_file(file_path, self._security_level)

    async def _load_extension_from_file(self, file_path: Path) -> Extension | None:
        """从单个 .py 文件加载扩展。"""
        module_name = f"ppc_ext_{file_path.parent.name}_{file_path.stem}"

        def spec_creator(path: Path):
            return importlib.util.spec_from_file_location(module_name, path)

        def scan_scope() -> str | None:
            block_result = self._perform_security_scan(file_path)
            if block_result is not None:
                return f"Security scan blocked loading (risk_score={block_result.risk_score})"
            return None

        return await self._load_extension_common(
            source_path=file_path,
            source_label="extension",
            extension_dir=file_path.parent,
            module_name=module_name,
            spec_creator=spec_creator,
            scan_scope_fn=scan_scope,
        )

    async def _load_extension_from_package(self, package_path: Path) -> Extension | None:
        """从包目录加载扩展。"""
        module_name = f"ppc_ext_{package_path.parent.name}_{package_path.name}"

        manifest = read_manifest_from_dir(package_path)

        # 读取包内 config.yml（扩展私有配置），深度合并到 manifest 的内联 config 上：
        # 包内 config.yml 的值覆盖 manifest 内联 config（同名键）。
        # 合并结果写入 manifest.config，由 to_metadata() 透传到 ExtensionMetadata.config。
        if manifest is not None:
            inline_config = getattr(manifest, "config", None) or {}
            package_config = self._read_package_config_yml(package_path)
            if package_config:
                merged_config = _deep_merge(inline_config, package_config)
                manifest.config = merged_config  # type: ignore[attr-defined]
                logger.debug(
                    f"Loaded package config.yml for '{manifest.name}': "
                    f"inline={inline_config}, package={package_config}, merged={merged_config}"
                )

        # 根据 package_type 分派加载逻辑：
        # - executable 类型走 _load_executable_extension（启动 .exe 进程 + 适配器）
        # - python 类型（默认）走 _load_extension_common（import + entry_point 实例化）
        if manifest is not None and manifest.package_type == "executable":
            return await self._load_executable_extension(package_path, manifest)

        declared_perms = extract_declared_permissions(package_path, manifest)

        def spec_creator(path: Path):
            init_file = path / "__init__.py"
            return importlib.util.spec_from_file_location(
                module_name, init_file, submodule_search_locations=[str(path)]
            )

        def scan_scope() -> str | None:
            return self._scan_coordinator.scan_package(
                package_path, declared_perms, self._security_level
            )

        return await self._load_extension_common(
            source_path=package_path,
            source_label="extension package",
            extension_dir=package_path,
            module_name=module_name,
            spec_creator=spec_creator,
            scan_scope_fn=scan_scope,
            manifest=manifest,
        )

    async def _load_executable_extension(
        self, package_path: Path, manifest: ExtensionManifest
    ) -> Extension | None:
        """加载 executable 类型的扩展（Windows ``.exe`` 进程 + IPC 适配器）。

        流程：平台检查 → exe 路径解析 → SHA-256 完整性校验 → 沙箱上下文创建 →
        进程启动 → 存活验证 → 适配器包装 → 资源监控守护线程启动 → 注册。
        失败时记录失败原因并返回 None。
        """
        # 1. 平台检查：executable 类型仅支持 Windows
        if not self._process_manager.supported_platform():
            logger.warning(
                f"Executable extension '{manifest.name}' is only supported on "
                f"Windows, skipping"
            )
            return None

        # 2. exe 路径解析 + 3. SHA-256 完整性校验（在启动进程前验证）
        try:
            exe_path = self._process_manager.resolve_exe_path(package_path, manifest)
            self._process_manager.verify_sha256(exe_path, manifest)
        except ExecutableExtensionError as e:
            logger.error(str(e))
            self._record_failure(package_path.name, str(e))
            return None

        # 4. 创建 SandboxContext（package_type="executable" 触发
        #    STRICT 强制 + sandbox 目录限制）
        data_dir = self.get_extension_data_dir(manifest.name)
        sandbox_context = SandboxContext(SandboxConfig(
            extension_name=manifest.name,
            permissions=[],
            audit_logger=self._audit_logger,
            security_level=self._security_level,
            data_dir=data_dir,
            package_type="executable",
            allowed_hosts=manifest.allowed_hosts,
        ))

        # 5. 启动 exe 进程 + 6. 等待进程启动并验证仍在运行
        try:
            process = self._process_manager.spawn(exe_path, manifest.name)
            self._process_manager.verify_alive(process, manifest.name)
        except ExecutableExtensionError as e:
            logger.error(str(e))
            self._record_failure(package_path.name, str(e))
            return None

        # 7. 包装为 ExecutableExtensionAdapter（含 IPC 服务端/客户端）
        adapter = self._build_executable_adapter(
            package_path, manifest, sandbox_context, process, data_dir
        )

        # 8. 启动资源监控守护线程（若 manifest.resource_limits 非空）
        if manifest.resource_limits:
            threading.Thread(
                target=self._monitor_resources,
                args=(process, manifest, adapter),
                daemon=True,
            ).start()

        # 9. 注册并返回
        self._loaded_extensions[manifest.name] = adapter
        logger.info(f"Executable extension loaded: {manifest.name}")
        self._audit_logger.log_event(
            manifest.name, "EXTENSION_LOAD", manifest.name, AuditResult.SUCCESS
        )
        return adapter

    def _build_executable_adapter(
        self,
        package_path: Path,
        manifest: ExtensionManifest,
        sandbox_context: SandboxContext,
        process: subprocess.Popen,
        data_dir: Path,
    ) -> ExecutableExtensionAdapter:
        """启动 IPC 服务端/客户端并包装 executable 扩展适配器。"""
        ipc_pipe_name = f"ppc_ext_{manifest.name}"
        ipc_server: IPCServer | None = None
        ipc_client: IPCClient | None = None
        try:
            ipc_server = IPCServer(ipc_pipe_name)
            ipc_server.on_request(
                lambda action, data: self._handle_ipc_request(
                    manifest.name, sandbox_context, action, data
                )
            )
            ipc_server.start()
        except Exception as e:
            logger.warning(
                f"Failed to start IPC server for '{manifest.name}': {e} "
                f"(falling back to stub mode)"
            )
            ipc_server = None
        # IPCClient 连接是惰性的（首次 send 时建立），此处创建不会失败
        try:
            ipc_client = IPCClient(ipc_pipe_name)
        except Exception as e:
            logger.warning(
                f"Failed to create IPC client for '{manifest.name}': {e}"
            )
            ipc_client = None

        adapter = ExecutableExtensionAdapter(
            manifest=manifest,
            process=process,
            sandbox_context=sandbox_context,
            audit_logger=self._audit_logger,
            ipc_client=ipc_client,
            ipc_server=ipc_server,
        )

        # 设置扩展上下文（与 python 路径保持一致）
        context = ExtensionContext(
            loader=self,
            extension_dir=package_path,
            data_dir=data_dir,
            sandbox=sandbox_context,
        )
        adapter._context = context
        return adapter

    def _handle_ipc_request(
        self,
        ext_name: str,
        sandbox_context: SandboxContext,
        action: str,
        data: dict,
    ) -> dict:
        """处理来自可执行文件扩展的 IPC 反向请求（委托 :class:`IPCRequestHandler`）。"""
        return self._ipc_handler.handle(ext_name, sandbox_context, action, data)

    def _monitor_resources(
        self,
        process: subprocess.Popen,
        manifest: ExtensionManifest,
        adapter: ExecutableExtensionAdapter,
        poll_interval: float = 5.0,
    ) -> None:
        """资源监控守护线程入口（委托 :class:`ResourceMonitor`）。

        检查 CPU / 内存 / 超时，超限则终止进程并标记适配器失败。
        """
        self._resource_monitor.monitor(
            process, manifest, adapter, poll_interval
        )

    @staticmethod
    def _exec_module_spec(
        spec_creator: Callable[[Path], importlib.machinery.ModuleSpec | None],
        source_path: Path,
        module_name: str,
    ) -> "importlib.machinery.Module":
        """创建模块 spec、注册到 sys.modules 并执行。"""
        spec = spec_creator(source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec from {source_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _instantiate_extension(
        self, module, manifest: ExtensionManifest | None, source_name: str
    ) -> Extension | None:
        """解析入口点并实例化扩展。

        优先使用 manifest.entry_point（"module:ClassName"）实例化类；
        若 manifest 缺失或未声明 entry_point，则回退到旧的
        _find_extension_instance（按模块级变量扫描，已废弃）并 WARNING。
        """
        if manifest is not None and manifest.entry_point:
            ext_cls = _resolve_entry_point(manifest.entry_point, module)
            extension = ext_cls()
            if not isinstance(extension, Extension):
                raise ImportError(
                    f"entry_point '{manifest.entry_point}' resolved to "
                    f"{type(extension).__name__} which is not an Extension subclass"
                )
            return extension

        ext_display_name = (
            manifest.name if manifest is not None else source_name
        )
        logger.warning(
            f"Extension '{ext_display_name}' has no entry_point in manifest, "
            f"falling back to _find_extension_instance (deprecated)"
        )
        return self._find_extension_instance(module)

    def _build_extension_context(
        self,
        extension_name: str,
        permissions: list,
        extension_dir: Path,
        data_dir: Path | None = None,
    ) -> ExtensionContext:
        """构造扩展上下文（数据目录 + 沙箱 + ExtensionContext）。"""
        if data_dir is None:
            data_dir = self.get_extension_data_dir(extension_name)
        sandbox = SandboxContext(SandboxConfig(
            extension_name=extension_name,
            permissions=permissions,
            audit_logger=self._audit_logger,
            security_level=self._security_level,
            data_dir=data_dir,
        ))
        return ExtensionContext(
            loader=self,
            extension_dir=extension_dir,
            data_dir=data_dir,
            sandbox=sandbox,
        )

    async def _activate_extension(
        self,
        extension: Extension,
        extension_dir: Path,
        manifest: ExtensionManifest | None,
        module_name: str,
    ) -> None:
        """应用 manifest 元数据、注入上下文、触发加载钩子并注册扩展。"""
        if manifest is not None:
            try:
                extension.metadata = manifest.to_metadata()
            except Exception as meta_e:
                logger.warning(
                    f"Failed to apply manifest metadata to extension "
                    f"{extension.metadata.name}: {meta_e}"
                )
        ext_name = extension.metadata.name

        extension._context = self._build_extension_context(
            ext_name, extension.metadata.permissions, extension_dir
        )

        try:
            await extension.before_load(extension._context)
        except Exception as hook_e:
            logger.warning(f"Extension {ext_name} before_load hook error: {hook_e}")

        await extension.initialize()

        try:
            await extension.after_load(extension._context)
        except Exception as hook_e:
            logger.warning(f"Extension {ext_name} after_load hook error: {hook_e}")

        self._loaded_extensions[ext_name] = extension
        self._extension_modules[ext_name] = module_name
        logger.info(f"Extension loaded successfully: {ext_name}")
        self._audit_logger.log_event(
            ext_name, "EXTENSION_LOAD", ext_name, AuditResult.SUCCESS
        )

    async def _handle_load_error(
        self,
        extension: Extension | None,
        source_name: str,
        source_label: str,
        extension_dir: Path,
        module_name: str,
        error: Exception,
    ) -> None:
        """加载失败时的统一错误处理：on_error 钩子 + 审计 + 失败记录。"""
        if extension is not None:
            try:
                err_ctx = extension._context
                if err_ctx is None:
                    err_ctx = self._build_extension_context(
                        extension.metadata.name,
                        extension.metadata.permissions,
                        extension_dir,
                    )
                await extension.on_error(err_ctx, error)
            except Exception as hook_e:
                logger.warning(f"Extension {extension.metadata.name} on_error hook error: {hook_e}")
            self._audit_logger.log_event(
                extension.metadata.name, "EXTENSION_LOAD", extension.metadata.name,
                AuditResult.ERROR, {"error": str(error)},
            )
        else:
            self._audit_logger.log_event(
                source_name, "EXTENSION_LOAD", source_name, AuditResult.ERROR,
                {"error": str(error)},
            )
        self._record_failure(source_name, str(error))
        sys.modules.pop(module_name, None)
        logger.warning(f"Failed to load {source_label} {source_name}: {error}")

    async def _load_extension_common(
        self,
        source_path: Path,
        source_label: str,
        extension_dir: Path,
        module_name: str,
        spec_creator: Callable[[Path], importlib.machinery.ModuleSpec | None],
        scan_scope_fn: Callable[[], str | None],
        manifest: ExtensionManifest | None = None,
    ) -> Extension | None:
        """加载扩展的公共逻辑（文件/包共享）。

        参数:
            source_path: 扩展源路径（.py 文件路径或包目录）
            source_label: 日志标签，"extension" 或 "extension package"
            extension_dir: 扩展目录（文件父目录或包目录）
            module_name: 动态模块名
            spec_creator: 根据源路径创建 ModuleSpec 的回调
            scan_scope_fn: 执行安全扫描，返回 None（放行）或阻止消息字符串
            manifest: 包目录的清单（仅包加载流程提供），用于覆盖运行时扩展元数据
        """
        source_name = source_path.name
        extension: Extension | None = None

        try:
            # 1. 安全扫描
            blocking_msg = scan_scope_fn()
            if blocking_msg is not None:
                self._record_failure(source_name, blocking_msg)
                logger.error(f"{source_label.capitalize()} loading blocked by security scan: {source_name}")
                return None

            # 1.5 依赖检查：python 类型扩展加载前检测 requirements.txt 中
            # 声明的依赖是否已安装，未安装时记录 WARNING（不阻塞，后续 import
            # 失败会被 try/except 捕获）。在安全扫描之后、模块执行之前执行。
            if manifest is not None and manifest.package_type == "python":
                _check_requirements(extension_dir, source_name)

            # 2. 创建模块 spec 并执行
            module = self._exec_module_spec(spec_creator, source_path, module_name)

            # 3. 解析入口点并实例化扩展
            extension = self._instantiate_extension(module, manifest, source_name)
            if extension is None:
                self._record_failure(source_name, "No extension class found")
                self._audit_logger.log_event(
                    source_name, "EXTENSION_LOAD", source_name, AuditResult.ERROR,
                    {"error": "No extension class found"},
                )
                sys.modules.pop(module_name, None)
                return None

            if not self._validate_extension(extension):
                self._record_failure(source_name, "Interface validation failed")
                self._audit_logger.log_event(
                    source_name, "EXTENSION_LOAD", source_name, AuditResult.ERROR,
                    {"error": "Interface validation failed"},
                )
                sys.modules.pop(module_name, None)
                return None

            # 4. 元数据覆盖 / 上下文注入 / 加载钩子 / 注册
            await self._activate_extension(extension, extension_dir, manifest, module_name)
            return extension

        except Exception as e:
            await self._handle_load_error(
                extension, source_name, source_label, extension_dir, module_name, e
            )
            return None

    def _find_extension_instance(self, module) -> Extension | None:
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

    def _record_failure(self, filename: str, error: str) -> None:
        self._failed_extensions.append(
            {
                "filename": filename,
                "error": error,
            }
        )

    def get_loaded_extensions(self) -> dict[str, Extension]:
        return self._loaded_extensions.copy()

    def get_failed_extensions(self) -> list[dict[str, str]]:
        return self._failed_extensions.copy()

    def get_extensions_by_type(self, extension_type: ExtensionType) -> list[Extension]:
        return [ext for ext in self._loaded_extensions.values() if ext.metadata.extension_type == extension_type]

    def get_split_strategies(self) -> list[Extension]:
        """返回所有 SPLIT_STRATEGY 类型的扩展。"""
        return self.get_extensions_by_type(ExtensionType.SPLIT_STRATEGY)

    def get_text_preprocessors(self) -> list[Extension]:
        """返回所有 TEXT_PREPROCESSOR 类型的扩展。"""
        return self.get_extensions_by_type(ExtensionType.TEXT_PREPROCESSOR)

    def get_text_postprocessors(self) -> list[Extension]:
        """返回所有 TEXT_POSTPROCESSOR 类型的扩展。"""
        return self.get_extensions_by_type(ExtensionType.TEXT_POSTPROCESSOR)

    def get_metadata_providers(self) -> list[Extension]:
        """返回所有 METADATA_PROVIDER 类型的扩展。"""
        return self.get_extensions_by_type(ExtensionType.METADATA_PROVIDER)

    def get_extension(self, name: str) -> Extension | None:
        return self._loaded_extensions.get(name)

    def get_extension_module(self, name: str):
        """返回扩展的动态加载模块对象，未加载或可执行扩展返回 None。

        扩展以 ``ppc_ext_*`` 名称注册在 ``sys.modules`` 中（见
        :meth:`_exec_module_spec`），供后端 API 按需获取其子模块
        （如 ``fanqie.downloader_core``）。
        """
        module_name = self._extension_modules.get(name)
        if module_name is None:
            return None
        return sys.modules.get(module_name)

    def get_status(self) -> list[dict]:
        """返回所有已加载扩展的状态信息。

        返回:
            [{"name": str, "enabled": bool, "extension_type": str | None}]
        """
        status = []
        for name, ext in self._loaded_extensions.items():
            ext_type = ext.metadata.extension_type
            status.append({
                "name": name,
                "enabled": ext.enabled,
                "extension_type": ext_type.value if ext_type is not None else None,
            })
        return status

    async def reload(self, extension_name: str | None = None) -> dict:
        """重新加载扩展（指定名称时仅重载该扩展）。

        返回:
            重载结果统计: {"reloaded": int, "failed": int}
        """
        if extension_name is not None:
            await self.unload(extension_name)
        else:
            await self.clear()
        await self.load_all_extensions()
        return {
            "reloaded": len(self._loaded_extensions),
            "failed": len(self._failed_extensions),
        }

    async def enable(self, name: str) -> bool:
        """启用扩展（触发 before_enable / on_enable 钩子）。"""
        return await self._lifecycle.enable(name)
    async def disable(self, name: str) -> bool:
        """禁用扩展（触发 on_disable / after_disable 钩子）。"""
        return await self._lifecycle.disable(name)
    async def unload(self, name: str) -> bool:
        """卸载并移除指定扩展（清理资源、触发禁用钩子并从注册表移除）。"""
        return await self._lifecycle.unload(name)
    def get_last_scan_results(self) -> list[SecurityScanResult]:
        """获取最近一次加载的安全扫描结果。"""
        return self._scan_coordinator.last_results()

    def set_security_level(self, security_level: SecurityLevel) -> None:
        """设置安全级别。"""
        self._security_level = security_level
        logger.info(f"Extension security level set to: {security_level.value}")

    def get_security_level(self) -> SecurityLevel:
        """获取当前安全级别。"""
        return self._security_level

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

        # 清理扫描结果与缓存，保证后续重载能重新扫描和解析最新文件内容
        self._scan_coordinator.reset()

    # -------------------------------------------------------------------------
    # CLI 子命令自动注册
    # -------------------------------------------------------------------------

    def has_cli(self, name: str) -> bool:
        """判断指定扩展是否实现了 register_cli 接口。"""
        ext = self._loaded_extensions.get(name)
        if ext is None:
            return False
        return callable(getattr(ext, "register_cli", None))

    def get_cli_extensions(self) -> list[Extension]:
        """返回所有实现了 register_cli 接口的扩展。"""
        return [ext for ext in self._loaded_extensions.values() if callable(getattr(ext, "register_cli", None))]

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


# ---------------------------------------------------------------------------
# 配置接线（config.extensions 段）
# ---------------------------------------------------------------------------


def configured_extension_dirs() -> list[Path]:
    """按 ``config.extensions.extension_dirs`` 解析扩展目录（相对路径基于项目根）。

    配置不可用（如尚未初始化）时返回空列表，交由 loader 默认目录逻辑处理。
    """
    try:
        from src.config import get_default_manager

        dirs = get_default_manager().get_config().extensions.extension_dirs
    except Exception:
        return []

    from src.utils.paths import get_project_root

    root = get_project_root()
    resolved: list[Path] = []
    for d in dirs or []:
        p = Path(d).expanduser()
        resolved.append(p if p.is_absolute() else root / p)
    return resolved


def extensions_enabled_by_config() -> bool:
    """``config.extensions.enabled`` 总开关（读取失败时默认启用）。"""
    try:
        from src.config import get_default_manager

        return bool(get_default_manager().get_config().extensions.enabled)
    except Exception:
        return True


def extension_auto_load_by_config() -> bool:
    """``config.extensions.auto_load``：是否在 CLI 启动路径自动加载扩展。"""
    try:
        from src.config import get_default_manager

        return bool(get_default_manager().get_config().extensions.auto_load)
    except Exception:
        return True


def build_loader_from_config() -> ExtensionLoader | None:
    """依据当前配置构造 loader；``enabled=False`` 时返回 None 表示扩展系统关闭。"""
    if not extensions_enabled_by_config():
        return None
    return ExtensionLoader(extension_dirs=configured_extension_dirs())
