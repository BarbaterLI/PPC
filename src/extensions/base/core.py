"""扩展核心定义。

包含基础扩展类、扩展上下文与扩展元数据。
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.extensions._paths import default_data_dir
from src.extensions.base.enums import ExtensionPermission, ExtensionType
from src.extensions.base.hooks import (
    ExtensionConfigMixin,
    ExtensionEnableMixin,
    ExtensionLifecycleMixin,
)

if TYPE_CHECKING:
    from src.extensions.loader import ExtensionLoader

logger = logging.getLogger(__name__)


@dataclass
class ExtensionContext:
    """扩展上下文对象，在钩子调用时传递给扩展。"""

    loader: "ExtensionLoader"
    extension_dir: Path
    data_dir: Path
    sandbox: Any = None


@dataclass(frozen=True)
class MetadataIdentity:
    """扩展身份分组（名称/版本/描述/作者/类型/标签）。"""

    name: str
    version: str
    description: str
    author: str
    extension_type: ExtensionType | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class MetadataConfiguration:
    """扩展配置分组（运行时配置/配置模式/依赖）。"""

    config: dict[str, Any]
    config_schema: Any
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class MetadataSecurity:
    """扩展安全分组（权限/网络主机白名单）。"""

    permissions: tuple[ExtensionPermission, ...]
    allowed_hosts: tuple[str, ...]


@dataclass(frozen=True)
class MetadataPackaging:
    """扩展打包分组（包类型/入口点/依赖清单/资源限制/完整性哈希）。"""

    package_type: str
    entry_point: str
    requirements: tuple[str, ...]
    resource_limits: dict[str, Any]
    sha256: str


@dataclass
class ExtensionMetadata:
    """扩展元数据。

    扁平字段保持扩展作者 API 不变；同时提供四个冻结的嵌套分组视图
    （:attr:`identity` / :attr:`configuration` / :attr:`security` /
    :attr:`packaging`）便于按语义整体访问。
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    extension_type: ExtensionType | None = None
    tags: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    config_schema: Any | None = None
    permissions: list[ExtensionPermission] = field(default_factory=list)
    package_type: str = "python"
    entry_point: str = ""
    requirements: list[str] = field(default_factory=list)
    allowed_hosts: list[str] = field(default_factory=list)
    resource_limits: dict[str, Any] = field(default_factory=dict)
    sha256: str = ""

    @property
    def identity(self) -> MetadataIdentity:
        """身份分组视图。"""
        return MetadataIdentity(
            name=self.name,
            version=self.version,
            description=self.description,
            author=self.author,
            extension_type=self.extension_type,
            tags=tuple(self.tags),
        )

    @property
    def configuration(self) -> MetadataConfiguration:
        """配置分组视图。"""
        return MetadataConfiguration(
            config=dict(self.config),
            config_schema=self.config_schema,
            dependencies=tuple(self.dependencies),
        )

    @property
    def security(self) -> MetadataSecurity:
        """安全分组视图。"""
        return MetadataSecurity(
            permissions=tuple(self.permissions),
            allowed_hosts=tuple(self.allowed_hosts),
        )

    @property
    def packaging(self) -> MetadataPackaging:
        """打包分组视图。"""
        return MetadataPackaging(
            package_type=self.package_type,
            entry_point=self.entry_point,
            requirements=tuple(self.requirements),
            resource_limits=dict(self.resource_limits),
            sha256=self.sha256,
        )


class Extension(ExtensionLifecycleMixin, ExtensionEnableMixin, ExtensionConfigMixin, ABC):
    """基础扩展类。

    钩子方法（before_load / after_load / on_error / on_enable 等）以可选
    Mixin 形式提供默认空实现，扩展作者只需实现 :meth:`initialize` 与
    :meth:`cleanup` 两个抽象方法。
    """

    def __init__(self, metadata: ExtensionMetadata):
        self.metadata = metadata
        self._initialized = False
        self._enabled = True
        self._context: ExtensionContext | None = None

    @abstractmethod
    async def initialize(self) -> None:
        """初始化扩展。"""
        self._initialized = True

    @abstractmethod
    async def cleanup(self) -> None:
        """清理扩展资源。"""
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_extension_data_dir(self) -> Path:
        """返回扩展专属数据目录路径（extensions_data/<name>/）。

        返回:
            扩展数据目录的 Path 对象
        """
        data_dir = default_data_dir() / self.metadata.name
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def load_config(self) -> dict:
        """从扩展数据目录加载 config.yml（经由沙箱验证 FILESYSTEM_READ 权限）。

        若 ``config.yml`` 不存在但旧 ``config.json`` 存在，自动迁移为 YAML 格式：
        读取 JSON 内容、写入 ``config.yml``、删除旧 ``config.json``。
        迁移过程为 best-effort，写入或删除失败时仅记录 warning，仍返回原始数据。

        返回:
            配置字典，如果文件不存在则返回空字典
        """
        data_dir = self.get_extension_data_dir()
        config_yml = data_dir / "config.yml"
        config_json = data_dir / "config.json"

        # 1. 优先读取 config.yml
        if config_yml.exists():
            return self._read_config_file(config_yml, use_yaml=True)

        # 2. config.yml 不存在但旧 config.json 存在：读取并迁移
        if config_json.exists():
            data = self._read_config_file(config_json, use_yaml=False)
            if data:
                self._migrate_json_to_yml(config_json, config_yml, data)
            return data

        # 3. 两者都不存在
        return {}

    def _read_config_file(self, path: Path, *, use_yaml: bool) -> dict:
        """读取配置文件并解析为字典（经由沙箱验证 FILESYSTEM_READ 权限）。

        沙箱未初始化（测试或非加载流程调用）时回退到直接读取。

        参数:
            path: 配置文件路径
            use_yaml: True 使用 yaml.safe_load 解析；False 使用 json.loads 解析

        返回:
            解析后的字典；解析失败或文件不存在返回空字典
        """
        if self._context is None or self._context.sandbox is None:
            try:
                with open(path, encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Extension {self.metadata.name} failed to load config: {e}")
                return {}
        else:
            try:
                content = self._context.sandbox.safe_read_file(path)
            except FileNotFoundError:
                # 文件不存在（在 exists() 检查后被删除）返回默认空字典
                return {}
            except Exception as e:
                # 包含沙箱拒绝（PermissionError）等，记录警告而非静默吞掉
                logger.warning(f"Extension {self.metadata.name} failed to load config: {e}")
                return {}
        try:
            if use_yaml:
                parsed = yaml.safe_load(content)
                return parsed if isinstance(parsed, dict) else {}
            return json.loads(content) if content else {}
        except Exception as e:
            logger.warning(f"Extension {self.metadata.name} failed to parse config: {e}")
            return {}

    def _migrate_json_to_yml(
        self, config_json: Path, config_yml: Path, data: dict
    ) -> None:
        """将旧 ``config.json`` 迁移为 ``config.yml``（best-effort）。

        写入 ``config.yml`` 后删除旧 ``config.json``；任一步骤失败时仅记录
        warning，不抛出异常，调用方仍可使用已读取的数据。

        参数:
            config_json: 旧 JSON 配置文件路径
            config_yml: 目标 YAML 配置文件路径
            data: 已读取的配置字典
        """
        try:
            yaml_content = yaml.safe_dump(
                data, default_flow_style=False, sort_keys=False, allow_unicode=True
            )
            if self._context is None or self._context.sandbox is None:
                with open(config_yml, "w", encoding="utf-8") as f:
                    f.write(yaml_content)
                config_json.unlink(missing_ok=True)
            else:
                self._context.sandbox.safe_write_file(config_yml, yaml_content)
                self._context.sandbox.safe_delete_file(config_json)
            logger.info(
                f"Extension {self.metadata.name}: Migrated config.json -> config.yml"
            )
        except Exception as e:
            logger.warning(
                f"Extension {self.metadata.name}: "
                f"config.json -> config.yml migration failed: {e}"
            )

    def save_config(self, config: dict) -> None:
        """保存配置到扩展数据目录的 config.yml（经由沙箱验证 FILESYSTEM_WRITE 权限）。

        参数:
            config: 要保存的配置字典
        """
        config_file = self.get_extension_data_dir() / "config.yml"
        content = yaml.safe_dump(
            config, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
        # 沙箱未初始化（测试或非加载流程调用），回退直接写入
        if self._context is None or self._context.sandbox is None:
            try:
                with open(config_file, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                logger.warning(f"Extension {self.metadata.name} failed to save config: {e}")
            return
        try:
            self._context.sandbox.safe_write_file(config_file, content)
        except Exception as e:
            logger.warning(f"Extension {self.metadata.name} failed to save config: {e}")

    def publish_event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """通过 EventBus 发布扩展事件。"""
        try:
            from src.events.event_bus import Event, get_event_bus

            bus = get_event_bus()
            event = Event(source=self.metadata.name, metadata=data or {})
            bus.publish(event)
        except Exception as e:
            logger.warning(f"Extension {self.metadata.name} failed to publish event: {e}")
