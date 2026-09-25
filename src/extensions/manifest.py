"""扩展清单（manifest.yml）定义与读取。

包含 :class:`ExtensionManifest` 数据类及从 zip 包 / 已解压目录读取
manifest 的函数。
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.extensions.base import ExtensionMetadata, ExtensionPermission, ExtensionType
from src.version import __version__

logger = logging.getLogger(__name__)


@dataclass
class ExtensionManifest:
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    extension_type: str = "tool_integration"
    entry: str = "extension.py"
    dependencies: list[str] = field(default_factory=list)
    min_ppc_version: str = __version__
    tags: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    package_type: str = "python"
    entry_point: str = ""
    requirements: list[str] = field(default_factory=list)
    config_schema: dict | None = None
    allowed_hosts: list[str] = field(default_factory=list)
    resource_limits: dict = field(default_factory=dict)
    sha256: str = ""

    def to_metadata(self) -> ExtensionMetadata:
        """转换为运行时 ExtensionMetadata。

        将 permissions 从 list[str] 转换为 list[ExtensionPermission]，
        将 extension_type 从 str 转换为 ExtensionType。无效值被跳过并记录警告，
        其余字段按原值复制。config / config_schema 按原值复制（可能为 None）。
        """
        permissions: list[ExtensionPermission] = []
        for perm_str in self.permissions or []:
            try:
                permissions.append(ExtensionPermission(perm_str))
            except ValueError:
                logger.warning(
                    f"无效的扩展权限字符串 '{perm_str}'，已跳过 (扩展: {self.name})"
                )

        extension_type: ExtensionType | None = None
        if self.extension_type:
            try:
                extension_type = ExtensionType(self.extension_type)
            except ValueError:
                logger.warning(
                    f"无效的扩展类型 '{self.extension_type}'，已跳过 (扩展: {self.name})"
                )

        return ExtensionMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            author=self.author,
            extension_type=extension_type,
            tags=list(self.tags or []),
            dependencies=list(self.dependencies or []),
            config=getattr(self, "config", None) or {},
            config_schema=self.config_schema,
            permissions=permissions,
            package_type=self.package_type,
            entry_point=self.entry_point,
            requirements=list(self.requirements or []),
            allowed_hosts=list(self.allowed_hosts or []),
            resource_limits=dict(self.resource_limits or {}),
            sha256=self.sha256,
        )


def _build_manifest(raw: dict) -> ExtensionManifest:
    """从解析后的 manifest 字典构建 ExtensionManifest。"""
    return ExtensionManifest(
        name=raw["name"],
        version=raw.get("version", "1.0.0"),
        description=raw.get("description", ""),
        author=raw.get("author", ""),
        extension_type=raw.get("extension_type", "tool_integration"),
        entry=raw.get("entry", "extension.py"),
        dependencies=raw.get("dependencies", []),
        min_ppc_version=raw.get("min_ppc_version", __version__),
        tags=raw.get("tags", []),
        permissions=raw.get("permissions", []),
        package_type=raw.get("package_type", "python"),
        entry_point=raw.get("entry_point", ""),
        requirements=raw.get("requirements", []),
        config_schema=raw.get("config_schema"),
        allowed_hosts=raw.get("allowed_hosts", []),
        resource_limits=raw.get("resource_limits", {}),
        sha256=raw.get("sha256", ""),
    )


def read_manifest(zip_path: Path) -> ExtensionManifest:
    """从 zip 包中读取 manifest.yml 并返回 ExtensionManifest。

    Raises:
        ValueError: manifest.yml 缺失或内容无效时抛出。
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise ValueError(f"扩展包文件不存在: {zip_path}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            manifest_data = None
            for name in zf.namelist():
                basename = Path(name).name
                if basename == "manifest.yml":
                    manifest_data = zf.read(name)
                    break

            if manifest_data is None:
                raise ValueError(f"扩展包中缺少 manifest.yml: {zip_path}")

            raw = yaml.safe_load(manifest_data)
            if not isinstance(raw, dict):
                raise ValueError(f"manifest.yml 内容无效，期望字典类型: {zip_path}")

            required_fields = ("name",)
            for fld in required_fields:
                if fld not in raw:
                    raise ValueError(f"manifest.yml 缺少必填字段 '{fld}': {zip_path}")

            return _build_manifest(raw)
    except zipfile.BadZipFile:
        raise ValueError(f"文件不是有效的 zip 包: {zip_path}") from None


def read_manifest_from_dir(dir_path: Path) -> ExtensionManifest | None:
    """从已解压的扩展目录读取 manifest.yml。

    返回 ExtensionMetadata；若 manifest.yml 缺失或无效则返回 None。
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return None

    manifest_file = None
    for child in dir_path.iterdir():
        if child.is_file() and child.name == "manifest.yml":
            manifest_file = child
            break
    if manifest_file is None:
        return None

    try:
        raw = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError):
        return None
    if not isinstance(raw, dict) or "name" not in raw:
        return None

    return _build_manifest(raw)
