"""扩展包格式定义、验证与管理。

本模块是扩展包管理的门面（facade）：

* 清单解析：:class:`ExtensionManifest` 与 :func:`read_manifest` /
  :func:`read_manifest_from_dir`（实现见 :mod:`src.extensions.manifest`）
* 包验证：:func:`validate_package` 与 :class:`PackageValidationReport`
  （实现见 :mod:`src.extensions.package_validation`）
* 安装/卸载/列表：:class:`ExtensionPackageManager` 门面委托给
  :mod:`src.extensions.package_ops` 中的 Installer / Uninstaller / Lister 组件
"""

from src.extensions._paths import default_user_extensions_dir
from src.extensions.manifest import (
    ExtensionManifest,
    read_manifest,
    read_manifest_from_dir,
)
from src.extensions.package_ops import (
    ExtensionInstaller,
    ExtensionLister,
    ExtensionUninstaller,
)
from src.extensions.package_validation import (
    PackageValidationReport,
    validate_package,
)

__all__ = [
    "ExtensionManifest",
    "read_manifest",
    "read_manifest_from_dir",
    "validate_package",
    "PackageValidationReport",
    "ExtensionPackageManager",
]


class ExtensionPackageManager:
    """扩展包管理器门面，负责安装、卸载、列表和模板创建。

    内部委托给 :class:`ExtensionInstaller` / :class:`ExtensionUninstaller` /
    :class:`ExtensionLister` 三个组件。
    """

    def __init__(self, extensions_dir=None):
        self._extensions_dir = extensions_dir or default_user_extensions_dir()
        self._installer = ExtensionInstaller(self._extensions_dir)
        self._uninstaller = ExtensionUninstaller(self._extensions_dir)
        self._lister = ExtensionLister(self._extensions_dir)

    @property
    def extensions_dir(self):
        return self._extensions_dir

    def install_package(self, zip_path, force: bool = False) -> dict:
        """从 .zip 文件安装扩展。

        Returns:
            {"success": bool, "name": str, "version": str, "error": str or None}
        """
        return self._installer.install(zip_path, force=force)

    def uninstall_package(self, name: str, force: bool = False) -> dict:
        """按名称卸载扩展。

        Returns:
            {"success": bool, "name": str, "error": str or None}
        """
        return self._uninstaller.uninstall(name, force=force)

    def list_packages(self) -> list[dict]:
        """列出所有已安装的扩展。

        Returns:
            [{"name": str, "version": str, "type": str, "enabled": bool, "path": str}]
        """
        return self._lister.list_packages()

    def get_package_info(self, name: str) -> dict | None:
        """获取已安装扩展的详细信息。

        Returns:
            完整 manifest 数据 + path + file_list + enabled 状态，若不存在返回 None
        """
        return self._lister.get_package_info(name)

    def list_installed(self) -> list[dict]:
        """扫描 extensions_dir 中所有含 ``.installed`` 标记文件的子目录，返回已安装扩展列表。

        Returns:
            [{"name": str, "version": str, "installed_at": str, "source": str, "path": str}]
        """
        return self._lister.list_installed()

    def create_template(self, name: str, output_dir=None):
        """创建扩展包模板并打包为 .ppcext.zip。

        Returns:
            创建的 zip 文件路径
        """
        return self._installer.create_template(name, output_dir)
