"""扩展包管理组件：安装 / 卸载 / 列表。

:class:`ExtensionInstaller` 负责从 zip 安装扩展与创建模板；
:class:`ExtensionUninstaller` 负责按名称卸载（含依赖检查）；
:class:`ExtensionLister` 负责列出已安装扩展与读取详细信息。
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

from src.extensions.manifest import read_manifest
from src.extensions.package_validation import validate_package
from src.version import __version__

logger = logging.getLogger(__name__)


class ExtensionInstaller:
    """扩展安装组件：从 .zip 包安装扩展并写入安装标记。"""

    def __init__(self, extensions_dir: Path):
        self._extensions_dir = extensions_dir

    def install(self, zip_path: Path, force: bool = False) -> dict:
        """从 .zip 文件安装扩展。

        Returns:
            {"success": bool, "name": str, "version": str, "error": str or None}
        """
        zip_path = Path(zip_path)

        report = validate_package(zip_path)
        if not report.is_valid:
            return {
                "success": False,
                "name": "",
                "version": "",
                "error": "包验证失败: " + "; ".join(report.messages),
            }

        try:
            manifest = read_manifest(zip_path)
        except ValueError as exc:
            return {"success": False, "name": "", "version": "", "error": str(exc)}

        target_dir = self._extensions_dir / manifest.name
        if target_dir.exists() and not force:
            return {
                "success": False,
                "name": manifest.name,
                "version": manifest.version,
                "error": f"扩展 '{manifest.name}' 已安装，使用 force=True 覆盖",
            }

        try:
            if target_dir.exists():
                shutil.rmtree(target_dir)

            target_dir.mkdir(parents=True, exist_ok=True)

            self._extract_package(zip_path, target_dir)

            self._write_installed_marker(
                target_dir,
                name=manifest.name,
                version=manifest.version,
                source=str(zip_path),
            )

            logger.info(f"扩展 '{manifest.name}' v{manifest.version} 安装成功")
            return {
                "success": True,
                "name": manifest.name,
                "version": manifest.version,
                "error": None,
            }
        except Exception as exc:
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            logger.error(f"安装扩展失败: {exc}")
            return {
                "success": False,
                "name": manifest.name,
                "version": manifest.version,
                "error": str(exc),
            }

    @staticmethod
    def _extract_package(zip_path: Path, target_dir: Path) -> None:
        """解压 zip 到目标目录（跳过路径遍历条目）。"""
        with zipfile.ZipFile(zip_path, "r") as zf:
            target_resolved = target_dir.resolve()
            for info in zf.infolist():
                if info.is_dir():
                    continue
                parts = Path(info.filename).parts

                if len(parts) > 1:
                    remaining = Path(*parts[1:])
                    dest = target_dir / remaining
                else:
                    dest = target_dir / Path(info.filename)

                try:
                    dest.resolve().relative_to(target_resolved)
                except ValueError:
                    continue

                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dest, "wb") as dst:
                    dst.write(src.read())

    @staticmethod
    def _write_installed_marker(
        ext_dir: Path, name: str, version: str, source: str
    ) -> None:
        """写入 ``.installed`` 标记文件（JSON 格式）。

        包含 name / version / installed_at / source 字段，用于替代 config.yml 中的
        ``installed_extensions`` 注册表。安装成功后写入，卸载时随目录一并删除。
        """
        marker = {
            "name": name,
            "version": version,
            "installed_at": datetime.now().isoformat(),
            "source": source,
        }
        (ext_dir / ".installed").write_text(
            json.dumps(marker, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def create_template(self, name: str, output_dir: Path | None = None) -> Path:
        """创建扩展包模板并打包为 .ppcext.zip。

        Returns:
            创建的 zip 文件路径
        """
        output_dir = output_dir or Path.cwd()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        class_name = _to_camel_case(name)

        manifest_content = yaml.dump(
            {
                "name": name,
                "version": "1.0.0",
                "description": f"{name} 扩展",
                "author": "",
                "extension_type": "tool_integration",
                "entry": "extension.py",
                "dependencies": [],
                "min_ppc_version": __version__,
                "tags": [],
                "permissions": [],
            },
            allow_unicode=True,
            default_flow_style=False,
        )

        extension_content = (
            f'"""{name} 扩展"""\n'
            f"\n"
            f"from src.extensions.base import Extension, ExtensionMetadata, ExtensionType, ToolIntegration\n"
            f"\n"
            f"\n"
            f"class {class_name}Extension(Extension, ToolIntegration):\n"
            f"    def __init__(self):\n"
            f"        metadata = ExtensionMetadata(\n"
            f'            name="{name}",\n'
            f'            version="1.0.0",\n'
            f'            description="{name} 扩展",\n'
            f"            extension_type=ExtensionType.TOOL_INTEGRATION,\n"
            f"        )\n"
            f"        super().__init__(metadata)\n"
            f"\n"
            f"    async def initialize(self) -> None:\n"
            f"        await super().initialize()\n"
            f"\n"
            f"    async def cleanup(self) -> None:\n"
            f"        await super().cleanup()\n"
            f"\n"
            f"    def is_available(self) -> bool:\n"
            f"        return True\n"
            f"\n"
            f"    def get_info(self) -> dict:\n"
            f'        return {{"name": "{name}", "version": "1.0.0"}}\n'
            f"\n"
            f"\n"
            f"extension = {class_name}Extension()\n"
        )

        zip_path = output_dir / f"{name}.ppcext.zip"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "manifest.yml").write_text(manifest_content, encoding="utf-8")
            (tmp_path / "extension.py").write_text(extension_content, encoding="utf-8")

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(tmp_path / "manifest.yml", "manifest.yml")
                zf.write(tmp_path / "extension.py", "extension.py")

        logger.info(f"扩展模板已创建: {zip_path}")
        return zip_path


class ExtensionUninstaller:
    """扩展卸载组件：按名称卸载（非 force 时检查依赖方）。"""

    def __init__(self, extensions_dir: Path):
        self._extensions_dir = extensions_dir

    def uninstall(self, name: str, force: bool = False) -> dict:
        """按名称卸载扩展。

        Returns:
            {"success": bool, "name": str, "error": str or None}
        """
        target_dir = self._extensions_dir / name
        if not target_dir.exists():
            return {
                "success": False,
                "name": name,
                "error": f"扩展 '{name}' 未安装",
            }

        if not force:
            dependents = self._find_dependents(name)
            if dependents:
                dep_names = ", ".join(d["name"] for d in dependents)
                return {
                    "success": False,
                    "name": name,
                    "error": f"扩展 '{name}' 被以下扩展依赖: {dep_names}，使用 force=True 强制卸载",
                }

        try:
            shutil.rmtree(target_dir)
            logger.info(f"扩展 '{name}' 已卸载")
            return {"success": True, "name": name, "error": None}
        except Exception as exc:
            logger.error(f"卸载扩展失败: {exc}")
            return {"success": False, "name": name, "error": str(exc)}

    def _find_dependents(self, name: str) -> list[dict]:
        """查找依赖指定扩展的所有已安装扩展。"""
        dependents: list[dict] = []
        if not self._extensions_dir.exists():
            return dependents

        for subdir in self._extensions_dir.iterdir():
            if not subdir.is_dir() or subdir.name == name:
                continue

            manifest_path = None
            for child in subdir.iterdir():
                if child.name == "manifest.yml":
                    manifest_path = child
                    break

            if manifest_path is None:
                continue

            try:
                raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and name in raw.get("dependencies", []):
                    dependents.append(
                        {
                            "name": raw.get("name", subdir.name),
                            "version": raw.get("version", "1.0.0"),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to read manifest in {subdir}: {e}")
                continue

        return dependents


class ExtensionLister:
    """扩展列表组件：列出已安装扩展、读取详细信息与安装标记。"""

    def __init__(self, extensions_dir: Path):
        self._extensions_dir = extensions_dir

    def list_packages(self) -> list[dict]:
        """列出所有已安装的扩展。

        Returns:
            [{"name": str, "version": str, "type": str, "enabled": bool, "path": str}]
        """
        result: list[dict] = []
        if not self._extensions_dir.exists():
            return result

        for subdir in sorted(self._extensions_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if subdir.name.startswith("_") or subdir.name.startswith("."):
                continue

            manifest_path = self._find_manifest(subdir)
            if manifest_path is None:
                continue

            try:
                raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    continue
                enabled = not (subdir / ".disabled").exists()
                result.append(
                    {
                        "name": raw.get("name", subdir.name),
                        "version": raw.get("version", "1.0.0"),
                        "type": raw.get("extension_type", "tool_integration"),
                        "enabled": enabled,
                        "path": str(subdir),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to read manifest in {subdir}: {e}")
                continue

        return result

    def get_package_info(self, name: str) -> dict | None:
        """获取已安装扩展的详细信息。

        Returns:
            完整 manifest 数据 + path + file_list + enabled 状态，若不存在返回 None
        """
        target_dir = self._extensions_dir / name
        if not target_dir.exists():
            return None

        manifest_path = self._find_manifest(target_dir)
        if manifest_path is None:
            return None

        try:
            raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None
        except Exception as e:
            logger.warning(f"Failed to read manifest in {target_dir}: {e}")
            return None

        file_list = []
        for f in target_dir.rglob("*"):
            if f.is_file():
                file_list.append(str(f.relative_to(target_dir)))

        enabled = not (target_dir / ".disabled").exists()

        info = dict(raw)
        info["path"] = str(target_dir)
        info["file_list"] = sorted(file_list)
        info["enabled"] = enabled
        return info

    def list_installed(self) -> list[dict]:
        """扫描 extensions_dir 中所有含 ``.installed`` 标记文件的子目录，返回已安装扩展列表。

        Returns:
            [{"name": str, "version": str, "installed_at": str, "source": str, "path": str}]
        """
        result: list[dict] = []
        if not self._extensions_dir.exists():
            return result

        for subdir in sorted(self._extensions_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if subdir.name.startswith("_") or subdir.name.startswith("."):
                continue

            marker = self._read_installed_marker(subdir)
            if marker is None:
                continue

            result.append(
                {
                    "name": marker.get("name", subdir.name),
                    "version": marker.get("version", "1.0.0"),
                    "installed_at": marker.get("installed_at", ""),
                    "source": marker.get("source", ""),
                    "path": str(subdir),
                }
            )

        return result

    @staticmethod
    def _find_manifest(ext_dir: Path) -> Path | None:
        """查找目录顶层的 manifest.yml。"""
        for child in ext_dir.iterdir():
            if child.is_file() and child.name == "manifest.yml":
                return child
        return None

    @staticmethod
    def _read_installed_marker(ext_dir: Path) -> dict | None:
        """读取 ``.installed`` 标记文件，返回解析后的字典；不存在或无效时返回 None。"""
        marker_file = ext_dir / ".installed"
        if not marker_file.exists():
            return None
        try:
            data = json.loads(marker_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, ValueError) as exc:
            logger.warning(f"读取 .installed 标记文件失败 {marker_file}: {exc}")
        return None


def _to_camel_case(name: str) -> str:
    """将 snake_case 或连字符名称转为 CamelCase。

    示例: my_extension → MyExtension, my-extension → MyExtension
    """
    parts = re.split(r"[_\-]+", name)
    return "".join(p.capitalize() for p in parts if p)
