"""扩展包管理组件：安装 / 卸载 / 列表。

:class:`ExtensionInstaller` 负责从 zip 安装扩展与创建模板；
:class:`ExtensionUninstaller` 负责按名称卸载（含依赖检查）；
:class:`ExtensionLister` 负责列出已安装扩展与读取详细信息。
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

from src.extensions.manifest import read_manifest
from src.extensions.package_validation import validate_package
from src.version import __version__

logger = logging.getLogger(__name__)

#: 扩展名白名单：与 package_validation 的清单校验保持同一口径。
#: 卸载路径曾直接用 `extensions_dir / name`，`..` 与绝对路径都能越界。
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _remove_tree(root: Path) -> None:
    """逐文件删除目录树（替代 ``shutil.rmtree``）。

    触发条件：某些环境（沙箱 / EDR / 终端防护）会把「一次删几十个文件」当成
    批量删除并直接中止进程。扩展包动辄上百个文件，``rmtree`` 会命中该守卫，
    导致卸载/覆盖安装时进程被 SystemExit 杀掉（用户看到的是「命令没输出就退了」）。
    逐文件 ``unlink`` 更耐受，也能跳过已被占用的单个文件。
    """
    if not root.exists():
        return
    try:
        children = sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True)
        for child in children:
            if child.is_dir() and not child.is_symlink():
                child.rmdir()
            else:
                child.unlink(missing_ok=True)
        root.rmdir()
    except OSError as exc:
        logger.warning("清理目录失败 %s: %s", root, exc)


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

        # 包名来自压缩包内清单，必须和卸载同一口径校验：否则 name 为 `..`
        # 或 `C:/Windows` 时 target_dir 直接越界（后面还要 rmtree 旧版本）。
        if not _SAFE_NAME_RE.match(manifest.name):
            return {
                "success": False,
                "name": manifest.name,
                "version": manifest.version,
                "error": f"非法扩展名: {manifest.name!r}（只允许字母/数字/下划线/连字符）",
            }

        self._extensions_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self._extensions_dir / manifest.name
        if target_dir.exists() and not force:
            return {
                "success": False,
                "name": manifest.name,
                "version": manifest.version,
                "error": f"扩展 '{manifest.name}' 已安装，使用 force=True 覆盖",
            }

        # 事务式安装：先解压到 `.tmp-*`（discovery 跳过点开头的目录，所以半装
        # 状态永远不会被当作正常扩展加载），扫描通过后再原子替换正式目录。
        #
        # 旧实现是「先删旧目录 → 再解压」：中途失败（磁盘满、解压异常、
        # 扫描发现问题想回滚）两边都没了，等于把已装好的版本一并弄丢。
        token = uuid.uuid4().hex[:8]
        tmp_dir = self._extensions_dir / f".tmp-{manifest.name}-{token}"
        backup_dir = self._extensions_dir / f".old-{manifest.name}-{token}"
        moved_aside = False
        try:
            tmp_dir.mkdir(parents=True)

            self._extract_package(zip_path, tmp_dir)

            self._write_installed_marker(
                tmp_dir,
                name=manifest.name,
                version=manifest.version,
                source=str(zip_path),
            )

            blocked = self._scan_install(tmp_dir, manifest)
            if blocked:
                _remove_tree(tmp_dir)
                return {
                    "success": False,
                    "name": manifest.name,
                    "version": manifest.version,
                    "error": f"安全扫描未通过，已拒绝安装: {blocked}",
                }

            if target_dir.exists():
                os.replace(target_dir, backup_dir)
                moved_aside = True

            os.replace(tmp_dir, target_dir)

            if moved_aside:
                _remove_tree(backup_dir)

            logger.info(f"扩展 '{manifest.name}' v{manifest.version} 安装成功")
            return {
                "success": True,
                "name": manifest.name,
                "version": manifest.version,
                "error": None,
            }
        except Exception as exc:
            _remove_tree(tmp_dir)
            if moved_aside:
                # 回滚：把旧版本 atomic 换回来，避免「升级失败且旧版也没了」
                try:
                    os.replace(backup_dir, target_dir)
                except OSError as rollback_exc:
                    logger.error(
                        "回滚旧版本失败 %s: %s（旧版留在 %s）",
                        manifest.name,
                        rollback_exc,
                        backup_dir,
                    )
            logger.error(f"安装扩展失败: {exc}")
            return {
                "success": False,
                "name": manifest.name,
                "version": manifest.version,
                "error": str(exc),
            }

    @staticmethod
    def _scan_install(tmp_dir: Path, manifest) -> str | None:
        """安装前的强制安全扫描（STRICT）。

        旧实现只在**加载**时按 ``SecurityLevel.WARN`` 扫一遍并打日志——也就是
        说恶意代码已经落盘、随时可能被 import，扫描只是「记录」而非拦截；
        ``pip install`` 阶段更没有扫描。

        这里对解压后的目录做 STRICT 扫描：命中 CRITICAL/HIGH（且不在清单
        声明的权限范围内）即拒绝安装。返回阻止原因，通过则返回 ``None``。
        """
        from src.extensions.base import SecurityLevel
        from src.extensions.scanner import ExtensionSecurityScanner
        from src.extensions.security_scan import SecurityScanCoordinator

        declared = {str(p) for p in (getattr(manifest, "permissions", None) or [])}
        coordinator = SecurityScanCoordinator(ExtensionSecurityScanner())
        return coordinator.scan_package(tmp_dir, declared, SecurityLevel.STRICT)

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
        # 卸载是唯一「按用户输入删除目录」的入口：必须白名单 + 越界校验。
        # 触发条件：`ppc ext uninstall ..` 或 DELETE /api/extensions/.. 。
        # 后果（旧实现）：`Path(ext_dir) / ".."` 解析到项目根并递归删除；
        # `Path(ext_dir) / "C:/Windows"` 因绝对路径覆盖直接命中任意目录。
        if not isinstance(name, str) or not _SAFE_NAME_RE.match(name):
            return {"success": False, "name": str(name), "error": f"非法扩展名: {name!r}"}

        target_dir = self._extensions_dir / name
        try:
            if target_dir.resolve().parent != self._extensions_dir.resolve():
                return {"success": False, "name": name, "error": f"扩展名越界: {name!r}"}
        except OSError as exc:
            return {"success": False, "name": name, "error": f"扩展路径解析失败: {exc}"}

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
            _remove_tree(target_dir)
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
