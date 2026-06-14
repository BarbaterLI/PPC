"""扩展包格式定义、验证与管理。

本模块实现扩展包 (.ppc10ext.zip) 的清单解析、包验证、安装/卸载/模板创建等功能。
"""

import logging
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from packaging import version as pkg_version

import ppc10

logger = logging.getLogger(__name__)


@dataclass
class ExtensionManifest:
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    extension_type: str = "tool_integration"
    entry: str = "extension.py"
    dependencies: List[str] = field(default_factory=list)
    min_ppc10_version: str = "10.0.0"
    tags: List[str] = field(default_factory=list)
    pipelines: List[str] = field(default_factory=list)


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

            return ExtensionManifest(
                name=raw["name"],
                version=raw.get("version", "1.0.0"),
                description=raw.get("description", ""),
                author=raw.get("author", ""),
                extension_type=raw.get("extension_type", "tool_integration"),
                entry=raw.get("entry", "extension.py"),
                dependencies=raw.get("dependencies", []),
                min_ppc10_version=raw.get("min_ppc10_version", "10.0.0"),
                tags=raw.get("tags", []),
                pipelines=raw.get("pipelines", []),
            )
    except zipfile.BadZipFile:
        raise ValueError(f"文件不是有效的 zip 包: {zip_path}")


def validate_package(zip_path: Path) -> Tuple[bool, List[str]]:
    """验证扩展包的完整性。

    检查项:
      1. zip 包中包含 manifest.yml
      2. manifest 具有必填字段 (name, version, entry)
      3. entry 指定的文件存在于 zip 中
      4. min_ppc10_version 与当前 PPC10 版本兼容

    Returns:
        (is_valid, errors) 元组
    """
    zip_path = Path(zip_path)
    errors: List[str] = []

    if not zip_path.exists():
        return False, [f"扩展包文件不存在: {zip_path}"]

    try:
        zf = zipfile.ZipFile(zip_path, "r")
    except zipfile.BadZipFile:
        return False, [f"文件不是有效的 zip 包: {zip_path}"]

    with zf:
        names = zf.namelist()

        manifest_name = None
        for name in names:
            if Path(name).name == "manifest.yml":
                manifest_name = name
                break

        if manifest_name is None:
            errors.append("扩展包中缺少 manifest.yml")
            return False, errors

        try:
            manifest_data = yaml.safe_load(zf.read(manifest_name))
        except yaml.YAMLError as exc:
            errors.append(f"manifest.yml 解析失败: {exc}")
            return False, errors

        if not isinstance(manifest_data, dict):
            errors.append("manifest.yml 内容无效，期望字典类型")
            return False, errors

        for fld in ("name", "version", "entry"):
            if fld not in manifest_data:
                errors.append(f"manifest.yml 缺少必填字段 '{fld}'")

        if errors:
            return False, errors

        entry_file = manifest_data.get("entry", "extension.py")
        entry_found = False
        for name in names:
            if Path(name).name == entry_file:
                entry_found = True
                break
        if not entry_found:
            errors.append(f"入口文件 '{entry_file}' 在扩展包中不存在")

        min_ver = manifest_data.get("min_ppc10_version", "10.0.0")
        try:
            current = pkg_version.parse(ppc10.__version__)
            required = pkg_version.parse(min_ver)
            if current < required:
                errors.append(
                    f"当前 PPC10 版本 {ppc10.__version__} 低于扩展要求的最低版本 {min_ver}"
                )
        except pkg_version.InvalidVersion as exc:
            errors.append(f"版本号格式无效: {exc}")

    is_valid = len(errors) == 0
    return is_valid, errors


def _to_camel_case(name: str) -> str:
    """将 snake_case 或连字符名称转为 CamelCase。

    示例: my_extension → MyExtension, my-extension → MyExtension
    """
    parts = re.split(r"[_\-]+", name)
    return "".join(p.capitalize() for p in parts if p)


class ExtensionPackageManager:
    """扩展包管理器，负责安装、卸载、列表和模板创建。"""

    def __init__(self, extensions_dir: Optional[Path] = None):
        self._extensions_dir = extensions_dir or Path(__file__).parent.parent.parent / "extensions"

    @property
    def extensions_dir(self) -> Path:
        return self._extensions_dir

    def install_package(self, zip_path: Path, force: bool = False) -> Dict:
        """从 .zip 文件安装扩展。

        Returns:
            {"success": bool, "name": str, "version": str, "error": str or None}
        """
        zip_path = Path(zip_path)

        is_valid, errors = validate_package(zip_path)
        if not is_valid:
            return {
                "success": False,
                "name": "",
                "version": "",
                "error": "包验证失败: " + "; ".join(errors),
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

            with zipfile.ZipFile(zip_path, "r") as zf:
                target_resolved = target_dir.resolve()
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    parts = Path(info.filename).parts
                    if len(parts) > 1 and not Path(info.filename).name == "manifest.yml" and not Path(info.filename).name == manifest.entry:
                        rel = Path(*parts[1:])
                    else:
                        rel = Path(info.filename)

                    if len(parts) > 1:
                        top_dir = parts[0]
                        remaining = Path(*parts[1:])
                        dest = target_dir / remaining
                    else:
                        dest = target_dir / Path(info.filename)

                    if not str(dest.resolve()).startswith(str(target_resolved)):
                        continue

                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(dest, "wb") as dst:
                        dst.write(src.read())

            installed_file = target_dir / ".installed"
            installed_file.write_text(
                f'{{"installed_at": "{datetime.now().isoformat()}", "source": "{str(zip_path)}"}}',
                encoding="utf-8",
            )

            self._register_in_config(manifest, zip_path)

            self._install_pipelines(manifest, target_dir)

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

    def uninstall_package(self, name: str, force: bool = False) -> Dict:
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
            self._unregister_from_config(name)
            logger.info(f"扩展 '{name}' 已卸载")
            return {"success": True, "name": name, "error": None}
        except Exception as exc:
            logger.error(f"卸载扩展失败: {exc}")
            return {"success": False, "name": name, "error": str(exc)}

    def list_packages(self) -> List[Dict]:
        """列出所有已安装的扩展。

        Returns:
            [{"name": str, "version": str, "type": str, "enabled": bool, "path": str}]
        """
        result: List[Dict] = []
        if not self._extensions_dir.exists():
            return result

        for subdir in sorted(self._extensions_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if subdir.name.startswith("_") or subdir.name.startswith("."):
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
                if not isinstance(raw, dict):
                    continue
                enabled = not (subdir / ".disabled").exists()
                result.append({
                    "name": raw.get("name", subdir.name),
                    "version": raw.get("version", "1.0.0"),
                    "type": raw.get("extension_type", "tool_integration"),
                    "enabled": enabled,
                    "path": str(subdir),
                })
            except Exception:
                continue

        return result

    def get_package_info(self, name: str) -> Optional[Dict]:
        """获取已安装扩展的详细信息。

        Returns:
            完整 manifest 数据 + path + file_list + enabled 状态，若不存在返回 None
        """
        target_dir = self._extensions_dir / name
        if not target_dir.exists():
            return None

        manifest_path = None
        for child in target_dir.iterdir():
            if child.name == "manifest.yml":
                manifest_path = child
                break

        if manifest_path is None:
            return None

        try:
            raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None
        except Exception:
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

    def create_template(self, name: str, output_dir: Optional[Path] = None) -> Path:
        """创建扩展包模板并打包为 .ppc10ext.zip。

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
                "min_ppc10_version": "10.0.0",
                "tags": [],
                "pipelines": ["pipelines/example_pipeline.yaml"],
            },
            allow_unicode=True,
            default_flow_style=False,
        )

        extension_content = (
            f'"""{name} 扩展"""\n'
            f"\n"
            f"from src_m.extensions.base import Extension, ExtensionMetadata, ExtensionType, ToolIntegration\n"
            f"\n"
            f"\n"
            f"class {class_name}Extension(Extension, ToolIntegration):\n"
            f"    def __init__(self):\n"
            f'        metadata = ExtensionMetadata(\n'
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

        zip_path = output_dir / f"{name}.ppc10ext.zip"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "manifest.yml").write_text(manifest_content, encoding="utf-8")
            (tmp_path / "extension.py").write_text(extension_content, encoding="utf-8")

            pipelines_dir = tmp_path / "pipelines"
            pipelines_dir.mkdir(parents=True, exist_ok=True)
            example_pipeline_content = yaml.dump(
                {
                    "name": f"{name}_example",
                    "description": f"{name} 扩展示例管道",
                    "steps": [
                        {
                            "name": "example_step",
                            "type": "fanqie_download",
                            "params": {"book_id": "12345"},
                        },
                    ],
                    "variables": {},
                },
                allow_unicode=True,
                default_flow_style=False,
            )
            (pipelines_dir / "example_pipeline.yaml").write_text(example_pipeline_content, encoding="utf-8")

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(tmp_path / "manifest.yml", "manifest.yml")
                zf.write(tmp_path / "extension.py", "extension.py")
                zf.write(pipelines_dir / "example_pipeline.yaml", "pipelines/example_pipeline.yaml")

        logger.info(f"扩展模板已创建: {zip_path}")
        return zip_path

    def _find_dependents(self, name: str) -> List[Dict]:
        """查找依赖指定扩展的所有已安装扩展。"""
        dependents: List[Dict] = []
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
                    dependents.append({
                        "name": raw.get("name", subdir.name),
                        "version": raw.get("version", "1.0.0"),
                    })
            except Exception:
                continue

        return dependents

    def _install_pipelines(self, manifest: ExtensionManifest, target_dir: Path) -> None:
        """安装扩展包中包含的管道 YAML 文件到配置的管道目录，并注册到 saved_pipelines。"""
        if not manifest.pipelines:
            return

        try:
            from src_m.config.manager import ConfigManager
            from src_m.config.schema import PipelineDefinitionConfig, PipelineStepConfig

            cfg_mgr = ConfigManager()
            cfg = cfg_mgr.get_config()
            pipeline_dirs = cfg.pipeline.pipeline_dirs

            dest_dir = None
            for dir_path in pipeline_dirs:
                p = Path(dir_path)
                if not p.is_absolute():
                    p = Path(cfg_mgr.config_dir) / p
                p.mkdir(parents=True, exist_ok=True)
                dest_dir = p
                break

            if dest_dir is None:
                dest_dir = Path(cfg_mgr.config_dir) / "pipelines"
                dest_dir.mkdir(parents=True, exist_ok=True)

            for pipeline_file in manifest.pipelines:
                src = target_dir / pipeline_file
                if not src.exists():
                    logger.warning(f"管道文件 '{pipeline_file}' 在扩展目录中不存在，跳过")
                    continue

                dest = dest_dir / src.name
                shutil.copy2(src, dest)
                logger.info(f"管道文件已安装: {src.name} -> {dest}")

                try:
                    pipeline_data = yaml.safe_load(dest.read_text(encoding="utf-8"))
                    if isinstance(pipeline_data, dict) and "name" in pipeline_data:
                        pipe_name = pipeline_data["name"]
                        steps = []
                        for step_data in pipeline_data.get("steps", []):
                            if isinstance(step_data, dict):
                                steps.append(PipelineStepConfig(
                                    name=step_data.get("name", ""),
                                    step_type=step_data.get("type", step_data.get("step_type", "")),
                                    depends_on=step_data.get("depends_on", []),
                                    params=step_data.get("params", {}),
                                    retry_count=step_data.get("retry", step_data.get("retry_count", 0)),
                                    timeout_seconds=step_data.get("timeout", step_data.get("timeout_seconds")),
                                    on_failure=step_data.get("on_failure", "stop"),
                                ))

                        pipe_id = f"ext_{manifest.name}_{src.stem}"
                        cfg.pipeline.saved_pipelines[pipe_id] = PipelineDefinitionConfig(
                            name=pipe_name,
                            description=pipeline_data.get("description", ""),
                            steps=steps,
                            variables=pipeline_data.get("variables", {}),
                        )
                except Exception as exc:
                    logger.warning(f"解析管道文件 '{pipeline_file}' 失败: {exc}")

            cfg_mgr.save()
        except Exception as exc:
            logger.warning(f"安装管道文件失败: {exc}")

    def _register_in_config(self, manifest: ExtensionManifest, zip_path: Path) -> None:
        """将扩展注册到配置中。"""
        try:
            from src_m.config.schema import InstalledExtensionInfo
            from src_m.config.manager import ConfigManager

            cfg_mgr = ConfigManager()
            cfg = cfg_mgr.get_config()
            cfg.extensions.installed_extensions[manifest.name] = InstalledExtensionInfo(
                name=manifest.name,
                version=manifest.version,
                installed_at=datetime.now().isoformat(),
                source_path=str(zip_path),
            )
            cfg_mgr.save()
        except Exception as exc:
            logger.warning(f"注册扩展到配置失败: {exc}")

    def _unregister_from_config(self, name: str) -> None:
        """从配置中取消注册扩展。"""
        try:
            from src_m.config.manager import ConfigManager

            cfg_mgr = ConfigManager()
            cfg = cfg_mgr.get_config()
            cfg.extensions.installed_extensions.pop(name, None)
            cfg_mgr.save()
        except Exception as exc:
            logger.warning(f"从配置中取消注册扩展失败: {exc}")
