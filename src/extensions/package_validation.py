"""扩展包验证：将 ``validate_package`` 拆分为独立的检查函数。

检查项（每个对应一个检查函数）:
  1. :func:`_check_package_openable` — 包文件存在且为有效 zip
  2. :func:`_check_manifest_present` — zip 中包含 manifest.yml
  3. :func:`_check_manifest_format` — manifest.yml 可解析为字典
  4. :func:`_check_required_fields` — 必填字段 (name/version/entry) 与名称格式
  5. :func:`_check_entry_file` — entry 指定的文件存在于 zip 中
  6. :func:`_check_version_compatible` — min_ppc_version 与当前版本兼容
  7. :func:`_check_package_type` — package_type 取值 / entry_point 格式 /
     executable 类型 sha256 哈希
  8. :func:`_collect_warnings` — 推荐项（requirements.txt / config.yml，WARNING 级别）

结果统一为 :class:`PackageValidationReport` dataclass。
"""

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from packaging import version as pkg_version

# 版本号单一来源：直接 import ppc 在 Nuitka standalone 下可能因主脚本
# 未作为可导入模块编译而失败（ModuleNotFoundError）。
from src.version import __version__ as ppc_version

logger = logging.getLogger(__name__)


@dataclass
class PackageValidationReport:
    """扩展包验证报告。

    属性:
        errors: 阻塞性错误列表（非空时包无效）
        warnings: 警告列表（``"WARNING: ..."`` 前缀，不阻塞安装）
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """无阻塞性错误时返回 True。"""
        return len(self.errors) == 0

    @property
    def messages(self) -> list[str]:
        """错误与警告的合并列表（错误在前）。"""
        return self.errors + self.warnings


def _check_manifest_present(names: list[str]) -> str | None:
    """检查 zip 中是否包含 manifest.yml，返回其条目名或 None。"""
    for name in names:
        if Path(name).name == "manifest.yml":
            return name
    return None


def _check_manifest_format(zf: zipfile.ZipFile, manifest_name: str) -> dict | None:
    """检查 manifest.yml 可解析为字典；失败时抛出 ValueError。"""
    try:
        manifest_data = yaml.safe_load(zf.read(manifest_name))
    except yaml.YAMLError as exc:
        raise ValueError(f"manifest.yml 解析失败: {exc}") from None
    if not isinstance(manifest_data, dict):
        raise ValueError("manifest.yml 内容无效，期望字典类型")
    return manifest_data


def _check_required_fields(manifest: dict) -> list[str]:
    """检查必填字段 (name, version, entry) 与名称字符格式。"""
    errors: list[str] = []
    for fld in ("name", "version", "entry"):
        if fld not in manifest:
            errors.append(f"manifest.yml 缺少必填字段 '{fld}'")
    if errors:
        return errors

    name = manifest.get("name", "")
    if not re.match(r"^[a-zA-Z0-9_-]+$", name):
        errors.append(
            "Invalid extension name: must contain only alphanumeric, underscore, and hyphen characters"
        )
    return errors


def _check_entry_file(names: list[str], manifest: dict) -> list[str]:
    """检查 entry 指定的入口文件存在于 zip 中。"""
    entry_file = manifest.get("entry", "extension.py")
    for name in names:
        if Path(name).name == entry_file:
            return []
    return [f"入口文件 '{entry_file}' 在扩展包中不存在"]


def _check_version_compatible(manifest: dict) -> list[str]:
    """检查 min_ppc_version 与当前 PPC_NEXT 版本兼容。"""
    min_ver = manifest.get("min_ppc_version", ppc_version)
    try:
        current = pkg_version.parse(ppc_version)
        required = pkg_version.parse(min_ver)
        if current < required:
            return [f"当前 PPC_NEXT 版本 {ppc_version} 低于扩展要求的最低版本 {min_ver}"]
    except pkg_version.InvalidVersion as exc:
        return [f"版本号格式无效: {exc}"]
    return []


def _check_package_type(manifest: dict) -> list[str]:
    """检查 package_type 取值、entry_point 格式与 executable 类型 sha256 哈希。"""
    errors: list[str] = []

    # 校验 package_type 必须为 python / executable 之一
    package_type = manifest.get("package_type", "python")
    if package_type not in ("python", "executable"):
        errors.append(
            f"无效的 package_type '{package_type}'，必须为 'python' 或 'executable'"
        )

    # 校验 entry_point 格式（若存在）：必须为 "module:ClassName"
    entry_point = manifest.get("entry_point", "")
    if entry_point:
        if ":" not in entry_point:
            errors.append(
                f"entry_point 格式无效 '{entry_point}'，必须为 'module:ClassName'"
            )
        else:
            module_part, _, class_part = entry_point.partition(":")
            if not module_part or not class_part:
                errors.append(
                    f"entry_point 格式无效 '{entry_point}'，module 与 ClassName 均不能为空"
                )

    # python 类型：entry_point 为空时仅 WARNING（不阻塞，加载时 fallback 到 entry）
    if package_type == "python" and not entry_point:
        logger.warning(
            "扩展 %s 的 entry_point 为空，加载时将 fallback 到 entry 字段",
            manifest.get("name", ""),
        )

    # executable 类型：sha256 必填且为 64 位十六进制；entry_point 必填
    if package_type == "executable":
        if not entry_point:
            errors.append("executable 类型扩展必须提供 entry_point 字段")
        sha256 = manifest.get("sha256", "")
        if not sha256:
            errors.append("executable 类型扩展必须提供 sha256 字段")
        elif len(sha256) != 64:
            errors.append(
                f"sha256 长度必须为 64 个十六进制字符，当前长度 {len(sha256)}"
            )
        elif not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
            errors.append("sha256 必须仅包含十六进制字符 (0-9, a-f, A-F)")

    return errors


def _collect_warnings(names: list[str], manifest: dict) -> list[str]:
    """收集推荐项警告（requirements.txt / config.yml，WARNING 级别，不阻塞）。"""
    warnings: list[str] = []
    package_type = manifest.get("package_type", "python")
    name = manifest.get("name", "")

    # python 类型：推荐包含 requirements.txt
    # executable 类型使用独立的运行时，无需 requirements.txt，跳过该检查
    if package_type == "python":
        has_requirements = any(Path(n).name == "requirements.txt" for n in names)
        if not has_requirements:
            warnings.append(
                "WARNING: requirements.txt not found in python extension package (recommended)"
            )

    # 推荐包含 config.yml 扩展私有配置
    has_config_yml = any(Path(n).name == "config.yml" for n in names)
    if not has_config_yml:
        warnings.append(
            "WARNING: config.yml not found in package (recommended)"
        )
    _ = name
    return warnings


def validate_package(zip_path: Path) -> PackageValidationReport:
    """验证扩展包的完整性。

    检查项:
      1. zip 包包含 manifest.yml 且可解析为字典
      2. manifest 具有必填字段 (name, version, entry)，名称格式合法
      3. entry 指定的文件存在于 zip 中
      4. min_ppc_version 与当前 PPC_NEXT 版本兼容
      5. package_type 合法；entry_point 格式正确；executable 类型 sha256 完整
      6. python 类型推荐包含 requirements.txt（WARNING 级别，不阻塞）
      7. 推荐包含 config.yml 扩展私有配置（WARNING 级别，不阻塞）

    Returns:
        :class:`PackageValidationReport`。``report.is_valid`` 为 False 时
        ``report.errors`` 非空；警告位于 ``report.warnings``（不阻塞）。
    """
    zip_path = Path(zip_path)
    report = PackageValidationReport()

    if not zip_path.exists():
        report.errors.append(f"扩展包文件不存在: {zip_path}")
        return report

    try:
        zf = zipfile.ZipFile(zip_path, "r")
    except zipfile.BadZipFile:
        report.errors.append(f"文件不是有效的 zip 包: {zip_path}")
        return report

    with zf:
        names = zf.namelist()

        manifest_name = _check_manifest_present(names)
        if manifest_name is None:
            report.errors.append("扩展包中缺少 manifest.yml")
            return report

        try:
            manifest = _check_manifest_format(zf, manifest_name)
        except ValueError as exc:
            report.errors.append(str(exc))
            return report

        assert manifest is not None
        report.errors.extend(_check_required_fields(manifest))
        if report.errors:
            return report

        report.errors.extend(_check_entry_file(names, manifest))
        report.errors.extend(_check_version_compatible(manifest))
        report.errors.extend(_check_package_type(manifest))
        report.warnings.extend(_collect_warnings(names, manifest))

    return report
