"""扩展安全扫描编排。

包装 :class:`ExtensionSecurityScanner`，提供：

* 基于文件 mtime/size 的扫描结果缓存（避免重复扫描）
* 已声明权限提取（manifest 优先，回退 AST 静态解析）
* 按安全级别（STRICT/WARN/ALLOW）决定是否阻止加载并记录扫描问题
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from src.extensions.base import SecurityLevel
from src.extensions.manifest import ExtensionManifest
from src.extensions.scanner import (
    ExtensionSecurityScanner,
    SecurityScanResult,
)

logger = logging.getLogger(__name__)

# Caches to avoid repeated expensive work across reloads
# _AST_PERMISSIONS_CACHE: path -> (mtime, size, permissions_set)
_AST_PERMISSIONS_CACHE: dict[str, tuple[float, int, set[str]]] = {}

# _SECURITY_SCAN_CACHE: path -> (mtime, size, SecurityScanResult)
_SECURITY_SCAN_CACHE: dict[str, tuple[float, int, SecurityScanResult]] = {}


def clear_caches() -> None:
    """清空权限提取与扫描结果缓存（重载扩展前调用）。"""
    _AST_PERMISSIONS_CACHE.clear()
    _SECURITY_SCAN_CACHE.clear()


def _ast_extract_permissions(ext_file: Path) -> set[str]:
    """从 extension.py 的 ExtensionMetadata(...) 调用中静态提取声明的权限。

    使用基于文件 mtime/size 的简单缓存避免在多次重载或扫描过程中重复解析同一文件。

    匹配形如 permissions=[ExtensionPermission.FILESYSTEM_WRITE, ...] 的字面量，
    返回权限值集合（小写，与 ExtensionPermission.value 一致）。
    解析失败或无声明时返回空集。
    """
    try:
        stat = ext_file.stat()
    except OSError:
        return set()

    key = str(ext_file)
    mtime = stat.st_mtime
    size = stat.st_size

    cache_entry = _AST_PERMISSIONS_CACHE.get(key)
    if cache_entry is not None and cache_entry[0] == mtime and cache_entry[1] == size:
        return set(cache_entry[2])

    try:
        tree = ast.parse(ext_file.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, OSError):
        _AST_PERMISSIONS_CACHE[key] = (mtime, size, set())
        return set()

    declared: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name != "ExtensionMetadata":
            continue
        for kw in node.keywords:
            if kw.arg != "permissions" or not isinstance(kw.value, ast.List):
                continue
            for elt in kw.value.elts:
                # 形如 ExtensionPermission.FILESYSTEM_WRITE
                if (
                    isinstance(elt, ast.Attribute)
                    and isinstance(elt.value, ast.Name)
                    and elt.value.id == "ExtensionPermission"
                ):
                    declared.add(elt.attr.lower())

    # 保存缓存副本
    _AST_PERMISSIONS_CACHE[key] = (mtime, size, set(declared))
    return declared


def extract_declared_permissions(
    extension_dir: Path, manifest: ExtensionManifest | None
) -> set[str]:
    """提取扩展已声明的权限（ExtensionPermission.value 字符串集合）。

    优先使用 manifest.permissions；若 manifest 缺失，则回退到对 extension.py
    做 AST 解析，提取 ExtensionMetadata(...) 中 permissions=[...] 字面量。
    """
    if manifest is not None and manifest.permissions:
        # manifest.permissions 为 list[str]，元素为权限值（如 "filesystem_write"）
        return {str(p) for p in manifest.permissions}
    ext_file = extension_dir / "extension.py"
    if ext_file.is_file():
        return _ast_extract_permissions(ext_file)
    return set()


class SecurityScanCoordinator:
    """安全扫描编排器：缓存、权限抑制、级别策略与问题日志。"""

    def __init__(self, scanner: ExtensionSecurityScanner):
        self._scanner = scanner
        self._last_results: list[SecurityScanResult] = []

    def last_results(self) -> list[SecurityScanResult]:
        """获取最近一次加载的安全扫描结果（副本）。"""
        return self._last_results.copy()

    def reset(self) -> None:
        """清空扫描结果与缓存（重载扩展前调用）。"""
        self._last_results.clear()
        clear_caches()

    def scan_file(
        self, file_path: Path, security_level: SecurityLevel
    ) -> SecurityScanResult | None:
        """执行单文件安全扫描并根据安全级别决定是否阻止加载。

        使用基于文件 mtime/size 的缓存避免重复扫描同一文件。

        参数:
            file_path: 要扫描的文件路径
            security_level: 当前安全级别

        返回:
            SecurityScanResult 如果应该阻止加载则返回扫描结果，否则返回 None
        """
        try:
            stat = file_path.stat()
        except OSError:
            return None

        key = str(file_path)
        mtime = stat.st_mtime
        size = stat.st_size

        cache_entry = _SECURITY_SCAN_CACHE.get(key)
        if cache_entry is not None and cache_entry[0] == mtime and cache_entry[1] == size:
            scan_result = cache_entry[2]
        else:
            scan_result = self._scanner.scan_file(file_path)
            # store cached result
            _SECURITY_SCAN_CACHE[key] = (mtime, size, scan_result)

        self._last_results.append(scan_result)

        if scan_result.issues:
            self.log_scan_issues(scan_result, security_level)

        if security_level == SecurityLevel.STRICT and scan_result.has_critical_or_high():
            return scan_result

        return None

    def scan_package(
        self,
        package_path: Path,
        declared_permissions: set[str],
        security_level: SecurityLevel,
    ) -> str | None:
        """扫描扩展包目录（全部文件），按已声明权限抑制对应 issue。

        参数:
            package_path: 扩展包目录
            declared_permissions: 扩展已声明的权限集合
            security_level: 当前安全级别

        返回:
            需要阻止加载时返回阻止消息字符串，否则返回 None
        """
        package_results = self._scanner.scan_directory(package_path)
        self._last_results.extend(package_results)
        has_blocking_issues = False
        for scan_result in package_results:
            # 先用扩展已声明的权限抑制对应 issue，再决定是否记录/阻止。
            # required_permissions 集合不受影响，仍反映扩展的真实依赖。
            scan_result.filter_by_declared_permissions(declared_permissions)
            if scan_result.issues:
                self.log_scan_issues(scan_result, security_level)
            if security_level == SecurityLevel.STRICT and scan_result.has_critical_or_high():
                has_blocking_issues = True
        if has_blocking_issues:
            total_risk = sum(r.risk_score for r in package_results)
            return f"Security scan blocked loading (total_risk_score={total_risk})"
        return None

    def log_scan_issues(self, scan_result: SecurityScanResult, security_level: SecurityLevel) -> None:
        """记录扫描发现的问题。

        参数:
            scan_result: 扫描结果
            security_level: 当前安全级别（决定日志级别）
        """
        file_name = scan_result.file_path.name
        risk_score = scan_result.risk_score
        issue_count = len(scan_result.issues)

        critical_count = scan_result.summary.get("critical", 0)
        high_count = scan_result.summary.get("high", 0)
        medium_count = scan_result.summary.get("medium", 0)
        low_count = scan_result.summary.get("low", 0)

        if security_level == SecurityLevel.ALLOW:
            logger.debug(
                f"[SECURITY AUDIT] {file_name}: risk_score={risk_score}, "
                f"issues={issue_count} (C:{critical_count} H:{high_count} M:{medium_count} L:{low_count})"
            )
        else:
            log_level = logging.WARNING if security_level == SecurityLevel.WARN else logging.ERROR
            logger.log(
                log_level,
                f"[SECURITY SCAN] {file_name}: risk_score={risk_score}, "
                f"issues={issue_count} (C:{critical_count} H:{high_count} M:{medium_count} L:{low_count})",
            )
            for issue in scan_result.issues:
                logger.log(log_level, f"  - [{issue.severity.value.upper()}] {issue.rule_id}: {issue.message} @ {issue.location}")
                logger.log(log_level, f"    建议: {issue.suggestion}")
