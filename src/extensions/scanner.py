"""扩展安全扫描模块。

在扩展加载前进行静态安全扫描，检测潜在的安全风险，包括：
- 危险导入和函数调用
- 敏感信息硬编码
- 网络请求、文件操作和子进程调用检测
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SecuritySeverity(Enum):
    """安全问题严重程度枚举。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> int:
        """返回严重程度对应的权重值，用于计算风险分数。"""
        weights = {
            SecuritySeverity.LOW: 1,
            SecuritySeverity.MEDIUM: 5,
            SecuritySeverity.HIGH: 20,
            SecuritySeverity.CRITICAL: 50,
        }
        return weights[self]


@dataclass
class SecurityIssue:
    """安全问题条目。"""

    severity: SecuritySeverity
    rule_id: str
    message: str
    location: str
    suggestion: str

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "severity": self.severity.value,
            "rule_id": self.rule_id,
            "message": self.message,
            "location": self.location,
            "suggestion": self.suggestion,
        }


@dataclass
class SecurityScanResult:
    """安全扫描结果。"""

    file_path: Path
    issues: list[SecurityIssue] = field(default_factory=list)
    risk_score: int = 0
    summary: dict[str, int] = field(default_factory=dict)
    required_permissions: set[str] = field(default_factory=set)

    def calculate_risk_score(self) -> None:
        """根据问题严重程度计算风险分数并生成摘要。"""
        self.risk_score = 0
        self.summary = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        for issue in self.issues:
            self.risk_score += issue.severity.weight
            self.summary[issue.severity.value] += 1

    def has_critical_or_high(self) -> bool:
        """检查是否包含 CRITICAL 或 HIGH 级别的问题。"""
        return any(
            issue.severity in (SecuritySeverity.CRITICAL, SecuritySeverity.HIGH)
            for issue in self.issues
        )

    def filter_by_declared_permissions(self, declared: set[str]) -> None:
        """根据扩展已声明的权限抑制对应 issue，并重算 risk_score。

        仅抑制 _RULE_TO_PERMISSION 中映射到已声明权限的 issue；
        pickle/eval/exec/SUBPROCESS_SHELL_TRUE 等无对应权限或本身高危的
        规则不受影响，仍会保留。required_permissions 集合保持不变，
        以便审计/统计仍能反映扩展的实际依赖。

        参数:
            declared: 扩展已声明的权限值集合（ExtensionPermission.value，
                      小写，如 {"subprocess", "filesystem_write"}）
        """
        if not declared:
            return
        self.issues = [
            issue
            for issue in self.issues
            if not (
                issue.rule_id in _RULE_TO_PERMISSION
                and _RULE_TO_PERMISSION[issue.rule_id] in declared
            )
        ]
        self.calculate_risk_score()

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "file_path": str(self.file_path),
            "issues": [issue.to_dict() for issue in self.issues],
            "risk_score": self.risk_score,
            "summary": self.summary,
            "required_permissions": list(self.required_permissions),
        }


_SENSITIVE_KEY_PATTERNS = [
    re.compile(
        r"""(?ix)
        (?P<name>api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|
           client[_-]?secret|app[_-]?secret|private[_-]?key|password|passwd|pwd)
        \b
        \s*['"]?\s*[:=]\s*
        ['"](?P<value>[^'"\s]{6,})['"]
        """
    ),
]

_DANGEROUS_IMPORTS = {
    "pickle": (SecuritySeverity.CRITICAL, "DANGEROUS_IMPORT_PICKLE", "导入 pickle 模块可能导致反序列化漏洞"),
    "subprocess": (SecuritySeverity.MEDIUM, "IMPORT_SUBPROCESS", "导入 subprocess 模块，标记为需要 SUBPROCESS 权限"),
    "ctypes": (SecuritySeverity.HIGH, "DANGEROUS_IMPORT", "ctypes can load shared libraries and execute arbitrary code"),
    "socket": (SecuritySeverity.MEDIUM, "DANGEROUS_IMPORT", "socket provides raw network access"),
    "code": (SecuritySeverity.HIGH, "DANGEROUS_IMPORT", "code module enables interactive interpreter"),
    "pty": (SecuritySeverity.HIGH, "DANGEROUS_IMPORT", "pty can spawn terminal processes"),
    "importlib": (SecuritySeverity.MEDIUM, "DANGEROUS_IMPORT", "importlib enables dynamic module loading"),
}

_DANGEROUS_FUNCTIONS = {
    "pickle.loads": (SecuritySeverity.CRITICAL, "PICKLE_LOADS", "调用 pickle.loads() 可能导致任意代码执行"),
    "pickle.load": (SecuritySeverity.CRITICAL, "PICKLE_LOAD", "调用 pickle.load() 可能导致任意代码执行"),
    "os.system": (SecuritySeverity.HIGH, "OS_SYSTEM", "调用 os.system() 存在命令注入风险"),
    "os.popen": (SecuritySeverity.HIGH, "OS_POPEN", "调用 os.popen() 存在命令注入风险"),
    "eval": (SecuritySeverity.CRITICAL, "EVAL", "调用 eval() 可能导致任意代码执行"),
    "exec": (SecuritySeverity.CRITICAL, "EXEC", "调用 exec() 可能导致任意代码执行"),
    "__import__": (SecuritySeverity.HIGH, "DYNAMIC_IMPORT", "使用 __import__() 动态导入模块"),
    "marshal.loads": (SecuritySeverity.CRITICAL, "MARSHAL_LOADS", "调用 marshal.loads() 可能导致任意代码执行"),
    "os.remove": (SecuritySeverity.MEDIUM, "FILE_DELETE", "os.remove deletes files"),
    "os.unlink": (SecuritySeverity.MEDIUM, "FILE_DELETE", "os.unlink deletes files"),
    "shutil.rmtree": (SecuritySeverity.HIGH, "DIRECTORY_DELETE", "shutil.rmtree deletes directory trees"),
    "os.execv": (SecuritySeverity.HIGH, "PROCESS_REPLACE", "os.execv replaces current process"),
    "os.execve": (SecuritySeverity.HIGH, "PROCESS_REPLACE", "os.execve replaces current process"),
    "os.execvp": (SecuritySeverity.HIGH, "PROCESS_REPLACE", "os.execvp replaces current process"),
    "os.execvpe": (SecuritySeverity.HIGH, "PROCESS_REPLACE", "os.execvpe replaces current process"),
    "pty.spawn": (SecuritySeverity.HIGH, "SUBPROCESS_SPAWN", "pty.spawn spawns terminal processes"),
    "code.interact": (SecuritySeverity.HIGH, "INTERACTIVE_EXEC", "code.interact starts interactive interpreter"),
    "importlib.import_module": (SecuritySeverity.MEDIUM, "DYNAMIC_IMPORT", "importlib.import_module dynamically loads modules"),
}

# subprocess 中会执行命令的函数名（导入别名后仍需按真实点分名识别）
_SUBPROCESS_EXEC_FUNCS = {
    "call", "Popen", "run", "check_output", "check_call", "getoutput"
}

_NETWORK_MODULES = {"requests", "urllib", "urllib.request", "urllib3", "aiohttp", "httpx", "socket"}
_NETWORK_FUNCTIONS = {
    "requests.get", "requests.post", "requests.put", "requests.delete",
    "requests.patch", "requests.head", "aiohttp.ClientSession.get",
    "aiohttp.ClientSession.post",
}

_FILE_WRITE_MODES = {"w", "a", "wb", "ab", "w+", "a+", "wb+", "ab+", "x", "xb", "xt", "x+", "xb+", "w+b"}

# rule_id → ExtensionPermission.value。命中这些 rule_id 的 issue 在扩展
# 已声明对应权限时可被抑制（不打印、不计入 risk_score）。
# 注意：SUBPROCESS_SHELL_TRUE（CRITICAL 命令注入）不在此列，即使声明
# SUBPROCESS 权限也不应被抑制；pickle/eval/exec/ctypes 等本身无对应权限，
# 永远上报。
_RULE_TO_PERMISSION: dict[str, str] = {
    "IMPORT_SUBPROCESS": "subprocess",
    "SUBPROCESS_CALL": "subprocess",
    "NETWORK_MODULE_DETECTED": "network",
    "NETWORK_REQUEST": "network",
    "FILE_WRITE": "filesystem_write",
    "FILE_DELETE": "filesystem_delete",
    "DIRECTORY_DELETE": "filesystem_delete",
}


def _get_suggestion_for_function(func_name: str) -> str:
    """根据函数名返回修复建议。"""
    suggestions = {
        "pickle.loads": "避免使用 pickle.loads() 反序列化不可信数据，改用 JSON 等安全格式",
        "pickle.load": "避免使用 pickle.load() 反序列化不可信数据，改用 JSON 等安全格式",
        "os.system": "避免使用 os.system()，改用 subprocess.run() 并设置 shell=False",
        "os.popen": "避免使用 os.popen()，改用 subprocess.Popen() 并设置 shell=False",
        "eval": "禁止使用 eval()，改用 ast.literal_eval() 或其他安全的解析方式",
        "exec": "禁止使用 exec()，重构代码避免动态执行",
        "__import__": "避免使用 __import__()，使用常规 import 语句",
        "marshal.loads": "避免使用 marshal.loads() 反序列化不可信数据",
    }
    return suggestions.get(func_name, "请审查该函数调用的安全性")


class ExtensionSecurityScanner:
    """扩展安全扫描器。

    使用 AST 分析和正则表达式扫描 Python 源代码，检测潜在的安全问题。
    """

    def __init__(self) -> None:
        """初始化安全扫描器。"""
        self._rules: dict[str, tuple[SecuritySeverity, str, str]] = {}
        self._compile_rules()

    def _compile_rules(self) -> None:
        """编译检测规则。"""
        self._rules.update(_DANGEROUS_IMPORTS)
        self._rules.update(_DANGEROUS_FUNCTIONS)

    def scan_file(self, file_path: Path) -> SecurityScanResult:
        """扫描单个 Python 文件。

        参数:
            file_path: 要扫描的文件路径

        返回:
            SecurityScanResult 扫描结果
        """
        result = SecurityScanResult(file_path=file_path)

        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"无法读取文件 {file_path}: {e}")
            return result

        self._scan_source(source, result, str(file_path))
        result.calculate_risk_score()
        return result

    def scan_directory(self, dir_path: Path) -> list[SecurityScanResult]:
        """扫描目录下所有 Python 文件。

        参数:
            dir_path: 要扫描的目录路径

        返回:
            SecurityScanResult 列表
        """
        results: list[SecurityScanResult] = []

        if not dir_path.is_dir():
            return results

        for py_file in sorted(dir_path.rglob("*.py")):
            # 必须扫描 __init__.py：包扩展的加载入口正是它（loader 以
            # ``spec_from_file_location(module_name, __init__.py)`` 构建模块），
            # 若按「下划线开头即跳过」过滤，恶意代码写进 __init__.py 就完全
            # 绕过静态扫描并在 exec_module 阶段先于任何校验执行。
            # 仍跳过其它私有模块（_foo.py）与 __pycache__。
            if py_file.is_file() and (
                py_file.name == "__init__.py" or not py_file.name.startswith("_")
            ):
                results.append(self.scan_file(py_file))

        return results

    def _scan_source(self, source: str, result: SecurityScanResult, file_display: str) -> None:
        """扫描源代码字符串。

        参数:
            source: Python 源代码
            result: 扫描结果对象
            file_display: 文件显示路径
        """
        self._scan_regex_patterns(source, result, file_display)

        try:
            tree = ast.parse(source, filename=str(result.file_path))
            self._scan_ast(tree, result, file_display)
        except SyntaxError as e:
            logger.debug(f"AST 解析失败 {result.file_path}: {e}")

    def _scan_regex_patterns(self, source: str, result: SecurityScanResult, file_display: str) -> None:
        """使用正则表达式扫描源代码。

        参数:
            source: Python 源代码
            result: 扫描结果对象
            file_display: 文件显示路径
        """
        lines = source.splitlines()

        for line_no, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue

            code_part = line.split("#")[0] if "#" in line else line

            for pattern in _SENSITIVE_KEY_PATTERNS:
                for match in pattern.finditer(code_part):
                    location = f"{file_display}:{line_no}"
                    result.issues.append(
                        SecurityIssue(
                            severity=SecuritySeverity.HIGH,
                            rule_id="HARDCODED_SECRET",
                            message=f"检测到可能硬编码的敏感信息: {match.group('name')}",
                            location=location,
                            suggestion="请将敏感信息迁移到环境变量、配置文件或密钥管理服务中",
                        )
                    )

    def _scan_ast(self, tree: ast.AST, result: SecurityScanResult, file_display: str) -> None:
        """使用 AST 扫描源代码。

        参数:
            tree: AST 语法树
            result: 扫描结果对象
            file_display: 文件显示路径
        """
        visitor = _SecurityASTVisitor(result, file_display)
        visitor.visit(tree)


class _SecurityASTVisitor(ast.NodeVisitor):
    """AST 访问器，用于遍历语法树检测安全问题。"""

    def __init__(self, result: SecurityScanResult, file_display: str) -> None:
        """初始化 AST 访问器。

        参数:
            result: 扫描结果对象
            file_display: 文件显示路径
        """
        self._result = result
        self._file_display = file_display
        # 导入别名表：本地绑定名 -> 真实点分名。
        # 触发条件：`import os as o` / `from os import system` /
        # `from subprocess import run as r` 等写法；若不还原，规则只匹配
        # 完整点分名（os.system），别名调用会被完全放过。
        # 后果：恶意扩展可用一行别名导入绕过全部函数级规则。
        self._aliases: dict[str, str] = {}
        self._aliases_collected = False

    def visit(self, node: ast.AST) -> None:
        """遍历入口：先收集全文件导入别名，再做规则匹配。

        触发条件：import 语句出现在调用之后（如函数内延迟导入）时，
        单次前序遍历会漏掉别名。
        后果：漏检。因此先对整棵树做一次导入别名预扫描。
        """
        if not self._aliases_collected:
            self._aliases_collected = True
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    self._record_import_aliases(child)
        super().visit(node)

    def _record_import_aliases(self, node: ast.Import | ast.ImportFrom) -> None:
        """记录 import / from-import 的本地名到真实点分名的映射。

        覆盖三种形式：``import X as a``、``from X import Y``、
        ``from X import Y as a``。相对导入（``from . import Y``）的真实
        模块名无法静态确定，跳过以避免误报。
        """
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    # import os.path as osp -> osp 绑定到 os.path
                    self._aliases[alias.asname] = alias.name
                else:
                    # import os.path 只绑定顶层名 os
                    base = alias.name.split(".")[0]
                    self._aliases.setdefault(base, base)
            return

        # ImportFrom
        if node.level or not node.module:
            return
        for alias in node.names:
            if alias.name == "*":
                continue
            local = alias.asname or alias.name
            self._aliases[local] = f"{node.module}.{alias.name}"

    def _get_location(self, node: ast.AST) -> str:
        """获取节点位置信息。

        参数:
            node: AST 节点

        返回:
            格式化的位置字符串 (file:line)
        """
        line_no = getattr(node, "lineno", 0)
        return f"{self._file_display}:{line_no}"

    def _add_issue(
        self,
        node: ast.AST,
        severity: SecuritySeverity,
        rule_id: str,
        message: str,
        suggestion: str,
    ) -> None:
        """添加安全问题。

        参数:
            node: AST 节点
            severity: 严重程度
            rule_id: 规则 ID
            message: 问题描述
            suggestion: 修复建议
        """
        self._result.issues.append(
            SecurityIssue(
                severity=severity,
                rule_id=rule_id,
                message=message,
                location=self._get_location(node),
                suggestion=suggestion,
            )
        )

    def _get_full_name(self, node: ast.AST) -> str | None:
        """获取属性访问的完整名称（已按导入别名还原）。

        参数:
            node: AST 节点

        返回:
            完整名称或 None
        """
        if isinstance(node, ast.Name):
            return self._resolve_alias(node.id)
        if isinstance(node, ast.Attribute):
            parts: list[str] = []
            current: ast.AST | None = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return self._resolve_alias(".".join(reversed(parts)))
        return None

    def _resolve_alias(self, dotted_name: str) -> str:
        """把点分名的首段按导入别名还原为真实模块名。

        例如别名表 ``{"o": "os"}`` 时 ``o.system`` -> ``os.system``。
        """
        head, sep, rest = dotted_name.partition(".")
        real = self._aliases.get(head)
        if not real:
            return dotted_name
        return f"{real}.{rest}" if sep else real

    def _get_value_name(self, node: ast.AST) -> str | None:
        """解析表达式对应的真实点分名（支持嵌套 getattr 字符串访问）。"""
        name = self._get_full_name(node)
        if name is not None:
            return name
        if isinstance(node, ast.Call):
            return self._resolve_getattr_target(node)
        return None

    def _resolve_getattr_target(self, call: ast.Call) -> str | None:
        """解析 ``getattr(os, "system")`` 形式的字符串属性访问目标名。

        触发条件：函数名为 getattr 且前两个参数可静态确定为
        「模块名 + 字符串属性」。
        后果：若不解析，``getattr(os, "system")(cmd)`` 不产生任何告警。
        """
        if self._get_full_name(call.func) not in {"getattr", "builtins.getattr"}:
            return None
        if len(call.args) < 2:
            return None
        base = self._get_value_name(call.args[0])
        attr = self._get_string_value(call.args[1])
        if base is None or attr is None or not attr.isidentifier():
            return None
        return f"{base}.{attr}"

    def _report_dynamic_attribute(self, node: ast.Call, target: str) -> None:
        """对命中危险规则的字符串属性访问（getattr）给出保守告警。

        rule_id 统一加 ``GETATTR_`` 前缀，避免被扩展声明的权限直接抑制
        （声明 SUBPROCESS 权限不应豁免动态取命令执行函数）。
        """
        normalized = target[len("builtins."):] if target.startswith("builtins.") else target

        rule = _DANGEROUS_FUNCTIONS.get(normalized)
        if (
            rule is None
            and normalized.startswith("subprocess.")
            and normalized.split(".")[-1] in _SUBPROCESS_EXEC_FUNCS
        ):
            rule = (
                SecuritySeverity.HIGH,
                "SUBPROCESS_DYNAMIC_CALL",
                f"通过 getattr 动态获取 {normalized} 执行子进程",
            )
        if rule is None:
            return

        severity, rule_id, message = rule
        self._add_issue(
            node,
            severity,
            f"GETATTR_{rule_id}",
            f"通过字符串属性访问获取危险函数: {message}",
            _get_suggestion_for_function(normalized),
        )

    def _has_keyword_arg(self, node: ast.Call, arg_name: str, value: Any) -> bool:
        """检查函数调用是否包含指定的关键字参数。

        参数:
            node: Call 节点
            arg_name: 参数名
            value: 期望值

        返回:
            是否存在该关键字参数且值匹配
        """
        for keyword in node.keywords:
            if keyword.arg == arg_name and isinstance(keyword.value, ast.Constant):
                return bool(keyword.value.value == value)
        return False

    def _get_string_value(self, node: ast.AST) -> str | None:
        """获取 AST 节点的字符串值。

        参数:
            node: AST 节点

        返回:
            字符串值或 None
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def visit_Import(self, node: ast.Import) -> None:
        """处理 import 语句。

        参数:
            node: Import 节点
        """
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            if module_name in _DANGEROUS_IMPORTS:
                severity, rule_id, message = _DANGEROUS_IMPORTS[module_name]
                self._add_issue(
                    node,
                    severity,
                    rule_id,
                    message,
                    f"请审查是否真的需要导入 {module_name}，并确保安全使用",
                )
            if module_name in _NETWORK_MODULES:
                self._result.required_permissions.add("NETWORK")
                self._add_issue(
                    node,
                    SecuritySeverity.LOW,
                    "NETWORK_MODULE_DETECTED",
                    f"检测到网络模块导入: {module_name}",
                    "请确保扩展已声明 NETWORK 权限",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """处理 from ... import ... 语句。

        参数:
            node: ImportFrom 节点
        """
        module_name = node.module or ""
        base_module = module_name.split(".")[0]

        if base_module == "pickle":
            for alias in node.names:
                if alias.name in {"load", "loads"}:
                    self._add_issue(
                        node,
                        SecuritySeverity.CRITICAL,
                        f"DANGEROUS_IMPORT_PICKLE_{alias.name.upper()}",
                        f"从 pickle 导入 {alias.name} 可能导致反序列化漏洞",
                        "避免使用 pickle 反序列化不可信数据",
                    )
                else:
                    self._add_issue(
                        node,
                        SecuritySeverity.HIGH,
                        "DANGEROUS_IMPORT_PICKLE",
                        "从 pickle 导入模块可能导致反序列化漏洞",
                        "请审查是否真的需要使用 pickle",
                    )
        if base_module in _NETWORK_MODULES:
            self._result.required_permissions.add("NETWORK")
            self._add_issue(
                node,
                SecuritySeverity.LOW,
                "NETWORK_MODULE_DETECTED",
                f"检测到网络模块导入: {base_module}",
                "请确保扩展已声明 NETWORK 权限",
            )
        if base_module == "subprocess":
            self._result.required_permissions.add("SUBPROCESS")
            self._add_issue(
                node,
                SecuritySeverity.MEDIUM,
                "IMPORT_SUBPROCESS",
                "导入 subprocess 模块，标记为需要 SUBPROCESS 权限",
                "请确保扩展已声明 SUBPROCESS 权限并安全使用",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """处理函数调用。

        参数:
            node: Call 节点
        """
        func_name = self._get_full_name(node.func)

        # getattr 字符串属性访问：`getattr(os, "system")(cmd)` 这类写法
        # 的函数名无法静态解析为 os.system，必须单独做保守告警。
        getattr_target = self._resolve_getattr_target(node)
        if getattr_target is not None:
            self._report_dynamic_attribute(node, getattr_target)

        if func_name:
            if func_name in _DANGEROUS_FUNCTIONS:
                severity, rule_id, message = _DANGEROUS_FUNCTIONS[func_name]
                suggestion = _get_suggestion_for_function(func_name)
                self._add_issue(node, severity, rule_id, message, suggestion)

            if func_name.startswith("subprocess.") and func_name.split(".")[-1] in _SUBPROCESS_EXEC_FUNCS:
                self._result.required_permissions.add("SUBPROCESS")
                has_shell_true = self._has_keyword_arg(node, "shell", True)
                if has_shell_true:
                    self._add_issue(
                        node,
                        SecuritySeverity.CRITICAL,
                        "SUBPROCESS_SHELL_TRUE",
                        f"调用 {func_name} 时使用 shell=True，存在命令注入风险",
                        "请使用 shell=False 并以列表形式传递参数，避免命令注入",
                    )
                else:
                    self._add_issue(
                        node,
                        SecuritySeverity.MEDIUM,
                        "SUBPROCESS_CALL",
                        f"调用 {func_name} 执行子进程",
                        "请确保输入参数经过验证，避免命令注入",
                    )

            if func_name in _NETWORK_FUNCTIONS or func_name.startswith(("urllib.", "aiohttp.")):
                self._result.required_permissions.add("NETWORK")
                self._add_issue(
                    node,
                    SecuritySeverity.LOW,
                    "NETWORK_REQUEST",
                    f"检测到网络请求: {func_name}",
                    "请确保扩展已声明 NETWORK 权限",
                )

            if func_name == "yaml.load" or func_name == "yaml.load_all":
                has_loader = any(kw.arg == "Loader" for kw in node.keywords)
                if not has_loader:
                    self._add_issue(
                        node,
                        SecuritySeverity.HIGH,
                        "UNSAFE_YAML_LOAD",
                        f"调用 {func_name}() 时未指定 Loader 参数，存在反序列化风险",
                        "请使用 yaml.safe_load() 或显式指定 Loader=yaml.SafeLoader",
                    )

            if func_name == "open" and len(node.args) >= 2:
                mode = self._get_string_value(node.args[1])
                if mode and mode in _FILE_WRITE_MODES:
                    self._result.required_permissions.add("FILESYSTEM_WRITE")
                    self._add_issue(
                        node,
                        SecuritySeverity.MEDIUM,
                        "FILE_WRITE",
                        f"检测到文件写入操作: open(..., mode='{mode}')",
                        "请确保扩展已声明 FILESYSTEM_WRITE 权限",
                    )

        self.generic_visit(node)


__all__ = [
    "SecuritySeverity",
    "SecurityIssue",
    "SecurityScanResult",
    "ExtensionSecurityScanner",
]
