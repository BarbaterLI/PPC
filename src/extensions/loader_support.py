"""扩展加载辅助纯函数：entry_point 解析、requirements 检测、字典深合并。"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path

from src.utils.core import deep_merge as _deep_merge

logger = logging.getLogger(__name__)


def _resolve_entry_point(entry_point: str, module) -> type:
    """解析 ``"module:ClassName"`` 格式的 entry_point 并返回对应的类。

    参数:
        entry_point: ``"module:ClassName"`` 格式的字符串（如 ``"extension:ExampleExtension"``）
        module: 已加载的模块对象，当 module 部分为 ``"extension"`` 或与已加载模块名
            匹配时直接从中 getattr，避免重复加载

    返回:
        解析到的类对象（未实例化，调用方负责实例化与 Extension 子类校验）

    异常:
        ImportError: entry_point 格式错误（缺少冒号、module/ClassName 为空）
            或类在指定模块中不存在时抛出
    """
    if ":" not in entry_point:
        raise ImportError(
            f"entry_point must be in 'module:ClassName' format, got: {entry_point}"
        )

    module_name_str, _, class_name = entry_point.partition(":")
    if not module_name_str:
        raise ImportError(
            f"entry_point must be in 'module:ClassName' format, got: {entry_point} "
            f"(empty module name)"
        )
    if not class_name:
        raise ImportError(
            f"entry_point must be in 'module:ClassName' format, got: {entry_point} "
            f"(empty class name)"
        )

    # "extension" 是约定关键字，指代当前已加载的扩展模块（extension.py / __init__.py）；
    # 若显式指定模块名且与已加载模块名一致，同样复用，避免重复 exec。
    loaded_module_name = getattr(module, "__name__", None)
    if module_name_str == "extension" or module_name_str == loaded_module_name:
        target_module = module
    else:
        target_module = importlib.import_module(module_name_str)

    cls = getattr(target_module, class_name, None)
    if cls is None:
        raise ImportError(
            f"Class '{class_name}' not found in module '{module_name_str}'"
        )

    return cls


def _parse_requirement_name(line: str) -> str:
    """从 requirements.txt 的一行中解析出包名。

    处理形如 ``package==1.0`` / ``package>=2.0`` / ``package`` /
    ``package-name~=1.0`` / ``package; python_version < "3.10"`` 的格式，
    返回版本限定符与环境标记之前的包名（去除空白）。
    空行或以 ``#`` 开头的注释行返回空字符串。
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return ""
    # 去除行内注释（requirements.txt 支持行内 " # ..." 注释）
    inline_comment = line.find(" #")
    if inline_comment != -1:
        line = line[:inline_comment].strip()
    if not line:
        return ""
    # 取版本限定符（=, <, >, !, ~）与环境标记（;）之前的部分
    name = line
    for sep in ("=", "<", ">", "!", "~", ";"):
        idx = name.find(sep)
        if idx != -1:
            name = name[:idx]
    return name.strip()


def _check_requirements(extension_dir: Path, ext_name: str) -> None:
    """检查扩展依赖的 Python 包是否已安装。

    读取 ``extension_dir/requirements.txt``，逐行解析包名（处理版本限定符），
    使用 :func:`importlib.util.find_spec` 检测是否已安装；未安装时记录 WARNING
    并提示 ``pip install -r`` 命令。仅作提示，不阻塞加载（缺失的包会在后续
    import 时通过 try/except 自然失败）。

    参数:
        extension_dir: 扩展目录（包含 requirements.txt 的目录）
        ext_name: 扩展名称（用于日志提示）
    """
    req_file = extension_dir / "requirements.txt"
    if not req_file.is_file():
        return

    try:
        content = req_file.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning(
            f"Failed to read requirements.txt for extension '{ext_name}': {exc}"
        )
        return

    for line in content.splitlines():
        pkg_name = _parse_requirement_name(line)
        if not pkg_name:
            continue
        # find_spec 接受模块名（下划线分隔），而 pip 包名可能含连字符
        spec_name = pkg_name.replace("-", "_")
        if importlib.util.find_spec(spec_name) is None:
            logger.warning(
                f"Extension '{ext_name}' requires '{pkg_name}' but it's not installed. "
                f"Run: pip install -r {req_file}"
            )



