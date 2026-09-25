"""Windows UAC elevation utility.

当程序遇到 Windows 权限问题(winerror 1313 / 5 / PermissionError)时,
通过 UAC 重新启动当前进程以获取管理员权限。

典型用法::

    from src.utils.elevation import handle_permission_error_or_elevate

    try:
        do_something()
    except (PermissionError, OSError) as exc:
        if handle_permission_error_or_elevate(exc):
            sys.exit(0)  # 已发起提权,原进程退出
        raise
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import sys
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Windows 系统错误码
ERROR_ACCESS_DENIED = 5
ERROR_NO_SUCH_PRIVILEGE = 1313  # "win13" 权限问题: 指定的特权不存在

# errno 13 = Permission denied (POSIX)
_EACCES = 13

# ShellExecuteW 返回值 <= 32 表示错误
_SE_ERROR_THRESHOLD = 32

# Spec unify-log-and-cli-ui Task 5: 优先使用全局 console 输出权限提示,
# 使颜色受 --no-color 控制。
#
# 注意:src.utils.__init__ 在加载时即 from .elevation import ...,
# 而 src.cli.output 又 from src.utils.format import format_duration —— 这形成
# 循环导入,utils 层不能直接依赖 cli 层。
# 因此改用 console 工厂注入:由 src.cli.output.setup_console() 在初始化全局
# console 后调用 set_console_factory() 注入,本模块保持对 cli 层零依赖;
# 未注入(如非 CLI 场景)时回退到原始 print,保证输出仍可读。
_console_factory: Callable[[], Any] | None = None


def set_console_factory(factory: Callable[[], Any] | None) -> None:
    """注入 console 工厂。由 CLI 层(src.cli.output)在初始化 console 后调用。"""
    global _console_factory
    _console_factory = factory


def _get_console() -> Any:
    """通过注入的工厂获取全局 console;未注入或工厂异常时返回 None。"""
    factory = _console_factory
    if factory is None:
        return None
    try:
        return factory()
    except Exception:  # noqa: BLE001 - 故意宽泛,确保任何工厂异常都安全降级
        return None


def is_windows() -> bool:
    """是否运行在 Windows 平台。"""
    return platform.system() == "Windows"


def is_admin() -> bool:
    """检查当前进程是否拥有管理员权限。

    Windows 通过 ``shell32.IsUserAnAdmin`` 判断;非 Windows 平台
    通过 ``os.geteuid() == 0`` 判断。
    """
    if not is_windows():
        return hasattr(os, "geteuid") and os.geteuid() == 0
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception as exc:
        logger.debug("IsUserAnAdmin 调用失败: %s", exc)
        return False


def is_permission_error(exc: BaseException) -> bool:
    """判断异常是否为 Windows 权限问题。

    覆盖以下情况:

    - :class:`PermissionError`
    - :class:`OSError` 且 ``winerror`` 为 5(ACCESS_DENIED) 或 1313(NO_SUCH_PRIVILEGE)
    - :class:`OSError` 且 ``errno`` 为 13(EACCES)
    """
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, OSError):
        winerror = getattr(exc, "winerror", None)
        if winerror in (ERROR_ACCESS_DENIED, ERROR_NO_SUCH_PRIVILEGE):
            return True
        if getattr(exc, "errno", None) == _EACCES:
            return True
    return False


def _build_relaunch_params() -> tuple[str, str]:
    """构建提权重启所需的 (executable, parameters)。

    自动适配两种运行形态:

    - 脚本模式: ``python.exe "ppc.py" --args``
    - 编译产物: ``ppc.exe --args``
    """
    argv0 = sys.argv[0] or ""
    quoted_args = " ".join(f'"{a}"' for a in sys.argv[1:])

    if argv0.lower().endswith(".py"):
        # 脚本模式: python.exe "script.py" args
        executable = sys.executable
        params = f'"{argv0}"'
        if quoted_args:
            params = f"{params} {quoted_args}"
    else:
        # 编译产物模式: ppc.exe args
        executable = sys.executable
        params = quoted_args

    return executable, params


def elevate() -> int:
    """通过 UAC 重新启动当前进程(以管理员权限)。

    Returns
    -------
    int
        退出码:

        - ``0``  已成功发起提权,调用方应退出原进程
        - ``1``  非 Windows 平台
        - ``2``  已经是管理员权限,无需提权
        - ``3``  ShellExecuteW 返回错误(用户拒绝 UAC / 其他失败)
        - ``4``  调用异常
    """
    if not is_windows():
        logger.error("UAC 提权仅支持 Windows 平台")
        return 1

    if is_admin():
        logger.debug("已经是管理员权限,无需提权")
        return 2

    executable, params = _build_relaunch_params()
    cwd = os.getcwd()

    logger.info("发起 UAC 提权: executable=%s params=%s cwd=%s", executable, params, cwd)
    try:
        # ShellExecuteW(hwnd, lpOperation, lpFile, lpParameters, lpDirectory, nShowCmd)
        # "runas" 动词触发 UAC 提权对话框
        # SW_SHOWNORMAL = 1
        hinst = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", executable, params, cwd, 1
        )
        if hinst is None or (isinstance(hinst, int) and hinst <= _SE_ERROR_THRESHOLD):
            logger.error("UAC 提权失败,ShellExecuteW 返回: %s", hinst)
            return 3
        return 0
    except Exception as exc:
        logger.error("UAC 提权异常: %s", exc)
        return 4


def _build_command_hint() -> str:
    """构建提示用户重新运行的命令字符串。"""
    argv0 = sys.argv[0] or "ppc.py"
    args = " ".join(sys.argv[1:])
    if argv0.lower().endswith(".py"):
        return f'python "{argv0}" {args}'
    return f'"{argv0}" {args}'


def _emit_message(msg: str, *, style: str | None = None) -> None:
    """输出单行状态消息,优先走全局 console,否则 fallback 到 print。

    Spec unify-log-and-cli-ui Task 5: 替代 ``handle_permission_error_or_elevate``
    中直接使用的 ``print()``。
    """
    console = _get_console()
    if console is not None:
        console.print(msg, style=style)
    else:
        print(msg)


def _print_permission_hint(exc: BaseException) -> None:
    """打印权限问题的友好提示，建议用户以管理员权限重新运行。

    优先使用注入的全局 ``console``(Spec unify-log-and-cli-ui Task 5):颜色受
    ``--no-color`` 控制;若未注入 console 工厂(非 CLI 场景)则 fallback 到
    原始 ``print()``。
    """
    cmd = _build_command_hint()
    console = _get_console()

    if console is not None:
        console.print()
        console.print("=" * 60, style="red")
        console.print("[ppc] 保存文件时遇到权限问题 / Permission denied", style="bold red")
        console.print(f"  错误: {exc}")
        console.print()
        if is_windows():
            console.print('请使用管理员权限的 CMD 或 PowerShell 重新运行:')
            console.print(f'  1. 右键点击 CMD/PowerShell → "以管理员身份运行"')
            console.print(f"  2. 执行: {cmd}")
        else:
            console.print("请使用 sudo 重新运行:")
            console.print(f"  sudo {cmd}")
        console.print("=" * 60, style="red")
        return

    # Fallback: output 模块不可用时使用原始 print
    if is_windows():
        print()
        print("=" * 60)
        print("[ppc] 保存文件时遇到权限问题 / Permission denied")
        print(f"  错误: {exc}")
        print()
        print("请使用管理员权限的 CMD 或 PowerShell 重新运行:")
        print(f'  1. 右键点击 CMD/PowerShell → "以管理员身份运行"')
        print(f"  2. 执行: {cmd}")
        print("=" * 60)
    else:
        print()
        print("=" * 60)
        print("[ppc] 保存文件时遇到权限问题 / Permission denied")
        print(f"  错误: {exc}")
        print()
        print("请使用 sudo 重新运行:")
        print(f"  sudo {cmd}")
        print("=" * 60)


def handle_permission_error_or_elevate(exc: BaseException) -> bool:
    """遇到权限错误时尝试 UAC 提权重启。

    若当前已是管理员权限却仍报权限错误(例如文件被占用、ACL 限制),
    则不再尝试提权,返回 ``False`` 让调用方按正常错误处理。

    Parameters
    ----------
    exc : BaseException
        捕获到的异常。

    Returns
    -------
    bool
        ``True`` 表示已发起 UAC 提权(调用方应让原进程退出);
        ``False`` 表示未提权(调用方应继续按原逻辑处理错误)。
    """
    if not is_permission_error(exc):
        return False

    if not is_windows():
        # 非 Windows 平台: 提示用户使用 sudo
        _print_permission_hint(exc)
        return False

    if is_admin():
        # 已经是管理员却仍权限不足,提权无济于事
        logger.warning("当前已是管理员权限但仍遇到权限错误,不再尝试 UAC 提权: %s", exc)
        _print_permission_hint(exc)
        return False

    _emit_message(f"[ppc] 检测到 Windows 权限问题 ({exc}),正在尝试 UAC 提权重启...")
    rc = elevate()
    if rc == 0:
        _emit_message("[ppc] UAC 提权请求已发起,请在弹出的 UAC 对话框中确认。")
        return True
    if rc == 2:
        # 已经是管理员,不应走到这里,但兜底返回 False
        return False
    # UAC 提权失败(用户拒绝等),打印详细提示
    _print_permission_hint(exc)
    return False


__all__ = [
    "is_windows",
    "is_admin",
    "is_permission_error",
    "elevate",
    "handle_permission_error_or_elevate",
    "set_console_factory",
    "ERROR_ACCESS_DENIED",
    "ERROR_NO_SUCH_PRIVILEGE",
]
