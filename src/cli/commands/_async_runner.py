"""异步命令统一执行器 - 集中 KeyboardInterrupt/CLIError/Exception 处理模板。"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from ...core.exceptions import CLIError
from ...core.exceptions import ErrorCodes as E
from ...core.exceptions import PPCTimeoutError


def run_cli_async(
    coro: Awaitable[Any],
    *,
    interrupt_message: str = "用户中断操作 (Ctrl+C)",
    on_interrupt: Callable[[], None] | None = None,
    failure_message: str | None = None,
    failure_hint: str | None = None,
    failure_exit_code: int | None = None,
    generic_message: str = "执行失败",
    generic_hint: str = "使用 --verbose 查看详细错误信息",
    catch_io_errors: bool = False,
    timeout_hint: str = "可尝试 --timeout-multiplier 增大超时时间，或减少并发数",
    on_result: Callable[[Any], Any] | None = None,
) -> Any:
    """执行异步协程并统一转换异常为 CLIError。

    - KeyboardInterrupt → CLIError(退出码 130)；指定 on_interrupt 时先执行；
    - CLIError 原样重抛；
    - OSError / PPCTimeoutError 仅在 catch_io_errors=True 时转换为 CLIError；
    - 其他异常 → CLIError(f"{generic_message}: {e}")；
    - 正常完成后先应用 on_result(若指定)，再在 failure_message 设置且
      返回值为假时抛 CLIError(failure_message)。
    """
    try:
        result = asyncio.run(coro)
    except KeyboardInterrupt:
        if on_interrupt is not None:
            on_interrupt()
        raise CLIError(E.E_BUSINESS, interrupt_message, exit_code=130) from None
    except CLIError:
        raise
    except OSError as e:
        if not catch_io_errors:
            raise
        raise CLIError(
            E.E_BUSINESS,
            f"文件操作失败: {e}",
            hint="请检查文件路径、权限或磁盘空间",
        ) from e
    except PPCTimeoutError as e:
        if not catch_io_errors:
            raise
        raise CLIError(E.E_BUSINESS, f"操作超时: {e}", hint=timeout_hint) from e
    except Exception as e:
        raise CLIError(E.E_BUSINESS, f"{generic_message}: {e}", hint=generic_hint) from e

    if on_result is not None:
        result = on_result(result)

    if failure_message is not None and not result:
        if failure_exit_code is not None:
            raise CLIError(E.E_BUSINESS, failure_message, exit_code=failure_exit_code)
        raise CLIError(E.E_BUSINESS, failure_message, hint=failure_hint)
    return result
