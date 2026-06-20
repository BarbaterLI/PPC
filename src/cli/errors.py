"""统一错误类型与退出码规约 - PPC10 CLI.

定义 :class:`CLIError` 与 :class:`ErrorCode`,用于在所有子命令
中以统一格式表达错误,被 :class:`OutputFormatter` 渲染为

    [ERROR] <CODE>  <message>
    Hint:  <next-step>

并在 ``--verbose`` 模式下追加 stack。

退出码规约(详见 ``docs/exit-codes.md``):

=====  =================================
Code  含义
=====  =================================
0     成功 / 正常工作(即使"无操作")
1     业务错误(TTS 调用失败、文件 IO)
2     参数错误(输入目录不存在等)
3     网络 / 外部依赖错误(edge-tts)
4     权限错误(无法写输出目录等)
5     配置错误(配置缺失/无法解析)
=====  =================================
"""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """CLI 错误代码枚举。

    每个值既是枚举名,也是 str,可直接用于日志/JSON 序列化。
    mvp-cleanup:6 个语义码;``E_DEPENDENCY`` / ``E_INTERNAL`` /
    ``E_PERMISSION`` 已合并到 :attr:`E_BUSINESS`。
    """

    E_INPUT_NOT_FOUND = "E_INPUT_NOT_FOUND"
    E_INPUT_EMPTY = "E_INPUT_EMPTY"
    E_CONFIG_MISSING = "E_CONFIG_MISSING"
    E_CONFIG_INVALID = "E_CONFIG_INVALID"
    E_NETWORK = "E_NETWORK"
    E_BUSINESS = "E_BUSINESS"


# 各错误码的默认退出码
_DEFAULT_EXIT_CODES: dict[ErrorCode, int] = {
    ErrorCode.E_INPUT_NOT_FOUND: 2,
    ErrorCode.E_INPUT_EMPTY: 2,
    ErrorCode.E_CONFIG_MISSING: 5,
    ErrorCode.E_CONFIG_INVALID: 5,
    ErrorCode.E_NETWORK: 3,
    ErrorCode.E_BUSINESS: 1,
}


def default_exit_code(code: ErrorCode) -> int:
    """根据 :class:`ErrorCode` 返回默认退出码。"""
    return _DEFAULT_EXIT_CODES.get(code, 1)


class CLIError(Exception):
    """统一的 CLI 业务异常。

    Parameters
    ----------
    code : ErrorCode
        错误代码。
    message : str
        给用户看的错误描述(单行,无换行)。
    hint : Optional[str]
        给用户的下一步操作建议(可省略)。
    exit_code : Optional[int]
        自定义退出码;省略时按 ``code`` 取默认值。
    cause : Optional[BaseException]
        原始异常(可选,仅在 ``--verbose`` 模式下渲染 traceback)。
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        hint: str | None = None,
        exit_code: int | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.code: ErrorCode = code
        self.message: str = message
        self.hint: str | None = hint
        self.exit_code: int = int(exit_code) if exit_code is not None else default_exit_code(code)
        if cause is not None:
            self.__cause__ = cause
        super().__init__(self.__str__())

    def to_dict(self) -> dict:
        """序列化为 dict(用于 JSON 输出)。"""
        return {
            "code": self.code.value,
            "message": self.message,
            "hint": self.hint,
            "exit_code": self.exit_code,
        }

    def __str__(self) -> str:  # noqa: D401
        return f"[{self.code.value}] {self.message}"


__all__ = [
    "ErrorCode",
    "CLIError",
    "default_exit_code",
]
