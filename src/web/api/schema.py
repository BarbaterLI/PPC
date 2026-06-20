"""统一的 API 响应契约。

所有 Web API Blueprint 的 JSON 响应都应通过 :func:`success_response` 或
:func:`error_response` 包装为 :class:`ApiResponse` 形状，确保前端可以按
统一格式解析。
"""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel


class ApiErrorDetail(BaseModel):
    """API 错误详情。"""

    message: str
    code: str | None = None


T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """统一 API 响应包装器。

    Args:
        success: 请求是否成功。
        data: 成功时的业务数据。
        error: 失败时的错误详情。
    """

    success: bool
    data: T | None = None
    error: ApiErrorDetail | None = None


def success_response(data: Any, status_code: int = 200) -> tuple[dict, int]:
    """构造成功响应。

    Returns:
        (response_body, http_status_code)
    """
    return ApiResponse[Any](success=True, data=data, error=None).model_dump(), status_code


def error_response(
    message: str,
    code: str | None = None,
    status_code: int = 500,
) -> tuple[dict, int]:
    """构造错误响应。

    Returns:
        (response_body, http_status_code)
    """
    return (
        ApiResponse[Any](
            success=False,
            data=None,
            error=ApiErrorDetail(message=message, code=code),
        ).model_dump(),
        status_code,
    )
