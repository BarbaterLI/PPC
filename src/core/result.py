from collections.abc import Callable
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E", bound=Exception)


class ResultState(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


class ExecutionMetrics:
    __slots__ = (
        "start_time",
        "end_time",
        "duration",
        "memory_peak_kb",
        "bytes_processed",
        "request_count",
        "retry_count",
        "extra",
    )

    def __init__(
        self,
        start_time: float = 0.0,
        end_time: float = 0.0,
        duration: float = 0.0,
        memory_peak_kb: int = 0,
        bytes_processed: int = 0,
        request_count: int = 0,
        retry_count: int = 0,
        extra: dict[str, Any] | None = None,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.memory_peak_kb = memory_peak_kb
        self.bytes_processed = bytes_processed
        self.request_count = request_count
        self.retry_count = retry_count
        self.extra = extra or {}

    def to_dict(self) -> dict:
        return {
            "duration": self.duration,
            "memory_peak_kb": self.memory_peak_kb,
            "bytes_processed": self.bytes_processed,
            "request_count": self.request_count,
            "retry_count": self.retry_count,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionMetrics":
        return cls(
            duration=data.get("duration", 0.0),
            memory_peak_kb=data.get("memory_peak_kb", 0),
            bytes_processed=data.get("bytes_processed", 0),
            request_count=data.get("request_count", 0),
            retry_count=data.get("retry_count", 0),
            extra={
                k: v
                for k, v in data.items()
                if k not in ("duration", "memory_peak_kb", "bytes_processed", "request_count", "retry_count")
            },
        )


class Result(Generic[T]):
    def __init__(
        self,
        state: ResultState = ResultState.SUCCESS,
        value: T | None = None,
        error: str | None = None,
        error_code: str | None = None,
        errors: list[str] | None = None,
        metrics: ExecutionMetrics | None = None,
    ):
        self.state = state
        self.value = value
        self.error = error
        self.error_code = error_code
        self.errors = errors or []
        self.metrics = metrics
        self.success = state != ResultState.FAILURE

    @property
    def is_success(self) -> bool:
        return self.state == ResultState.SUCCESS

    @property
    def is_failure(self) -> bool:
        return self.state == ResultState.FAILURE

    @property
    def is_partial(self) -> bool:
        return self.state == ResultState.PARTIAL

    @property
    def data(self) -> T | None:
        return self.value

    @classmethod
    def ok(cls, value: T | None = None, metrics: ExecutionMetrics | None = None) -> "Result[T]":
        return cls(state=ResultState.SUCCESS, value=value, metrics=metrics)

    @classmethod
    def fail(
        cls, error: str = "", error_code: str | None = None, metrics: ExecutionMetrics | None = None
    ) -> "Result[T]":
        return cls(state=ResultState.FAILURE, error=error, error_code=error_code, metrics=metrics)

    @classmethod
    def partial(
        cls, data: T | None = None, errors: list[str] | None = None, metrics: ExecutionMetrics | None = None
    ) -> "Result[T]":
        return cls(state=ResultState.PARTIAL, value=data, errors=errors or [], metrics=metrics)

    def unwrap(self) -> T:
        if self.state == ResultState.FAILURE:
            raise ValueError(self.error or "unwrap on failure")
        return self.value  # type: ignore

    def unwrap_or(self, default: T) -> T:
        if self.state == ResultState.FAILURE:
            return default
        return self.value  # type: ignore

    def expect(self, message: str) -> T:
        if self.state == ResultState.FAILURE:
            raise ValueError(f"{message}: {self.error}")
        return self.value  # type: ignore

    def map(self, func: Callable[[T], Any]) -> "Result[Any]":
        if self.state != ResultState.FAILURE:
            return Result.ok(func(self.value), self.metrics)  # type: ignore[arg-type]  # value 在运行时非 None
        return Result.fail(self.error or "", self.error_code, self.metrics)

    def map_err(self, func: Callable[[Any], Any]) -> "Result[T]":
        if self.state != ResultState.FAILURE:
            return Result.ok(self.value, self.metrics)  # type: ignore
        return Result.fail(func(self.error), self.error_code, self.metrics)

    def and_then(self, func: Callable[[T], "Result"]) -> "Result":
        if self.state != ResultState.FAILURE:
            return func(self.value)  # type: ignore
        return Result.fail(self.error or "", self.error_code, self.metrics)

    def or_else(self, func: Callable[..., "Result"]) -> "Result":
        if self.state != ResultState.FAILURE:
            return Result.ok(self.value, self.metrics)  # type: ignore
        return func(self.error)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "success": self.success,
            "partial": self.is_partial,
            "data": self.value,
            "error": self.error,
            "error_code": self.error_code,
            "errors": self.errors,
        }
        if self.metrics:
            result["metrics"] = self.metrics.to_dict()
        return result

    def __repr__(self) -> str:
        if self.state == ResultState.SUCCESS:
            return f"Ok({self.value!r})"
        if self.state == ResultState.PARTIAL:
            return f"Partial({self.value!r}, errors={self.errors!r})"
        return f"Err({self.error!r})"


class Ok(Result[T]):
    def __init__(self, value: T):
        super().__init__(state=ResultState.SUCCESS, value=value)


class Err(Result[T]):
    def __init__(self, error: Any = None):
        super().__init__(state=ResultState.FAILURE, error=str(error) if error is not None else "")


ExecutionResult = Result


def is_ok(result: Result) -> bool:
    return result.is_success


def is_err(result: Result) -> bool:
    return result.is_failure
