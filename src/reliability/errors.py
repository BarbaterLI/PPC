"""Error system

Provides domain-specific error types, error chain handling, and structured error information.
"""

from typing import Any


class MaxRetriesError(Exception):
    """Maximum retry attempts exceeded error"""

    def __init__(self, message: str, attempts: int = 0, last_error: Exception | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error

    def __str__(self) -> str:
        base = super().__str__()
        if self.last_error:
            return f"{base} (last error: {self.last_error})"
        return base


class CircuitBreakerError(Exception):
    """Circuit breaker open error"""

    def __init__(
        self,
        message: str,
        breaker_name: str = "",
        state: str = "open",
        failure_count: int = 0,
    ):
        super().__init__(message)
        self.breaker_name = breaker_name
        self.state = state
        self.failure_count = failure_count

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (breaker: {self.breaker_name}, state: {self.state})"


class ResourceExhaustedError(Exception):
    """Resource pool exhausted error"""

    def __init__(self, message: str, resource_type: str = "", available: int = 0):
        super().__init__(message)
        self.resource_type = resource_type
        self.available = available

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (resource: {self.resource_type}, available: {self.available})"


class OperationCancelledError(Exception):
    """Operation cancelled error"""

    def __init__(self, message: str = "Operation cancelled", reason: str = ""):
        super().__init__(message)
        self.reason = reason

    def __str__(self) -> str:
        base = super().__str__()
        if self.reason:
            return f"{base} (reason: {self.reason})"
        return base


class DeadlineExceededError(Exception):
    """Deadline exceeded error"""

    def __init__(
        self,
        message: str,
        deadline: float = 0.0,
        actual_duration: float = 0.0,
    ):
        super().__init__(message)
        self.deadline = deadline
        self.actual_duration = actual_duration

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (deadline: {self.deadline:.2f}s, actual: {self.actual_duration:.2f}s)"


def create_error_from_exception(
    exc: Exception,
    default_message: str = "An error occurred",
) -> Exception:
    """Create domain error from exception

    Args:
        exc: Original exception
        default_message: Default error message

    Returns:
        Appropriate domain error
    """
    error_message = str(exc) or default_message

    error_type = type(exc).__name__.lower()

    if "circuit" in error_type or "breaker" in error_type:
        return CircuitBreakerError(error_message)
    elif "timeout" in error_type or "deadline" in error_type:
        return DeadlineExceededError(error_message)
    elif "cancel" in error_type:
        return OperationCancelledError(error_message)
    elif "resource" in error_type or "pool" in error_type or "exhausted" in error_type:
        return ResourceExhaustedError(error_message)
    elif "retry" in error_type or "max" in error_type:
        return MaxRetriesError(error_message)

    return exc


def create_exception_chain(
    errors: list[Exception],
    message: str = "Multiple errors occurred",
) -> Exception:
    """Create exception chain

    Args:
        errors: List of exceptions
        message: Error message

    Returns:
        Exception with chain information
    """
    if not errors:
        return Exception(message)
    if len(errors) == 1:
        return errors[0]

    chain_info = f"{message}\nError chain:"
    for i, error in enumerate(errors, 1):
        error_type = type(error).__name__
        error_msg = str(error)
        chain_info += f"\n  {i}. {error_type}: {error_msg}"

    return Exception(chain_info)


def format_exception_chain(exc: Exception, max_depth: int = 5) -> list[dict[str, Any]]:
    """Format exception chain

    Args:
        exc: Exception instance
        max_depth: Maximum depth

    Returns:
        List of exception information dictionaries
    """
    chain = []
    current: BaseException | None = exc
    depth = 0

    while current and depth < max_depth:
        chain.append(
            {
                "type": type(current).__name__,
                "message": str(current),
                "module": type(current).__module__,
                "depth": depth,
            }
        )

        current = current.__cause__ or current.__context__
        depth += 1

    return chain
