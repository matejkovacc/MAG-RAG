"""Allowlisted failure metadata without provider messages, payloads or credentials."""

from typing import Literal, Mapping
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from src.knowledge_base.vector_store import RetrievalError

Operation = Literal["embedding", "chat"]
FailureCategory = Literal[
    "authentication",
    "permission",
    "rate_limit",
    "provider_server",
    "provider_request",
    "timeout",
    "connection",
    "invalid_response",
    "local_error",
]


class FailureDiagnostic(BaseModel):
    """Safe metadata retained for the failed question, not an HTTP response dump."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    category: FailureCategory
    operation: Operation | None = None
    http_status: int | None = None
    request_id: str | None = None


class LiveAnswerFailure(RetrievalError):
    """Expose only validated diagnostic fields through the error boundary."""

    def __init__(self, diagnostic: FailureDiagnostic) -> None:
        """Never include a raw exception message in UI text or saved output."""
        self.diagnostic = diagnostic
        super().__init__(
            f"Live answer failed ({diagnostic.category}); inspect saved diagnostics"
        )


def http_failure(
    status: int, headers: Mapping[str, str], operation: Operation
) -> FailureDiagnostic:
    """Retain only status/category and UUID-shaped Azure request identifiers."""
    categories: dict[int, FailureCategory] = {
        401: "authentication",
        403: "permission",
        429: "rate_limit",
    }
    category = categories.get(
        status, "provider_server" if status >= 500 else "provider_request"
    )
    request_id = None
    for name in ("apim-request-id", "x-request-id", "x-ms-request-id"):
        raw = headers.get(name)  # httpx.Headers; never serialize all headers.
        if isinstance(raw, str) and len(raw) == 36:
            try:
                parsed = str(UUID(raw))
                if parsed == raw.lower():
                    request_id = parsed
                    break
            except ValueError:
                pass
    return FailureDiagnostic(
        category=category,
        operation=operation,
        http_status=status,
        request_id=request_id,
    )


def exception_failure(exc: Exception, operation: Operation | None) -> FailureDiagnostic:
    """Map known exception classes to fixed labels; never persist messages/body/code."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    category: FailureCategory
    if names.intersection({"APITimeoutError", "TimeoutException", "TimeoutError"}):
        category = "timeout"
    elif names.intersection({"APIConnectionError", "ConnectError", "ConnectionError"}):
        category = "connection"
    elif isinstance(exc, (ValueError, RetrievalError)):
        category = "invalid_response"
    else:
        category = "local_error"
    return FailureDiagnostic(category=category, operation=operation)
