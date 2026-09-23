from contextvars import ContextVar


request_id_context: ContextVar[str] = ContextVar(
    "request_id",
    default="unknown",
)


def get_request_id() -> str:
    """Return the request ID for the current execution context."""

    return request_id_context.get()