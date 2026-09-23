from app.llm.errors import (
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)


RETRYABLE_ERRORS = (
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)


def is_retryable_error(error: Exception) -> bool:
    """Return True when an LLM error can be safely retried."""

    return isinstance(error, RETRYABLE_ERRORS)