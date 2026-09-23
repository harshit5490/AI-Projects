from app.llm.error_classifier import is_retryable_error
from app.llm.errors import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
    LLMProviderError
)


def test_rate_limit_error_is_retryable():
    assert is_retryable_error(LLMRateLimitError()) is True


def test_service_unavailable_error_is_retryable():
    assert is_retryable_error(LLMServiceUnavailableError()) is True


def test_timeout_error_is_retryable():
    assert is_retryable_error(LLMTimeoutError()) is True


def test_provider_error_is_not_retryable():
    assert is_retryable_error(LLMProviderError()) is False

def test_authentication_error_is_not_retryable():
    assert is_retryable_error(LLMAuthenticationError()) is False    