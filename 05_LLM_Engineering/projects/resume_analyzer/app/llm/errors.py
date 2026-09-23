class LLMError(Exception):
    """Base class for LLM-related errors."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider rate-limits the request."""


class LLMServiceUnavailableError(LLMError):
    """Raised when the LLM provider is temporarily unavailable."""


class LLMTimeoutError(LLMError):
    """Raised when the LLM request times out."""


class LLMAuthenticationError(LLMError):
    """Raised when LLM provider authentication fails."""


class LLMProviderError(LLMError):
    """Raised for non-retryable provider errors."""