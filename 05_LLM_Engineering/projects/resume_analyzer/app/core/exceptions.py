class ResumeAnalyzerError(Exception):
    """Base exception for resume analyzer errors."""


class LLMResponseError(ResumeAnalyzerError):
    """Raised when the LLM response cannot be processed."""


class InvalidJSONError(LLMResponseError):
    """Raised when the LLM returns invalid JSON."""


class SchemaValidationError(LLMResponseError):
    """Raised when the LLM response fails schema validation."""