from fastapi import FastAPI

from app.api.exception_handlers import (
    invalid_json_handler,
    invalid_resume_handler,
    llm_authentication_handler,
    llm_provider_handler,
    llm_rate_limit_handler,
    llm_service_unavailable_handler,
    llm_timeout_handler,
    schema_validation_handler,
)
from app.api.routes.resume import router as resume_router
from app.api.routes.health import router as health_router
from app.core.exceptions import (
    InvalidJSONError,
    InvalidResumeError,
    SchemaValidationError,
)
from app.llm.errors import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from app.api.middleware import request_id_middleware

app = FastAPI(
    title="Resume Analyzer API",
    description="AI-powered resume analysis API.",
    version="1.0.0",
)

app.middleware("http")(request_id_middleware)
app.add_exception_handler(
    InvalidResumeError,
    invalid_resume_handler,
)

app.add_exception_handler(
    LLMRateLimitError,
    llm_rate_limit_handler,
)

app.add_exception_handler(
    LLMTimeoutError,
    llm_timeout_handler,
)

app.add_exception_handler(
    LLMServiceUnavailableError,
    llm_service_unavailable_handler,
)

app.add_exception_handler(
    LLMAuthenticationError,
    llm_authentication_handler,
)

app.add_exception_handler(
    LLMProviderError,
    llm_provider_handler,
)

app.add_exception_handler(
    InvalidJSONError,
    invalid_json_handler,
)

app.add_exception_handler(
    SchemaValidationError,
    schema_validation_handler,
)
app.include_router(health_router)

app.include_router(resume_router)

