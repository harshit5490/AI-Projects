from fastapi import Request
from fastapi.responses import JSONResponse

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
from app.schemas.errors import APIErrorResponse


def invalid_resume_handler(
    request: Request,
    exc: InvalidResumeError,
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=APIErrorResponse(
            error="invalid_resume",
            message=str(exc),
        ).model_dump(),
    )


def llm_rate_limit_handler(
    request: Request,
    exc: LLMRateLimitError,
) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content=APIErrorResponse(
            error="llm_rate_limit",
            message="The LLM provider rate limit was reached.",
        ).model_dump(),
    )


def llm_timeout_handler(
    request: Request,
    exc: LLMTimeoutError,
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content=APIErrorResponse(
            error="llm_timeout",
            message="The LLM service timed out. Please try again later.",
        ).model_dump(),
    )


def llm_service_unavailable_handler(
    request: Request,
    exc: LLMServiceUnavailableError,
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content=APIErrorResponse(
            error="llm_service_unavailable",
            message="The LLM service is temporarily unavailable.",
        ).model_dump(),
    )


def llm_authentication_handler(
    request: Request,
    exc: LLMAuthenticationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=APIErrorResponse(
            error="llm_configuration_error",
            message="The LLM service could not be authenticated.",
        ).model_dump(),
    )


def llm_provider_handler(
    request: Request,
    exc: LLMProviderError,
) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content=APIErrorResponse(
            error="llm_provider_error",
            message="The LLM provider returned an error.",
        ).model_dump(),
    )


def invalid_json_handler(
    request: Request,
    exc: InvalidJSONError,
) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content=APIErrorResponse(
            error="invalid_llm_response",
            message="The LLM returned an invalid response.",
        ).model_dump(),
    )


def schema_validation_handler(
    request: Request,
    exc: SchemaValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content=APIErrorResponse(
            error="invalid_llm_response",
            message="The LLM response did not match the expected schema.",
        ).model_dump(),
    )