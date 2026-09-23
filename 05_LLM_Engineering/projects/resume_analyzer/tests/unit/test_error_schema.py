from app.schemas.errors import APIErrorResponse


def test_api_error_response_schema():
    error = APIErrorResponse(
        error="llm_timeout",
        message="The LLM service timed out.",
    )

    assert error.error == "llm_timeout"
    assert error.message == "The LLM service timed out."