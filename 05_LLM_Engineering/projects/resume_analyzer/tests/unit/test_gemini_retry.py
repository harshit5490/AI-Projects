from unittest.mock import MagicMock, patch
from app.llm.errors import LLMAuthenticationError,LLMRateLimitError
import pytest

from app.core.config import settings
from app.llm.errors import (
    LLMProviderError,
    LLMServiceUnavailableError,
)
from app.llm.gemini import GeminiLLM


def create_success_response(text: str = '{"status": "success"}'):
    response = MagicMock()
    response.text = text
    return response


def create_gemini_mock():
    client = MagicMock()
    client.models.generate_content = MagicMock()
    return client


def test_gemini_success_on_first_attempt():
    client = create_gemini_mock()

    client.models.generate_content.return_value = create_success_response()

    with patch(
        "app.llm.gemini.genai.Client",
        return_value=client,
    ):
        llm = GeminiLLM()

        response = llm.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Return a success response.",
        )

    assert response == '{"status": "success"}'
    assert client.models.generate_content.call_count == 1


def test_gemini_retries_on_service_unavailable():
    client = create_gemini_mock()

    client.models.generate_content.side_effect = [
        Exception("503 Service Unavailable"),
        create_success_response(),
    ]

    with patch(
        "app.llm.gemini.genai.Client",
        return_value=client,
    ):
        llm = GeminiLLM()

        with patch("app.llm.gemini.time.sleep"):
            response = llm.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt="Return a success response.",
            )

    assert response == '{"status": "success"}'
    assert client.models.generate_content.call_count == 2


def test_gemini_raises_after_retries_are_exhausted():
    client = create_gemini_mock()

    client.models.generate_content.side_effect = Exception(
        "503 Service Unavailable"
    )

    with patch(
        "app.llm.gemini.genai.Client",
        return_value=client,
    ):
        llm = GeminiLLM()

        with patch("app.llm.gemini.time.sleep"):
            with pytest.raises(LLMServiceUnavailableError):
                llm.generate(
                    system_prompt="You are a helpful assistant.",
                    user_prompt="Return a success response.",
                )

    expected_attempts = settings.gemini_max_retries + 1

    assert client.models.generate_content.call_count == expected_attempts


def test_gemini_does_not_retry_non_retryable_error():
    client = create_gemini_mock()

    client.models.generate_content.side_effect = Exception(
        "400 Bad Request"
    )

    with patch(
        "app.llm.gemini.genai.Client",
        return_value=client,
    ):
        llm = GeminiLLM()

        with pytest.raises(LLMProviderError):
            llm.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt="Return a success response.",
            )

    assert client.models.generate_content.call_count == 1

def test_authentication_error_is_mapped_correctly():
    error = Exception("401 Unauthorized: invalid API key")

    mapped_error = GeminiLLM._map_error(error)

    assert isinstance(mapped_error, LLMAuthenticationError)

def test_rate_limit_error_is_mapped_correctly():
    error = Exception("429 Too Many Requests")

    mapped_error = GeminiLLM._map_error(error)

    assert isinstance(mapped_error, LLMRateLimitError)        

def test_deadline_exceeded_error_is_mapped_correctly():
    error = Exception(
        "504 DEADLINE_EXCEEDED: Deadline expired before operation could complete."
    )

    mapped_error = GeminiLLM._map_error(error)

    assert isinstance(mapped_error, LLMServiceUnavailableError)

def test_gemini_retries_on_deadline_exceeded():
    client = create_gemini_mock()

    client.models.generate_content.side_effect = [
        Exception("504 DEADLINE_EXCEEDED"),
        create_success_response(),
    ]

    with patch(
        "app.llm.gemini.genai.Client",
        return_value=client,
    ):
        llm = GeminiLLM()

        with patch("app.llm.gemini.time.sleep"):
            response = llm.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt="Return a success response.",
            )

    assert response == '{"status": "success"}'
    assert client.models.generate_content.call_count == 2        

def test_gemini_disables_sdk_retries():
    with patch("app.llm.gemini.genai.Client") as mock_client:
        GeminiLLM()

    _, kwargs = mock_client.call_args

    http_options = kwargs["http_options"]

    assert http_options.retry_options.attempts == 1    