import time

from google import genai
from google.genai import types

from app.core.config import settings
from app.llm.base import LLMClient
from app.llm.error_classifier import is_retryable_error
from app.llm.errors import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from app.utils.retry import calculate_backoff
from app.core.request_context import get_request_id
from app.utils.logging import get_logger

logger = get_logger(__name__)


class GeminiLLM(LLMClient):
    """Gemini implementation of the LLM client."""

    def __init__(self) -> None:
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=int(settings.gemini_timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a response from Gemini with retry handling."""

        for attempt in range(settings.gemini_max_retries + 1):
            logger.info(
                "Gemini request attempt=%d request_id=%s",
                attempt + 1,
                get_request_id(),
            )
            try:
                response = self.client.models.generate_content(
                    model=settings.gemini_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.2,
                        response_mime_type="application/json",
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        )    
                    ),
                )

                if not response.text:
                    raise LLMProviderError(
                        "Gemini returned an empty response."
                    )

                return response.text

            except LLMProviderError:
                raise

            except Exception as error:
                logger.error(
                    "Gemini request failed. error_type=%s error=%s request_id=%s",
                    type(error).__name__,
                    str(error),
                    get_request_id(),
                )
                mapped_error = self._map_error(error)

                if not is_retryable_error(mapped_error):
                    raise mapped_error from error

                if attempt >= settings.gemini_max_retries:
                    logger.error(
                        "Gemini request failed after maximum retries. "
                        "attempts=%d request_id=%s",
                        attempt + 1,
                        get_request_id(),
                    )
                    raise mapped_error from error

                delay = calculate_backoff(attempt)

                logger.warning(
                    "Retrying Gemini request after error. "
                    "attempt=%d backoff=%.3f seconds request_id=%s",
                    attempt + 1,
                    delay,
                    get_request_id(),
                )

                time.sleep(delay)

    @staticmethod
    def _map_error(error: Exception) -> Exception:
        """Map provider exceptions to application-level LLM errors."""

        error_text = str(error).lower()

        if (
            "401" in error_text
            or "unauthorized" in error_text
            or "authentication" in error_text
            or "api key" in error_text
        ):
            return LLMAuthenticationError(str(error))

        if "429" in error_text or "rate limit" in error_text:
            return LLMRateLimitError(str(error))

        if (
            "503" in error_text
            or "504" in error_text
            or "service unavailable" in error_text
            or "deadline_exceeded" in error_text
        ):
            return LLMServiceUnavailableError(str(error))

        if "timeout" in error_text:
            return LLMTimeoutError(str(error))

        return LLMProviderError(str(error))