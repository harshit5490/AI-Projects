import json
import time

from pydantic import ValidationError

from app.core.exceptions import (
    InvalidJSONError,
    InvalidResumeError,
    SchemaValidationError,
)
from app.core.request_context import get_request_id
from app.llm.base import LLMClient
from app.prompts.resume_prompts import (
    RESUME_ANALYSIS_SYSTEM_PROMPT,
    build_resume_analysis_prompt,
)
from app.schemas.resume import ResumeAnalysis
from app.utils.logging import get_logger


logger = get_logger(__name__)


class ResumeAnalyzer:
    """Business logic for analyzing resumes using an LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def analyze(self, resume_text: str) -> ResumeAnalysis:
        start_time = time.perf_counter()

        if not resume_text.strip():
            logger.warning(
                "Resume analysis rejected: empty resume text. request_id=%s",
                get_request_id(),
            )

            raise InvalidResumeError(
                "Resume text cannot be empty."
            )

        logger.info(
            "Resume analysis started. request_id=%s",
            get_request_id(),
        )

        user_prompt = build_resume_analysis_prompt(resume_text)

        llm_start = time.perf_counter()

        raw_response = self.llm_client.generate(
            system_prompt=RESUME_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        llm_duration = time.perf_counter() - llm_start

        logger.info(
            "LLM response received in %.3f seconds. request_id=%s",
            llm_duration,
            get_request_id(),
        )

        try:
            response_data = json.loads(raw_response)

        except json.JSONDecodeError as exc:
            logger.error(
                "LLM returned invalid JSON. request_id=%s",
                get_request_id(),
            )

            raise InvalidJSONError(
                "LLM returned invalid JSON."
            ) from exc

        try:
            analysis = ResumeAnalysis.model_validate(
                response_data
            )

        except ValidationError as exc:
            logger.error(
                "LLM response failed schema validation. request_id=%s",
                get_request_id(),
            )

            raise SchemaValidationError(
                "LLM response failed schema validation."
            ) from exc

        total_duration = time.perf_counter() - start_time

        logger.info(
            "Resume analysis completed successfully in %.3f seconds. request_id=%s",
            total_duration,
            get_request_id(),
        )

        return analysis