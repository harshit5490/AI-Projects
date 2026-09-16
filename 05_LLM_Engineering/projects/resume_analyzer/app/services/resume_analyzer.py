import json

from pydantic import ValidationError

from app.core.exceptions import (
    InvalidJSONError,
    SchemaValidationError,
)
from app.llm.base import LLMClient
from app.prompts.resume_prompts import (
    RESUME_ANALYSIS_SYSTEM_PROMPT,
    build_resume_analysis_prompt,
)
from app.schemas.resume import ResumeAnalysis


class ResumeAnalyzer:
    """Business logic for analyzing resumes using an LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def analyze(self, resume_text: str) -> ResumeAnalysis:
        user_prompt = build_resume_analysis_prompt(resume_text)

        raw_response = self.llm_client.generate(
            system_prompt=RESUME_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        try:
            response_data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise InvalidJSONError(
                "LLM returned invalid JSON."
            ) from exc

        try:
            return ResumeAnalysis.model_validate(response_data)
        except ValidationError as exc:
            raise SchemaValidationError(
                "LLM response failed schema validation."
            ) from exc