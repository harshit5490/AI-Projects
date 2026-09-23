from app.llm.gemini import GeminiLLM
from app.services.resume_analyzer import ResumeAnalyzer


def get_resume_analyzer() -> ResumeAnalyzer:
    """Create and return the resume analyzer service."""

    llm_client = GeminiLLM()

    return ResumeAnalyzer(llm_client)