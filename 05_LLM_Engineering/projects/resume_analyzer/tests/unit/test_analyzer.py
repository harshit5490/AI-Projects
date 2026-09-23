from unittest.mock import MagicMock
import pytest
from app.llm.mock import MockLLM
from app.services.resume_analyzer import ResumeAnalyzer
from app.core.exceptions import (
    InvalidJSONError,
    SchemaValidationError,
    InvalidResumeError
)
from app.llm.errors import (
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)


def test_resume_analyzer_returns_valid_analysis():
    mock_response = """
    {
        "name": "John Doe",
        "current_role": "Python Developer",
        "experience_years": 2.0,
        "skills": ["Python", "FastAPI", "Docker"],
        "education": ["B.Tech in Computer Science"],
        "strengths": ["Strong backend development"],
        "weaknesses": ["Limited cloud experience"],
        "recommendations": ["Add measurable project outcomes"]
    }
    """

    analyzer = ResumeAnalyzer(
        llm_client=MockLLM(response=mock_response)
    )

    result = analyzer.analyze(
        resume_text="John Doe is a Python Developer with 2 years of experience."
    )

    assert result.name == "John Doe"
    assert result.current_role == "Python Developer"
    assert result.experience_years == 2.0
    assert "Python" in result.skills

def test_resume_analyzer_rejects_invalid_json():
    analyzer = ResumeAnalyzer(
        llm_client=MockLLM(
            response="This is not valid JSON."
        )
    )

    try:
        analyzer.analyze("A valid-looking resume text with enough content.")
    except InvalidJSONError as exc:
        assert str(exc) == "LLM returned invalid JSON."
        return

    raise AssertionError("Invalid JSON should have raised ValueError")


def test_resume_analyzer_rejects_invalid_schema():
    mock_response = """
    {
        "name": "John Doe",
        "current_role": "Python Developer",
        "experience_years": -2,
        "skills": [],
        "education": [],
        "strengths": [],
        "weaknesses": [],
        "recommendations": []
    }
    """

    analyzer = ResumeAnalyzer(
        llm_client=MockLLM(response=mock_response)
    )

    try:
        analyzer.analyze("A valid-looking resume text with enough content.")
    except SchemaValidationError as exc:
        assert str(exc) == "LLM response failed schema validation."
        return

    raise AssertionError("Invalid schema should have raised ValueError")    

def test_whitespace_only_resume_is_rejected():
    llm = MockLLM(response='{}')
    analyzer = ResumeAnalyzer(llm)

    with pytest.raises(InvalidResumeError):
        analyzer.analyze("   ")

def test_invalid_resume_does_not_call_llm():
    llm = MockLLM(response="{}")
    analyzer = ResumeAnalyzer(llm)

    with pytest.raises(InvalidResumeError):
        analyzer.analyze("")

    assert llm.call_count == 0

def test_valid_resume_calls_llm():
    response = """
    {
        "name": "John Doe",
        "current_role": "Python Developer",
        "experience_years": 2,
        "skills": ["Python"],
        "education": ["B.Tech"],
        "strengths": ["Backend development"],
        "weaknesses": ["Limited experience"],
        "recommendations": ["Learn cloud technologies"]
    }
    """

    llm = MockLLM(response=response)
    analyzer = ResumeAnalyzer(llm)

    analyzer.analyze("John Doe is a Python Developer with 2 years of experience.")

    assert llm.call_count == 1   

def test_rate_limit_error_propagates_from_llm():
    llm = MockLLM(response="")
    llm.generate = MagicMock(
        side_effect=LLMRateLimitError("Rate limit exceeded")
    )

    analyzer = ResumeAnalyzer(llm)

    with pytest.raises(LLMRateLimitError):
        analyzer.analyze("John Doe is a Python Developer.")

    assert llm.generate.call_count == 1

def test_timeout_error_propagates_from_llm():
    llm = MockLLM(response="")
    llm.generate = MagicMock(
        side_effect=LLMTimeoutError("Request timed out")
    )

    analyzer = ResumeAnalyzer(llm)

    with pytest.raises(LLMTimeoutError):
        analyzer.analyze("John Doe is a Python Developer.")

    assert llm.generate.call_count == 1

def test_service_unavailable_error_propagates_from_llm():
    llm = MockLLM(response="")
    llm.generate = MagicMock(
        side_effect=LLMServiceUnavailableError(
            "Service temporarily unavailable"
        )
    )

    analyzer = ResumeAnalyzer(llm)

    with pytest.raises(LLMServiceUnavailableError):
        analyzer.analyze("John Doe is a Python Developer.")

    assert llm.generate.call_count == 1                     