from app.llm.mock import MockLLM
from app.services.resume_analyzer import ResumeAnalyzer
from app.core.exceptions import (
    InvalidJSONError,
    SchemaValidationError,
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