import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_resume_analyzer
from app.llm.mock import MockLLM
from app.main import app
from app.services.resume_analyzer import ResumeAnalyzer

from app.core.exceptions import InvalidJSONError, InvalidResumeError
from app.llm.errors import LLMRateLimitError, LLMTimeoutError


MOCK_RESPONSE = """
{
    "name": "John Doe",
    "current_role": "Python Developer",
    "experience_years": 2.0,
    "skills": [
        "Python",
        "FastAPI",
        "Docker"
    ],
    "education": [
        "B.Tech in Computer Science"
    ],
    "strengths": [
        "Backend development experience"
    ],
    "weaknesses": [
        "Limited experience mentioned"
    ],
    "recommendations": [
        "Add measurable project achievements"
    ]
}
"""


def get_mock_resume_analyzer() -> ResumeAnalyzer:
    mock_llm = MockLLM(MOCK_RESPONSE)
    return ResumeAnalyzer(mock_llm)


app.dependency_overrides[get_resume_analyzer] = get_mock_resume_analyzer

client = TestClient(app)


def test_analyze_resume_api():
    response = client.post(
        "/api/v1/resume/analyze",
        json={
            "resume_text": """
            John Doe

            Python Developer

            Python developer with 2 years of experience
            building backend applications and REST APIs.

            Skills:
            Python, FastAPI, Docker

            Education:
            B.Tech in Computer Science
            """
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "John Doe"
    assert data["current_role"] == "Python Developer"
    assert data["experience_years"] == 2.0
    assert "Python" in data["skills"]

def test_invalid_resume_returns_422():
    def failing_dependency() -> ResumeAnalyzer:
        mock_llm = MockLLM(MOCK_RESPONSE)
        analyzer = ResumeAnalyzer(mock_llm)

        def analyze(_: str):
            raise InvalidResumeError("Resume text cannot be empty.")

        analyzer.analyze = analyze
        return analyzer

    app.dependency_overrides[get_resume_analyzer] = failing_dependency

    response = client.post(
        "/api/v1/resume/analyze",
        json={
            "resume_text": "A valid-looking resume with enough text for testing."
        },
    )

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_resume"

def test_llm_rate_limit_returns_429():
    def failing_dependency() -> ResumeAnalyzer:
        mock_llm = MockLLM(MOCK_RESPONSE)

        class RateLimitLLM(MockLLM):
            def generate(
                self,
                system_prompt: str,
                user_prompt: str,
            ) -> str:
                raise LLMRateLimitError(
                    "Rate limit exceeded."
                )

        return ResumeAnalyzer(RateLimitLLM(MOCK_RESPONSE))

    app.dependency_overrides[get_resume_analyzer] = failing_dependency

    response = client.post(
        "/api/v1/resume/analyze",
        json={
            "resume_text": """
            John Doe is a Python developer with two years
            of experience building backend applications.
            """
        },
    )

    assert response.status_code == 429
    assert response.json()["error"] == "llm_rate_limit"

def test_llm_timeout_returns_503():
    class TimeoutLLM(MockLLM):
        def generate(
            self,
            system_prompt: str,
            user_prompt: str,
        ) -> str:
            raise LLMTimeoutError(
                "LLM request timed out."
            )

    def failing_dependency() -> ResumeAnalyzer:
        return ResumeAnalyzer(
            TimeoutLLM(MOCK_RESPONSE)
        )

    app.dependency_overrides[get_resume_analyzer] = failing_dependency

    response = client.post(
        "/api/v1/resume/analyze",
        json={
            "resume_text": """
            John Doe is a Python developer with two years
            of experience building backend applications.
            """
        },
    )

    assert response.status_code == 503
    assert response.json()["error"] == "llm_timeout"

def test_invalid_llm_json_returns_502():
    invalid_json = """
    This is not valid JSON.
    """

    def invalid_json_dependency() -> ResumeAnalyzer:
        return ResumeAnalyzer(
            MockLLM(invalid_json)
        )

    app.dependency_overrides[get_resume_analyzer] = (
        invalid_json_dependency
    )

    response = client.post(
        "/api/v1/resume/analyze",
        json={
            "resume_text": """
            John Doe is a Python developer with two years
            of experience building backend applications.
            """
        },
    )

    assert response.status_code == 502
    assert response.json()["error"] == "invalid_llm_response" 

def test_invalid_llm_json_returns_standard_error_schema():
    invalid_json = "This is not valid JSON."

    def invalid_json_dependency() -> ResumeAnalyzer:
        return ResumeAnalyzer(
            MockLLM(invalid_json)
        )

    app.dependency_overrides[get_resume_analyzer] = (
        invalid_json_dependency
    )

    response = client.post(
        "/api/v1/resume/analyze",
        json={
            "resume_text": """
            John Doe is a Python developer with two years
            of experience building backend applications.
            """
        },
    )

    assert response.status_code == 502

    data = response.json()

    assert set(data.keys()) == {"error", "message"}
    assert isinstance(data["error"], str)
    assert isinstance(data["message"], str)      

def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
    }


def test_readiness_endpoint():
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
    }       

def test_request_id_header():
    response = client.get("/health")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"]              