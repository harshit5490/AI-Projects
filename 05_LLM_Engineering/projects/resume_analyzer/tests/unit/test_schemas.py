from pydantic import ValidationError

from app.schemas.resume import ResumeRequest, ResumeAnalysis


def test_valid_resume_request():
    request = ResumeRequest(
        resume_text="A" * 100
    )

    assert request.resume_text == "A" * 100


def test_short_resume_is_rejected():
    try:
        ResumeRequest(resume_text="Too short")
    except ValidationError:
        return

    raise AssertionError("Short resume should have been rejected")


def test_valid_resume_analysis():
    analysis = ResumeAnalysis(
        name="John Doe",
        current_role="Python Developer",
        experience_years=2.5,
        skills=["Python", "FastAPI"],
        education=["B.Tech"],
        strengths=["Strong Python knowledge"],
        weaknesses=["Limited deployment experience"],
        recommendations=["Add measurable project impact"],
    )

    assert analysis.experience_years == 2.5


def test_negative_experience_is_rejected():
    try:
        ResumeAnalysis(
            name="John Doe",
            current_role="Python Developer",
            experience_years=-1,
            skills=[],
            education=[],
            strengths=[],
            weaknesses=[],
            recommendations=[],
        )
    except ValidationError:
        return

    raise AssertionError("Negative experience should have been rejected")