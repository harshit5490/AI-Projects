from app.llm.gemini import GeminiLLM
from app.schemas.resume import ResumeRequest
from app.services.resume_analyzer import ResumeAnalyzer


def main() -> None:
    """Run a real end-to-end resume analysis using Gemini."""

    resume_text = """
    John Doe

    Python Developer

    Professional Summary:
    Python developer with 2 years of experience building backend
    applications and REST APIs.

    Skills:
    Python, FastAPI, Docker, PostgreSQL, Git

    Experience:
    Python Developer at ABC Technologies
    2024 - Present

    Education:
    B.Tech in Computer Science
    """

    request = ResumeRequest(
        resume_text=resume_text
    )

    llm = GeminiLLM()

    analyzer = ResumeAnalyzer(
        llm_client=llm
    )

    analysis = analyzer.analyze(
        resume_text=request.resume_text
    )

    print("\nRESUME ANALYSIS:")
    print(
        analysis.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()