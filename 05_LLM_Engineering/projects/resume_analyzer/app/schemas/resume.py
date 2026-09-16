from pydantic import BaseModel, Field


class ResumeRequest(BaseModel):
    resume_text: str = Field(
        min_length=50,
        description="The candidate's resume text."
    )


class ResumeAnalysis(BaseModel):
    name: str
    current_role: str
    experience_years: float = Field(ge=0)
    skills: list[str]
    education: list[str]
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]