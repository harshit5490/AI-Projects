from fastapi import APIRouter, Depends

from app.api.dependencies import get_resume_analyzer
from app.schemas.resume import ResumeAnalysis, ResumeRequest
from app.services.resume_analyzer import ResumeAnalyzer


router = APIRouter(
    prefix="/api/v1/resume",
    tags=["Resume"],
)


@router.post(
    "/analyze",
    response_model=ResumeAnalysis,
)
def analyze_resume(
    request: ResumeRequest,
    analyzer: ResumeAnalyzer = Depends(get_resume_analyzer),
) -> ResumeAnalysis:
    return analyzer.analyze(request.resume_text)