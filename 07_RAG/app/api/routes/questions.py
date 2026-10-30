from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_answer_generation_service
from app.schemas.question import QuestionRequest, QuestionResponse
from app.services.answer_generation_service import AnswerGenerationService


router = APIRouter(
    prefix="/questions",
    tags=["Questions"],
)


@router.post(
    "/ask",
    response_model=QuestionResponse,
)
def ask_question(
    request: QuestionRequest,
    answer_service: AnswerGenerationService = Depends(
        get_answer_generation_service
    ),
):
    try:
        result = answer_service.answer_question(
            question=request.question,
            top_k=request.top_k,
        )
        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Answer generation failed.",
        ) from exc