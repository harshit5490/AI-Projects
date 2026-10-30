
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question to answer using uploaded documents.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of document chunks to retrieve.",
    )


class SourceReference(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunk_index: int = Field(ge=0)
    page: int | None = Field(default=None, ge=1)
    score: float
    text: str = Field(
        min_length=1,
        description="Text of the retrieved chunk used as supporting context.",
    )


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceReference]
