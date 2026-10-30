from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    id: str
    score: float
    text: str
    document_id: str
    filename: str
    file_type: str
    chunk_index: int = Field(ge=0)
    page: int | None = Field(default=None, ge=1)