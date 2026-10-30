from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    text: str = Field(min_length=1)
    document_id: str
    filename: str
    file_type: str
    chunk_index: int = Field(ge=0)
    page: int | None = Field(default=None, ge=1)