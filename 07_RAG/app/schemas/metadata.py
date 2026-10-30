from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunk_index: int = Field(ge=0)
    page: int | None = Field(default=None, ge=1)