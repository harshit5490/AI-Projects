from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    category: str = Field(min_length=1)