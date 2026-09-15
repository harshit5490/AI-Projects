from pydantic import BaseModel, Field


class AddNumbersArgs(BaseModel):
    a: int
    b: int


class MultiplyNumbersArgs(BaseModel):
    a: int
    b: int


class GetUserInfoArgs(BaseModel):
    name: str = Field(min_length=1)