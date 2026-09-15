from pydantic import BaseModel, Field


class AddNumbersOutput(BaseModel):
    result: int


class MultiplyNumbersOutput(BaseModel):
    result: int


class UserInfoOutput(BaseModel):
    name: str
    role: str
    experience: str