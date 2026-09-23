from pydantic import BaseModel


class APIErrorResponse(BaseModel):
    """Standard API error response."""

    error: str
    message: str