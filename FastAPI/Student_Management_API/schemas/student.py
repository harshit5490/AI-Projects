from pydantic import BaseModel,EmailStr


class StudentCreate(BaseModel):

    name: str
    age: int
    course: str
    email:EmailStr | None = None


class StudentUpdate(BaseModel):

    name: str
    age: int
    course: str
    email:EmailStr | None = None


class StudentResponse(BaseModel):

    id: int
    name: str
    age: int
    course: str
    email:EmailStr | None = None

    model_config = {
        "from_attributes": True
    }