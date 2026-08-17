from fastapi import FastAPI,status
from pydantic import BaseModel
from typing import Optional
app = FastAPI()

class StudentRequest(BaseModel):

    name: str

    age: int

    course: str = "AI Engineering" #Default value

    city:Optional[str] = None  #optional value

class StudentResponse(BaseModel):
    name : str
    course:str

class EmployeeRequest(BaseModel):

    name: str

    salary: float

    department: str


class EmployeeResponse(BaseModel):

    name: str

    department: str    


# @app.post("/student")
# def create_student(student:dict):
#     return{
#        "student":student 
#     } not good practice
def create_student(student:StudentRequest):
    return student

# @app.post("/student",response_model=StudentResponse)
def create_student(student:StudentRequest):
    return student

# Returning status
@app.post("/student",response_model=StudentResponse,status_code=status.HTTP_201_CREATED)
def create_student(student:StudentRequest):
    return student

@app.post("/employee",response_model=EmployeeResponse,status_code=status.HTTP_201_CREATED)
def create_employee(employee:EmployeeRequest):
    return employee