from fastapi import FastAPI,HTTPException,status
from pydantic import BaseModel

app = FastAPI()

class Student(BaseModel):

    id: int

    name: str

    age: int

    course: str

students = [
    {
        "id": 1,
        "name": "Harshit",
        "age": 24,
        "course": "AI Engineering",
    },
    {
        "id": 2,
        "name": "Rahul",
        "age": 23,
        "course": "Python",
    },
]

@app.get(
        "/student",
        tags=["Students"],
        summary="Get all students",
        description="Return all students stored in memory"
        )
def get_student():
    return students

@app.get("/studen/{student_id}")
def get_student(student_id:int):
    for student in students:
        if student["id"] == student_id:
            return student
    # return{
    #     "message":"student not found"
    # }
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="student not found"
    )

@app.post(

    "/students",

    tags=["Students"],

    summary="Create Student",

    description="Create a new student record."

)
def create_student(student:Student):
    students.append(
        student.model_dump()
    )
    return {
        "message": "Student Added Successfully",
        "student": student
    }

@app.put("/student/{student_id}")
def update_student(
    student_id : int,
    updated_student : Student
):
    for index,student in enumerate(students):

        if student["id"] == student_id:
            students[index] = updated_student.model_dump()
            return {
                "message": "Student Updated Successfully"
            }
    # return {
    #     "message": "Student not found"
    # }
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="student not found"
    )

@app.delete("/students/{student_id}")
def delete_student(student_id: int):

    for index, student in enumerate(students):

        if student["id"] == student_id:

            students.pop(index)

            return {
                "message": "Student Deleted Successfully"
            }

    # return {
    #     "message": "Student not found"
    # }
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="student not found"
    )
        

