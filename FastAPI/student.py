from fastapi import APIRouter,status,HTTPException
from pydantic import BaseModel

router = APIRouter()

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

@router.get(
        "/",
        tags=["Students"],
        summary="Get all students",
        description="Return all students stored in memory",
        status_code=status.HTTP_200_OK
        )
def get_student():
    return students

@router.get("/{student_id}",tags=["Students"],summary="Get Student Details",description="Return student details through ID",status_code=status.HTTP_200_OK)
def get_student(student_id:int):
    for student in students:
        if student["id"] == student_id:
            return student
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="student not found"
    )

@router.post(

    "/",

    tags=["Students"],

    summary="Create Student",

    description="Create a new student record.",

    status_code=status.HTTP_201_CREATED

)
def create_student(student:Student):
    students.append(
        student.model_dump()
    )
    return {
        "message": "Student Added Successfully",
        "student": student
    }

@router.put("/{student_id}",
            tags=["Students"],
            summary="Update Student Details",
            description="Update the given student details in memory",
            status_code=status.HTTP_200_OK
)
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
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="student not found"
    )

@router.delete("/{student_id}",
               tags=["Students"],
               summary="Delete student details",
               description="Delete student details from memory",
               status_code=status.HTTP_200_OK
)
def delete_student(student_id: int):

    for index, student in enumerate(students):

        if student["id"] == student_id:

            students.pop(index)

            return {
                "message": "Student Deleted Successfully"
            }
    raise HTTPException(
        status_code=status.HTTP_204_NO_CONTENT,
        detail="student not found"
    )
        

