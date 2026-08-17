from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database.session import get_db
from models.student import Student
from schemas.student import StudentCreate,StudentResponse,StudentUpdate
from services.student_service import get_all_students,create_student_record,get_student_by_id,update_student_record,delete_student_record


router = APIRouter()

@router.get(
    "/",
    response_model=list[StudentResponse],
    status_code=status.HTTP_200_OK,
)
def get_students(
    db: Session = Depends(get_db)
):

    # students = db.query(Student).all()

    return get_all_students(db)

@router.get(
    "/{student_id}",
    response_model=StudentResponse,
    status_code=status.HTTP_200_OK,
)
def get_student(
    student_id: int,
    db: Session = Depends(get_db)
):

    # student = db.get(
    #     Student,
    #     student_id
    # )
    student = get_student_by_id(db,student_id)

    if student is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found"
        )

    return student

@router.post(
    "/",
    response_model=StudentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_student(
    student: StudentCreate,
    db: Session = Depends(get_db)
):

    new_student = Student(
        name=student.name,
        age=student.age,
        course=student.course,
        email=student.email
    )

    # db.add(new_student)

    # db.commit()

    # db.refresh(new_student)

    # return new_student
    return create_student_record(db,new_student)

@router.delete(
    "/{student_id}",
    status_code=status.HTTP_200_OK,
)
def delete_student(
    student_id: int,
    db: Session = Depends(get_db)
):

    # student = db.get(
    #     Student,
    #     student_id
    # )
    student = get_student_by_id(db,student_id)

    if student is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found"
        )

    # db.delete(student)

    # db.commit()
    delete_student_record(db,student)

    return {
        "message": "Student deleted successfully"
    }

@router.put(
    "/{student_id}",
    response_model=StudentResponse,
    status_code=status.HTTP_200_OK,
)
def update_student(
    student_id: int,
    updated_student: StudentUpdate,
    db: Session = Depends(get_db)
):

    # student = db.get(
    #     Student,
    #     student_id
    # )
    student = get_student_by_id(db,student_id)

    if student is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found"
        )

    # student.name = updated_student.name
    # student.age = updated_student.age
    # student.course = updated_student.course

    # db.commit()

    # db.refresh(student)

    return update_student_record(
        db=db,
        student=student,
        name=updated_student.name,
        age=updated_student.age,
        course=updated_student.course,
        email=updated_student.email,
    )