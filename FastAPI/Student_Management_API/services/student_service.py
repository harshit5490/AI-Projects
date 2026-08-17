from sqlalchemy.orm import Session

from models.student import Student

from repository.student_repository import (
    get_all_students as repo_get_all_students,
    get_student_by_id as repo_get_student_by_id,
    create_student_record as repo_create_student,
    update_student_record as repo_update_student,
    delete_student_record as repo_delete_student,
)

def get_all_students(
    db: Session
):
    return repo_get_all_students(db)


def get_student_by_id(
    db: Session,
    student_id: int
):
    repo_get_student_by_id(db,student_id)

def create_student_record(
    db: Session,
    student: Student
):
    return repo_create_student(db,student)


def update_student_record(
    db: Session,
    student: Student,
    name: str,
    age: int,
    course: str,
    email: str | None
):
    return repo_update_student(
        db=db,
        student=student,
        name=name,
        age=age,
        course=course,
        email=email
    )


def delete_student_record(
    db: Session,
    student: Student
):
    return repo_delete_student(db,student)