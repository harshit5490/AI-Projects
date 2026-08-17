from sqlalchemy.orm import Session

from models.student import Student


def get_all_students(db: Session):
    return db.query(Student).all()


def get_student_by_id(
    db: Session,
    student_id: int
):
    return db.get(Student, student_id)


def create_student_record(
    db: Session,
    student: Student
):
    try:
        db.add(student)
        db.commit()
        db.refresh(student)

        return student

    except Exception:
        db.rollback()
        raise


def update_student_record(
    db: Session,
    student: Student,
    name: str,
    age: int,
    course: str,
    email: str | None
):
    try:
        student.name = name
        student.age = age
        student.course = course
        student.email = email

        db.commit()
        db.refresh(student)

        return student

    except Exception:
        db.rollback()
        raise


def delete_student_record(
    db: Session,
    student: Student
):
    try:
        db.delete(student)
        db.commit()

    except Exception:
        db.rollback()
        raise