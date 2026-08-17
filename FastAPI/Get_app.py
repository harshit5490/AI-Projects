from fastapi import FastAPI
from typing import Optional

app = FastAPI()

@app.get("/")
def home():
    return {
        "message" : "Welcome to FastAPI"
    }

@app.get("/about")
def about():
    return {
        "course": "AI Engineer Bootcamp",
        "module": "FastAPI",
        "week": 1
    }

@app.get("/contact")
def contact():
    return {
        "email": "support@example.com"
    }
# path parameter
@app.get("/student/{student_id}")
def get_student(student_id : int):
    return{
        "student_id":student_id
    }

@app.get("/employee/{employee_id}")
def get_employee(employee_id: int):
    return {
        "employee_id": employee_id,
        "name": "Harshit",
        "department": "AI",
        "salary": 85000
    }

@app.get("/hello/{name}")
def greet(name: str):
    return {
        "message": f"Hello {name}"
    }

@app.get("/student/{student_id}/course/{course_name}")
def student_course(student_id:int,course_name:str):
    return{
        "student_id":student_id,
        "course":course_name
    }
# Query parameter
@app.get("/student")
def get_student(student_id: int):
    return {
        "student_id": student_id
    }

@app.get("/employee")
def get_employee(
    name:str,
    salary:int
):
    return{
        "name" : name,
        "salary":salary
    }

@app.get("/course")
def get_course(
    course_name:str,
    duration:Optional[int] = None
):
    return{
        "course_name":course_name,
        "duration":duration
    }

@app.get("/product")
def get_product(
    product_name: str,
    quantity: int = 1
):
    return {
        "product_name": product_name,
        "quantity": quantity
    }

# Path and Query Parameter
@app.get("/users/{user_id}")
def get_user(
    user_id: int,
    active: bool = True
):
    return {
        "user_id": user_id,
        "active": active
    }