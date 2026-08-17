from fastapi.testclient import TestClient

from main import app


client = TestClient(app)

def test_get_students():

    response = client.get("/students/")

    assert response.status_code == 200

def test_get_student():

    response = client.get("/students/1")

    assert response.status_code == 200

def test_student_not_found():

    response = client.get("/students/99999")

    assert response.status_code == 404

def test_get_student_data():

    response = client.get("/students/1")

    data = response.json()

    assert response.status_code == 200
    assert data["id"] == 1         

def test_create_student():

    student = {
        "name": "Test Student",
        "age": 25,
        "course": "AI Engineering",
        "email": "teststudent@example.com"
    }

    response = client.post(
        "/students/",
        json=student
    )

    print(response.json())

    assert response.status_code == 201

def test_create_student_invalid_data():

    student = {
        "name": "Test Student",
        "age": "wrong",
        "course": "AI Engineering",
        "email": "teststudent@example.com"
    }

    response = client.post(
        "/students/",
        json=student
    )

    assert response.status_code == 422

def test_update_student():

    student = {
        "name": "Updated Student",
        "age": 26,
        "course": "GenAI Engineering",
        "email": "updated@example.com"
    }

    response = client.put(
        "/students/1",
        json=student
    )

    print(response.json())

    assert response.status_code == 200               

def test_delete_student():

    response = client.delete(
        "/students/1"
    )

    assert response.status_code == 200    