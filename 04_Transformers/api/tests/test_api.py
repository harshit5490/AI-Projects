from fastapi.testclient import TestClient

from app.main import app
from app.qa_model import MODEL_NAME


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == MODEL_NAME


def test_predict_answer():
    response = client.post(
        "/predict",
        json={
            "question": "Who created Python?",
            "context": (
                "Python was created by Guido van Rossum "
                "and first released in 1991."
            ),
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "Guido van Rossum"
    assert data["has_answer"] is True


def test_predict_no_answer():
    response = client.post(
        "/predict",
        json={
            "question": "Who invented Java?",
            "context": (
                "Python was created by Guido van Rossum "
                "and first released in 1991."
            ),
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == ""
    assert data["has_answer"] is False