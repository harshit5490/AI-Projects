from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_answer_generation_service


def test_ask_question_success():
    fake_service = type(
        "FakeService",
        (),
        {
            "answer_question": lambda self, **kwargs: {
                "question": kwargs["question"].strip(),
                "answer": "FastAPI is a Python web framework.",
                "sources": [
                    {
                        "document_id": "doc-1",
                        "filename": "fastapi.txt",
                        "file_type": "txt",
                        "chunk_index": 0,
                        "page": None,
                        "score": 0.85,
                        "text": "FastAPI is a Python framework for building APIs.",
                    }
                ],
            }
        },
    )()

    app.dependency_overrides[
        get_answer_generation_service
    ] = lambda: fake_service

    try:
        client = TestClient(app)
        response = client.post(
            "/questions/ask",
            json={
                "question": "What is FastAPI?",
                "top_k": 3,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == (
            "FastAPI is a Python web framework."
        )
        assert data["sources"][0]["filename"] == "fastapi.txt"

    finally:
        app.dependency_overrides.clear()


def test_ask_question_rejects_invalid_top_k():
    client = TestClient(app)

    response = client.post(
        "/questions/ask",
        json={
            "question": "What is FastAPI?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422