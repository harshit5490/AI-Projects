from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Semantic Search API is running."
    )


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["documents"] >= 0


def test_search():
    response = client.post(
        "/api/v1/search",
        json={
            "query": "How can I build a Python API?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "How can I build a Python API?"
    assert len(data["results"]) == 3

    for result in data["results"]:
        assert "id" in result
        assert "text" in result
        assert "source" in result
        assert "category" in result
        assert "distance" in result


def test_search_with_category_filter():
    response = client.post(
        "/api/v1/search",
        json={
            "query": "What technologies are used in AI?",
            "top_k": 3,
            "category": "AI",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["results"]) > 0

    for result in data["results"]:
        assert result["category"] == "AI"


def test_invalid_top_k():
    response = client.post(
        "/api/v1/search",
        json={
            "query": "Python programming",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_empty_query():
    response = client.post(
        "/api/v1/search",
        json={
            "query": "",
            "top_k": 3,
        },
    )

    assert response.status_code == 422