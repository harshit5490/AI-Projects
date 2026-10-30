import numpy as np
import pytest

from app.embeddings.service import EmbeddingService


@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService()


def test_single_embedding(embedding_service):

    embedding = embedding_service.embed_text(
        "Python is used for machine learning."
    )

    assert embedding.shape == (384,)


def test_embedding_is_normalized(embedding_service):

    embedding = embedding_service.embed_text(
        "Python is used for machine learning."
    )

    norm = np.linalg.norm(embedding)

    assert np.isclose(norm, 1.0)


def test_batch_embeddings(embedding_service):

    embeddings = embedding_service.embed_texts(
        [
            "Python is used for machine learning.",
            "FastAPI is used for APIs.",
        ]
    )

    assert embeddings.shape == (2, 384)


def test_empty_text(embedding_service):

    with pytest.raises(ValueError):

        embedding_service.embed_text("")


def test_empty_batch(embedding_service):

    result = embedding_service.embed_texts([])

    assert result == []


def test_batch_with_empty_text(embedding_service):

    with pytest.raises(ValueError):

        embedding_service.embed_texts(
            [
                "Python is useful.",
                "",
            ]
        )