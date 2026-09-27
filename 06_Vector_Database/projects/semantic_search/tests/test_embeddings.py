from pathlib import Path

from app.embeddings.model import EmbeddingModel
from app.ingestion.loader import load_documents


def get_documents():

    documents_path = (
        Path(__file__).parent.parent
        / "data"
        / "documents"
    )

    return load_documents(
        str(documents_path)
    )


def test_embedding_dimension():

    embedding_model = EmbeddingModel()

    assert embedding_model.dimension == 384


def test_encode_documents():

    documents = get_documents()

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.encode_documents(
        documents
    )

    assert len(embeddings) == len(documents)

    assert len(embeddings[0]) == 384


def test_encode_query():

    embedding_model = EmbeddingModel()

    embedding = embedding_model.encode_query(
        "How can I build a Python API?"
    )

    assert len(embedding) == 384