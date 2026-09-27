from pathlib import Path

from app.embeddings.model import EmbeddingModel
from app.ingestion.loader import load_documents
from app.vector_store.chroma import ChromaVectorStore


def get_documents():
    documents_path = (
        Path(__file__).parent.parent
        / "data"
        / "documents"
    )
    return load_documents(str(documents_path))


def test_create_vector_store(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_data")
    )

    assert store.collection is not None


def test_add_documents(tmp_path):
    documents = get_documents()

    embedding_model = EmbeddingModel()
    embeddings = embedding_model.encode_documents(documents)

    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_data")
    )

    store.add_documents(documents, embeddings)

    assert store.count() == 5


def test_document_data_is_stored(tmp_path):
    documents = get_documents()

    embedding_model = EmbeddingModel()
    embeddings = embedding_model.encode_documents(documents)

    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_data")
    )

    store.add_documents(documents, embeddings)

    result = store.collection.get(
        ids=["python"]
    )

    python_document = next(
        document
        for document in documents
        if document.id == "python"
    )

    assert result["ids"] == ["python"]
    assert result["documents"][0] == python_document.text
    assert result["metadatas"][0]["source"] == python_document.source
    assert result["metadatas"][0]["category"] == python_document.category