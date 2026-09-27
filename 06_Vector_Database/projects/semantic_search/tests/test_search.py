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


def setup_store(tmp_path):
    documents = get_documents()

    embedding_model = EmbeddingModel()
    embeddings = embedding_model.encode_documents(documents)

    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_data")
    )

    store.add_documents(documents, embeddings)

    return store, embedding_model


def test_top_k_search(tmp_path):
    store, embedding_model = setup_store(tmp_path)

    query_embedding = embedding_model.encode_query(
        "How can I build a Python API?"
    )

    results = store.search(
        query_embedding=query_embedding,
        top_k=3,
    )

    assert len(results["ids"][0]) == 3
    assert len(results["documents"][0]) == 3
    assert len(results["distances"][0]) == 3


def test_search_returns_python_document(tmp_path):
    store, embedding_model = setup_store(tmp_path)

    query_embedding = embedding_model.encode_query(
        "How can I build a Python API?"
    )

    results = store.search(
        query_embedding=query_embedding,
        top_k=3,
    )

    assert "python" in results["ids"][0]


def test_invalid_top_k(tmp_path):
    store, embedding_model = setup_store(tmp_path)

    query_embedding = embedding_model.encode_query(
        "Python programming"
    )

    try:
        store.search(
            query_embedding=query_embedding,
            top_k=0,
        )
        assert False
    except ValueError:
        assert True

def test_metadata_filter_by_category(tmp_path):
    store, embedding_model = setup_store(tmp_path)

    query_embedding = embedding_model.encode_query(
        "What technologies are used in AI?"
    )

    results = store.search(
        query_embedding=query_embedding,
        top_k=3,
        where={"category": "AI"},
    )

    assert len(results["ids"][0]) > 0

    for metadata in results["metadatas"][0]:
        assert metadata["category"] == "AI"

def test_metadata_filter_by_source(tmp_path):
    store, embedding_model = setup_store(tmp_path)

    query_embedding = embedding_model.encode_query(
        "Python programming"
    )

    results = store.search(
        query_embedding=query_embedding,
        top_k=3,
        where={"source": "python.txt"},
    )

    assert results["ids"][0] == ["python"]

    assert results["metadatas"][0][0]["source"] == "python.txt"

def test_metadata_filter_with_top_k(tmp_path):
    store, embedding_model = setup_store(tmp_path)

    query_embedding = embedding_model.encode_query(
        "Artificial intelligence"
    )

    results = store.search(
        query_embedding=query_embedding,
        top_k=2,
        where={"category": "AI"},
    )

    assert len(results["ids"][0]) <= 2

    for metadata in results["metadatas"][0]:
        assert metadata["category"] == "AI"                    

