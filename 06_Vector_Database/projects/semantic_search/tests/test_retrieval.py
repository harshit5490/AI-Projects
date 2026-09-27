from pathlib import Path

from app.embeddings.model import EmbeddingModel
from app.ingestion.loader import load_documents
from app.retrieval.search import RetrievalService
from app.vector_store.chroma import ChromaVectorStore


def setup_retrieval_service(tmp_path):
    documents_path = (
        Path(__file__).parent.parent
        / "data"
        / "documents"
    )

    documents = load_documents(str(documents_path))

    embedding_model = EmbeddingModel()
    embeddings = embedding_model.encode_documents(documents)

    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_data")
    )

    vector_store.add_documents(
        documents,
        embeddings,
    )

    retrieval_service = RetrievalService(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    return retrieval_service


def test_retrieval_returns_search_results(tmp_path):
    service = setup_retrieval_service(tmp_path)

    results = service.search(
        query="How can I build a Python API?",
        top_k=3,
    )

    assert len(results) == 3

    for result in results:
        assert result.id
        assert result.text
        assert result.source
        assert result.category
        assert isinstance(result.distance, float)


def test_retrieval_returns_python_document(tmp_path):
    service = setup_retrieval_service(tmp_path)

    results = service.search(
        query="How can I build a Python API?",
        top_k=3,
    )

    result_ids = [result.id for result in results]

    assert "python" in result_ids


def test_retrieval_metadata_filter(tmp_path):
    service = setup_retrieval_service(tmp_path)

    results = service.search(
        query="What technologies are used in AI?",
        top_k=3,
        where={"category": "AI"},
    )

    assert len(results) > 0

    for result in results:
        assert result.category == "AI"


def test_empty_query_is_rejected(tmp_path):
    service = setup_retrieval_service(tmp_path)

    try:
        service.search(
            query="   ",
            top_k=3,
        )
        assert False
    except ValueError:
        assert True


def test_invalid_top_k_is_rejected(tmp_path):
    service = setup_retrieval_service(tmp_path)

    try:
        service.search(
            query="Python programming",
            top_k=0,
        )
        assert False
    except ValueError:
        assert True