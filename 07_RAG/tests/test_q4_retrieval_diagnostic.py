
from app.generation.context_builder import ContextBuilder


def test_semantic_search_context_contains_expected_evidence():
    chunks = [
        {
            "id": "eval-knowledge-v1_chunk_0001",
            "document_id": "eval-knowledge-v1",
            "filename": "evaluation_documents.txt",
            "file_type": "txt",
            "chunk_index": 1,
            "page": None,
            "score": 0.4133,
            "text": (
                "meaning rather than only exact keyword matches. "
                "Retrieval-augmented generation (RAG) retrieves "
                "relevant information from an external knowledge "
                "source and provides it to a language model as context."
            ),
        }
    ]

    context = ContextBuilder().build(chunks)

    assert "meaning" in context.lower()
    assert "keyword" in context.lower()
    assert "exact keyword matches" in context.lower()
