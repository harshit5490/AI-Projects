import time
import uuid

from app.embeddings.service import EmbeddingService
from app.vectorstore.store import PineconeVectorStore


def test_pinecone_real_upsert_and_query():

    embedding_service = EmbeddingService()

    store = PineconeVectorStore(
        index_name="rag-document-index",
        namespace="integration-tests",
    )

    document_id = f"test-doc-{uuid.uuid4()}"

    text = (
        "Python is a high-level programming language "
        "commonly used for machine learning."
    )

    embedding = embedding_service.embed_text(text)

    vector_id = f"{document_id}_chunk_000"

    store.upsert(
        [
            {
                "id": vector_id,
                "values": embedding.tolist(),
                "metadata": {
                    "document_id": document_id,
                    "filename": "test.txt",
                    "file_type": "txt",
                    "chunk_index": 0,
                    "text": text,
                },
            }
        ]
    )

    query = "What is Python commonly used for?"

    query_embedding = embedding_service.embed_text(query)

    results = []

    for _ in range(10):
        results = store.query_chunks(
            query_vector=query_embedding,
            top_k=1,
            metadata_filter={
                "document_id": {
                    "$eq": document_id
                }
            },
        )

        if results:
            break

        time.sleep(1)

    try:
        assert len(results) >= 1

        result = results[0]

        assert result["document_id"] == document_id
        assert result["filename"] == "test.txt"
        assert "machine learning" in result["text"]

    finally:
        store.delete_document(document_id)