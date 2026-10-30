from unittest.mock import Mock

from app.schemas.chunk import DocumentChunk
from app.services.ingestion_service import IngestionService


def test_ingest_chunks():

    embedding_service = Mock()
    vector_store = Mock()

    embedding_service.embed_texts.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    service = IngestionService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    chunks = [
        DocumentChunk(
            text="Python is useful.",
            document_id="doc_001",
            filename="python.pdf",
            file_type="pdf",
            chunk_index=0,
            page=1,
        ),
        DocumentChunk(
            text="Python supports ML.",
            document_id="doc_001",
            filename="python.pdf",
            file_type="pdf",
            chunk_index=1,
            page=2,
        ),
    ]

    result = service.ingest_chunks(chunks)

    assert result == 2

    embedding_service.embed_texts.assert_called_once_with(
        [
            "Python is useful.",
            "Python supports ML.",
        ]
    )

    vector_store.upsert.assert_called_once()