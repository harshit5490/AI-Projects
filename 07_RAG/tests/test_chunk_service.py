from unittest.mock import Mock

from app.schemas.chunk import DocumentChunk
from app.services.chunk_service import ChunkService


def test_create_chunks_preserves_page_metadata():

    chunker = Mock()

    chunker.split.return_value = [
        "Python is used for machine learning.",
        "Python is also used for automation.",
    ]

    service = ChunkService(
        chunker=chunker
    )

    parsed_documents = [
        {
            "text": (
                "Python is used for machine learning. "
                "Python is also used for automation."
            ),
            "metadata": {
                "page": 3,
            },
        }
    ]

    chunks = service.create_chunks(
        parsed_documents=parsed_documents,
        document_id="doc_001",
        filename="python.pdf",
        file_type="pdf",
    )

    assert len(chunks) == 2

    assert isinstance(
        chunks[0],
        DocumentChunk,
    )

    assert chunks[0].document_id == "doc_001"
    assert chunks[0].filename == "python.pdf"
    assert chunks[0].file_type == "pdf"

    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1

    assert chunks[0].page == 3
    assert chunks[1].page == 3


def test_create_chunks_for_txt():

    chunker = Mock()

    chunker.split.return_value = [
        "Python is useful."
    ]

    service = ChunkService(
        chunker=chunker
    )

    parsed_documents = [
        {
            "text": "Python is useful.",
            "metadata": {},
        }
    ]

    chunks = service.create_chunks(
        parsed_documents=parsed_documents,
        document_id="doc_002",
        filename="notes.txt",
        file_type="txt",
    )

    assert len(chunks) == 1

    assert chunks[0].document_id == "doc_002"
    assert chunks[0].filename == "notes.txt"
    assert chunks[0].file_type == "txt"
    assert chunks[0].page is None    