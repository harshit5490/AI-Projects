from unittest.mock import Mock, patch

import pytest

from app.services.document_service import DocumentService


def test_process_pdf_document():

    ingestion_service = Mock()

    ingestion_service.ingest_chunks.return_value = 2

    chunker = Mock()

    chunker.split.return_value = [
        "Python is useful."
    ]

    document_service = DocumentService(
        ingestion_service=ingestion_service,
        chunker=chunker,
    )

    parser = Mock()

    parser.parse.return_value = [
        {
            "text": "Python is useful.",
            "metadata": {
                "page": 1,
                "file_type": "pdf",
            },
        }
    ]

    with patch(
        "app.services.document_service.DocumentParserFactory.get_parser",
        return_value=parser,
    ):

        result = document_service.process_document(
            file_path="temp/test.pdf",
            filename="test.pdf",
        )

    assert result["filename"] == "test.pdf"
    assert result["file_type"] == "pdf"
    assert result["chunk_count"] == 2
    assert "document_id" in result

    parser.parse.assert_called_once_with(
        "temp/test.pdf"
    )


def test_rejects_unsupported_file_type():

    ingestion_service = Mock()
    chunker = Mock()

    document_service = DocumentService(
        ingestion_service=ingestion_service,
        chunker=chunker,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported file type",
    ):
        document_service.process_document(
            file_path="temp/test.csv",
            filename="test.csv",
        )


def test_rejects_empty_document():

    ingestion_service = Mock()
    chunker = Mock()

    document_service = DocumentService(
        ingestion_service=ingestion_service,
        chunker=chunker,
    )

    parser = Mock()

    parser.parse.return_value = []

    with patch(
        "app.services.document_service.DocumentParserFactory.get_parser",
        return_value=parser,
    ):

        with pytest.raises(
            ValueError,
            match="does not contain readable text",
        ):
            document_service.process_document(
                file_path="temp/empty.pdf",
                filename="empty.pdf",
            )