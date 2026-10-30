import os
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.dependencies import get_document_service
from app.main import app


client = TestClient(app)


def test_upload_pdf():

    mock_document_service = Mock()

    mock_document_service.process_document.return_value = {
        "document_id": "doc-test-001",
        "filename": "test.pdf",
        "file_type": "pdf",
        "chunk_count": 3,
    }

    app.dependency_overrides[
        get_document_service
    ] = lambda: mock_document_service

    try:

        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "test.pdf",
                    b"fake pdf content",
                    "application/pdf",
                )
            },
        )

        assert response.status_code == 200

        assert response.json() == {
            "document_id": "doc-test-001",
            "filename": "test.pdf",
            "file_type": "pdf",
            "chunk_count": 3,
        }

        mock_document_service.process_document.assert_called_once()

        call_args = (
            mock_document_service
            .process_document
            .call_args
        )

        assert call_args.kwargs["filename"] == "test.pdf"

        temporary_path = call_args.kwargs["file_path"]

        assert os.path.exists(temporary_path) is False

    finally:

        app.dependency_overrides.clear()

def test_upload_unsupported_file_type():

    mock_document_service = Mock()

    app.dependency_overrides[
        get_document_service
    ] = lambda: mock_document_service

    try:

        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "test.csv",
                    b"some,data",
                    "text/csv",
                )
            },
        )

        assert response.status_code == 400

        assert response.json() == {
            "detail": "Unsupported file type: .csv"
        }

        mock_document_service.process_document.assert_not_called()

    finally:

        app.dependency_overrides.clear()


def test_upload_txt():

    mock_document_service = Mock()

    mock_document_service.process_document.return_value = {
        "document_id": "doc-test-002",
        "filename": "notes.txt",
        "file_type": "txt",
        "chunk_count": 2,
    }

    app.dependency_overrides[
        get_document_service
    ] = lambda: mock_document_service

    try:

        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "notes.txt",
                    b"Python is useful for AI engineering.",
                    "text/plain",
                )
            },
        )

        assert response.status_code == 200

        assert response.json()["filename"] == "notes.txt"

        assert response.json()["file_type"] == "txt"

    finally:

        app.dependency_overrides.clear()

def test_upload_docx():

    mock_document_service = Mock()

    mock_document_service.process_document.return_value = {
        "document_id": "doc-test-003",
        "filename": "notes.docx",
        "file_type": "docx",
        "chunk_count": 4,
    }

    app.dependency_overrides[
        get_document_service
    ] = lambda: mock_document_service

    try:

        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "notes.docx",
                    b"fake docx content",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

        assert response.status_code == 200

        assert response.json()["filename"] == "notes.docx"

        assert response.json()["file_type"] == "docx"

    finally:

        app.dependency_overrides.clear()                        