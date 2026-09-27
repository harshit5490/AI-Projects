from pathlib import Path

from app.ingestion.loader import load_documents


def test_load_documents():

    documents_path = (
        Path(__file__).parent.parent
        / "data"
        / "documents"
    )

    documents = load_documents(
        str(documents_path)
    )

    assert len(documents) == 5


def test_document_fields():

    documents_path = (
        Path(__file__).parent.parent
        / "data"
        / "documents"
    )

    documents = load_documents(
        str(documents_path)
    )

    document = documents[0]

    assert document.id
    assert document.text
    assert document.source
    assert document.category


def test_document_sources():

    documents_path = (
        Path(__file__).parent.parent
        / "data"
        / "documents"
    )

    documents = load_documents(
        str(documents_path)
    )

    sources = {
        document.source
        for document in documents
    }

    assert "python.txt" in sources
    assert "fastapi.txt" in sources
    assert "machine_learning.txt" in sources
    assert "transformers.txt" in sources
    assert "docker.txt" in sources