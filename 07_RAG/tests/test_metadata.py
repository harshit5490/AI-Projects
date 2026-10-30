import pytest

from app.schemas.metadata import ChunkMetadata


def test_valid_metadata():

    metadata = ChunkMetadata(
        document_id="doc_001",
        filename="test.pdf",
        file_type="pdf",
        chunk_index=0,
        page=1,
    )

    assert metadata.document_id == "doc_001"
    assert metadata.filename == "test.pdf"
    assert metadata.chunk_index == 0
    assert metadata.page == 1


def test_txt_metadata_without_page():

    metadata = ChunkMetadata(
        document_id="doc_002",
        filename="notes.txt",
        file_type="txt",
        chunk_index=0,
    )

    assert metadata.page is None


def test_negative_chunk_index():

    with pytest.raises(ValueError):

        ChunkMetadata(
            document_id="doc_001",
            filename="test.pdf",
            file_type="pdf",
            chunk_index=-1,
            page=1,
        )


def test_invalid_page():

    with pytest.raises(ValueError):

        ChunkMetadata(
            document_id="doc_001",
            filename="test.pdf",
            file_type="pdf",
            chunk_index=0,
            page=0,
        )