import pytest

from app.chunking.recursive import RecursiveCharacterChunker


def test_recursive_chunking():
    text = (
        "Machine learning is a branch of artificial intelligence. "
        "It allows systems to learn patterns from data. "
        "Supervised learning uses labeled data."
    )

    chunker = RecursiveCharacterChunker(
        chunk_size=80,
        chunk_overlap=10,
    )

    chunks = chunker.split(text)

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk) <= 100


def test_small_text_is_not_split():
    text = "Python is a programming language."

    chunker = RecursiveCharacterChunker(
        chunk_size=100,
        chunk_overlap=10,
    )

    chunks = chunker.split(text)

    assert chunks == [text]


def test_invalid_overlap():
    with pytest.raises(ValueError):
        RecursiveCharacterChunker(
            chunk_size=100,
            chunk_overlap=100,
        )

def test_chunk_overlap_does_not_exceed_chunk_size():
    chunker = RecursiveCharacterChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    text = (
        "Python is a programming language. "
        "FastAPI is a web framework for Python. "
        "Pinecone is a vector database. "
        "RAG combines retrieval with generation. "
    ) * 5

    chunks = chunker.split(text)

    assert chunks

    for chunk in chunks:
        assert len(chunk) <= 100

def test_overlap_does_not_start_with_partial_word():
    chunker = RecursiveCharacterChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    text = (
        "Retrieval augmented generation combines information retrieval "
        "with language generation. A RAG system retrieves relevant "
        "documents and provides them as context to a language model."
    )

    chunks = chunker.split(text)

    assert len(chunks) > 1

    for chunk in chunks[1:]:
        assert chunk[0].isalnum()
        assert not chunk.startswith("evant ")
        assert not chunk.startswith("lled ")