from app.chunking.semantic import SemanticChunker


def test_semantic_chunker_returns_chunks():
    text = """
    Python is a programming language.
    Python is widely used in software development.
    Machine learning models can learn patterns from data.
    Machine learning is widely used in artificial intelligence.
    """

    chunker = SemanticChunker(
        similarity_threshold=0.5,
    )

    chunks = chunker.chunk(text)

    assert len(chunks) >= 1
    assert all(chunk.strip() for chunk in chunks)