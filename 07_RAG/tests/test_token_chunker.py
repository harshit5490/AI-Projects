from app.chunking.token import TokenChunker


MODEL_NAME = "bert-base-uncased"


def test_token_chunker_returns_chunks():
    text = """
    Artificial intelligence is transforming software engineering.
    Machine learning models can analyze large amounts of data.
    Retrieval augmented generation combines retrieval with generation.
    Vector databases help retrieve semantically relevant information.
    """

    chunker = TokenChunker(
        model_name=MODEL_NAME,
        chunk_size=20,
        chunk_overlap=5,
    )

    chunks = chunker.chunk(text)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)