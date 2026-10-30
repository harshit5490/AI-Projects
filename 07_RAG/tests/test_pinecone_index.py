from app.vectorstore.index_manager import PineconeIndexManager


def test_pinecone_index_configuration():
    manager = PineconeIndexManager(
        index_name="rag-document-index",
        dimension=384,
        metric="cosine",
    )

    assert manager.index_name == "rag-document-index"
    assert manager.dimension == 384
    assert manager.metric == "cosine"