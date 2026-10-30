
from unittest.mock import Mock

import pytest

from app.retrieval.retriever import Retriever


def create_retriever():
    embedding_service = Mock()
    vector_store = Mock()
    query_transformer = Mock()

    embedding_service.embed_text.return_value = [0.1, 0.2, 0.3]
    vector_store.query_chunks.return_value = []

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        query_transformer=query_transformer,
    )

    return (
        retriever,
        embedding_service,
        vector_store,
        query_transformer,
    )


def test_retriever_uses_original_query_by_default():
    retriever, embedding, _, transformer = create_retriever()

    retriever.retrieve("What is FastAPI?")

    embedding.embed_text.assert_called_once_with(
        "What is FastAPI?"
    )
    transformer.transform.assert_not_called()


def test_retriever_uses_transformed_query_when_enabled():
    retriever, embedding, _, transformer = create_retriever()
    transformer.transform.return_value = "What is FastAPI?"

    retriever.retrieve(
        "what's that api thing?",
        transform_query=True,
    )

    transformer.transform.assert_called_once_with(
        "what's that api thing?"
    )
    embedding.embed_text.assert_called_once_with(
        "What is FastAPI?"
    )


def test_retriever_requires_transformer_when_enabled():
    embedding = Mock()
    vector_store = Mock()

    retriever = Retriever(
        embedding_service=embedding,
        vector_store=vector_store,
    )

    with pytest.raises(
        ValueError,
        match="Query transformer is not configured",
    ):
        retriever.retrieve(
            "What is FastAPI?",
            transform_query=True,
        )


def test_retriever_rejects_empty_query():
    retriever, _, _, _ = create_retriever()

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        retriever.retrieve("   ")


def test_retriever_rejects_empty_transformed_query():
    retriever, _, _, transformer = create_retriever()
    transformer.transform.return_value = "   "

    with pytest.raises(
        ValueError,
        match="empty query",
    ):
        retriever.retrieve(
            "What is FastAPI?",
            transform_query=True,
        )
