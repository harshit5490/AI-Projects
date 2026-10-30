from unittest.mock import Mock

import pytest

from app.retrieval.query_transformer import (
    QueryTransformer,
)


def test_query_transformer_rewrites_query():

    llm_client = Mock()

    llm_client.generate.return_value = (
        "What is FastAPI and what is it used for?"
    )

    transformer = QueryTransformer(
        llm_client=llm_client
    )

    result = transformer.transform(
        "what's that api thing?"
    )

    assert (
        result
        == "What is FastAPI and what is it used for?"
    )

    llm_client.generate.assert_called_once()


def test_query_transformer_rejects_empty_query():

    llm_client = Mock()

    transformer = QueryTransformer(
        llm_client=llm_client
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        transformer.transform("")


def test_query_transformer_rejects_empty_llm_response():

    llm_client = Mock()

    llm_client.generate.return_value = ""

    transformer = QueryTransformer(
        llm_client=llm_client
    )

    with pytest.raises(
        ValueError,
        match="empty query",
    ):
        transformer.transform(
            "What is FastAPI?"
        )