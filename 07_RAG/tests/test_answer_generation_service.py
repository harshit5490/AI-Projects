
from unittest.mock import Mock

import pytest

from app.services.answer_generation_service import (
    AnswerGenerationService,
)


@pytest.fixture
def service():
    retriever = Mock()
    context_builder = Mock()
    prompt_builder = Mock()
    llm_client = Mock()

    return AnswerGenerationService(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_client=llm_client,
    )


def sample_chunk():
    return {
        "id": "chunk-1",
        "score": 0.85,
        "text": "FastAPI is a Python web framework.",
        "document_id": "doc-1",
        "filename": "fastapi.txt",
        "file_type": "txt",
        "chunk_index": 0,
        "page": None,
    }


def test_rejects_empty_question(service):
    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        service.answer_question("   ")

    service.retriever.retrieve.assert_not_called()


def test_returns_fallback_when_no_chunks(service):
    service.retriever.retrieve.return_value = []

    result = service.answer_question("What is FastAPI?")

    assert result["question"] == "What is FastAPI?"
    assert "couldn't find relevant information" in result["answer"]
    assert result["sources"] == []

    service.llm_client.generate.assert_not_called()


def test_generates_answer_from_retrieved_context(service):
    chunk = sample_chunk()

    service.retriever.retrieve.return_value = [chunk]
    service.context_builder.build.return_value = (
        "[Source: fastapi.txt]\nFastAPI is a Python web framework."
    )
    service.prompt_builder.build.return_value = "grounded prompt"
    service.llm_client.generate.return_value = (
        "FastAPI is a Python web framework."
    )

    result = service.answer_question("What is FastAPI?")

    assert result["answer"] == (
        "FastAPI is a Python web framework."
    )
    assert len(result["sources"]) == 1
    assert result["sources"][0]["filename"] == "fastapi.txt"

    service.retriever.retrieve.assert_called_once()
    service.context_builder.build.assert_called_once_with([chunk])
    service.prompt_builder.build.assert_called_once()
    service.llm_client.generate.assert_called_once_with(
        "grounded prompt"
    )


def test_deduplicates_sources(service):
    chunk = sample_chunk()
    duplicate = {**chunk, "id": "chunk-1-copy"}

    service.retriever.retrieve.return_value = [chunk, duplicate]
    service.context_builder.build.return_value = "context"
    service.prompt_builder.build.return_value = "prompt"
    service.llm_client.generate.return_value = "Answer"

    result = service.answer_question("Question?")

    assert len(result["sources"]) == 1


def test_passes_retrieval_options(service):
    service.retriever.retrieve.return_value = []

    service.answer_question(
        "What is FastAPI?",
        top_k=5,
        similarity_threshold=0.4,
        metadata_filter={"filename": "fastapi.txt"},
    )

    service.retriever.retrieve.assert_called_once_with(
        query="What is FastAPI?",
        top_k=5,
        similarity_threshold=0.4,
        metadata_filter={"filename": "fastapi.txt"},
    )


def test_abstains_when_all_chunks_are_below_similarity_threshold(service):
    chunk = {
        **sample_chunk(),
        "score": 0.20,
    }
    service.retriever.retrieve.return_value = [chunk]

    result = service.answer_question(
        "What is FastAPI?",
        similarity_threshold=0.30,
    )

    assert "couldn't find relevant information" in result["answer"]
    assert result["sources"] == []

    service.context_builder.build.assert_not_called()
    service.prompt_builder.build.assert_not_called()
    service.llm_client.generate.assert_not_called()


def test_generates_answer_when_chunk_meets_similarity_threshold(service):
    chunk = {
        **sample_chunk(),
        "score": 0.35,
    }
    service.retriever.retrieve.return_value = [chunk]
    service.context_builder.build.return_value = (
        "[Source: fastapi.txt]\nFastAPI is a Python web framework."
    )
    service.prompt_builder.build.return_value = "grounded prompt"
    service.llm_client.generate.return_value = (
        "FastAPI is a Python web framework."
    )

    result = service.answer_question(
        "What is FastAPI?",
        similarity_threshold=0.35,
    )

    assert result["answer"] == (
        "FastAPI is a Python web framework."
    )
    service.llm_client.generate.assert_called_once_with(
        "grounded prompt"
    )

