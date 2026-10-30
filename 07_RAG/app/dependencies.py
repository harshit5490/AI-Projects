import os
from functools import lru_cache

from app.chunking.recursive import RecursiveCharacterChunker
from app.embeddings.service import EmbeddingService
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService
from app.vectorstore.store import PineconeVectorStore

from app.generation.context_builder import ContextBuilder
from app.generation.prompt_builder import PromptBuilder
from app.llm.gemini import GeminiClient
from app.retrieval.retriever import Retriever
from app.services.answer_generation_service import (
    AnswerGenerationService,
)


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache
def get_vector_store() -> PineconeVectorStore:
    index_name = os.getenv(
        "PINECONE_INDEX_NAME",
        "rag-document-index",
    )

    namespace = os.getenv(
        "PINECONE_NAMESPACE",
        "rag-documents",
    )

    return PineconeVectorStore(
        index_name=index_name,
        namespace=namespace,
    )


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


@lru_cache
def get_document_service() -> DocumentService:
    chunker = RecursiveCharacterChunker(
        chunk_size=500,
        chunk_overlap=50,
    )

    return DocumentService(
        ingestion_service=get_ingestion_service(),
        chunker=chunker,
    )


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


@lru_cache
def get_gemini_client() -> GeminiClient:
    return GeminiClient()


@lru_cache
def get_answer_generation_service() -> AnswerGenerationService:
    return AnswerGenerationService(
        retriever=get_retriever(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm_client=get_gemini_client(),
    )
