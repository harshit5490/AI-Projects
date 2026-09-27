from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.embeddings.model import EmbeddingModel
from app.retrieval.search import RetrievalService
from app.vector_store.chroma import ChromaVectorStore
from app.ingestion.pipeline import initialize_vector_store

from app.config import (
    CHROMA_DIR,
    DEFAULT_TOP_K,
    DOCUMENTS_DIR,
    MAX_TOP_K,
)

app = FastAPI(
    title="Semantic Search API",
    description="Semantic search using embeddings and ChromaDB.",
    version="1.0.0",
)


vector_store, embedding_model = initialize_vector_store(
    documents_directory=str(DOCUMENTS_DIR),
    persist_directory=str(CHROMA_DIR),
)

retrieval_service = RetrievalService(
    embedding_model=embedding_model,
    vector_store=vector_store,
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    category: str | None = None


class SearchResultResponse(BaseModel):
    id: str
    text: str
    source: str
    category: str
    distance: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultResponse]


@app.get("/")
def home():
    return {
        "message": "Semantic Search API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "documents": vector_store.count(),
    }


@app.post(
    "/api/v1/search",
    response_model=SearchResponse,
)
def search(request: SearchRequest):

    where = None

    if request.category:
        where = {
            "category": request.category
        }

    try:
        results = retrieval_service.search(
            query=request.query,
            top_k=request.top_k,
            where=where,
        )

        return SearchResponse(
            query=request.query,
            results=[
                SearchResultResponse(
                    id=result.id,
                    text=result.text,
                    source=result.source,
                    category=result.category,
                    distance=result.distance,
                )
                for result in results
            ],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )