from pydantic import BaseModel, Field

from app.embeddings.model import EmbeddingModel
from app.vector_store.chroma import ChromaVectorStore


class SearchResult(BaseModel):
    id: str
    text: str
    source: str
    category: str
    distance: float


class RetrievalService:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: ChromaVectorStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        top_k: int = 3,
        where: dict | None = None,
    ) -> list[SearchResult]:

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self.embedding_model.encode_query(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        search_results = []

        for i, document_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]

            search_results.append(
                SearchResult(
                    id=document_id,
                    text=results["documents"][0][i],
                    source=metadata["source"],
                    category=metadata["category"],
                    distance=results["distances"][0][i],
                )
            )

        return search_results