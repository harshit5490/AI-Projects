
from app.embeddings.service import EmbeddingService
from app.retrieval.query_transformer import QueryTransformer
from app.vectorstore.store import PineconeVectorStore


class Retriever:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: PineconeVectorStore,
        query_transformer: QueryTransformer | None = None,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.query_transformer = query_transformer

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float | None = 0.30,
        metadata_filter: dict | None = None,
        transform_query: bool = False,
    ) -> list[dict]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        search_query = query

        if transform_query:
            if self.query_transformer is None:
                raise ValueError(
                    "Query transformer is not configured."
                )

            search_query = self.query_transformer.transform(query)

            if not search_query.strip():
                raise ValueError(
                    "Query transformer returned an empty query."
                )

        query_embedding = self.embedding_service.embed_text(
            search_query
        )

        return self.vector_store.query_chunks(
            query_vector=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            metadata_filter=metadata_filter,
        )
