import chromadb
from app.models import Document


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str = "./chroma_data",
        collection_name: str = "documents",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents and embeddings must be equal."
            )

        self.collection.upsert(
            ids=[document.id for document in documents],
            documents=[document.text for document in documents],
            embeddings=embeddings,
            metadatas=[
                {
                    "source": document.source,
                    "category": document.category,
                }
                for document in documents
            ],
        )

    def count(self) -> int:
        return self.collection.count()

    def search(
    self,
    query_embedding: list[float],
    top_k: int = 3,
    where: dict | None = None,
) -> dict:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )