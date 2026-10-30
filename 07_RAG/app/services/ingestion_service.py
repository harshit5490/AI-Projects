from app.embeddings.service import EmbeddingService
from app.schemas.chunk import DocumentChunk
from app.vectorstore.store import PineconeVectorStore


class IngestionService:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: PineconeVectorStore,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def ingest_chunks(
        self,
        chunks: list[DocumentChunk],
    ):
        if not chunks:
            return 0

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.embedding_service.embed_texts(
            texts
        )

        vectors = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):
            vector_id = (
                f"{chunk.document_id}"
                f"_chunk_{chunk.chunk_index:04d}"
            )

            vectors.append(
                {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "document_id": chunk.document_id,
                        "filename": chunk.filename,
                        "file_type": chunk.file_type,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        **(
                            {"page": chunk.page}
                            if chunk.page is not None
                            else {}
                        ),
                    },
                }
            )

        self.vector_store.upsert(vectors)

        return len(vectors)